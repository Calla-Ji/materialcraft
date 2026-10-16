// SPDX-License-Identifier: MIT OR Apache-2.0
//! Converters / Shape Boolean.
//!
//! Combines two SDFs with a real boolean (union/subtract/intersect) in one
//! of five blend modes (`docs/hardsurface_nodes/HARDSURFACE_NODES.md` §3.3)
//! — ported from the spec's own formulas (iq's smooth minimum; hg_sdf's
//! round/chamfer/groove operators, MIT-licensed, see `THIRD_PARTY.md`), not
//! copied shader code.

use std::collections::BTreeMap;

use mc_core::ParamValue;

use crate::shaders::full_source;

const NODE_KIND: &str = "shape_boolean";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    Union,
    Subtract,
    Intersect,
}

impl BoolOp {
    pub const NAMES: [&'static str; 3] = ["Union", "Subtract", "Intersect"];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Union" => Some(Self::Union),
            "Subtract" => Some(Self::Subtract),
            "Intersect" => Some(Self::Intersect),
            _ => None,
        }
    }

    fn as_u32(self) -> u32 {
        match self {
            Self::Union => 0,
            Self::Subtract => 1,
            Self::Intersect => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolMode {
    Hard,
    Smooth,
    Round,
    Chamfer,
    /// Cuts a groove of width `radius` along B's outline into A, `groove_depth`
    /// deep — ignores `op` entirely (same as the reference): there's only
    /// one sensible direction for a groove, not a union/subtract/intersect
    /// choice.
    Groove,
}

impl BoolMode {
    pub const NAMES: [&'static str; 5] = ["Hard", "Smooth", "Round", "Chamfer", "Groove"];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Hard" => Some(Self::Hard),
            "Smooth" => Some(Self::Smooth),
            "Round" => Some(Self::Round),
            "Chamfer" => Some(Self::Chamfer),
            "Groove" => Some(Self::Groove),
            _ => None,
        }
    }

    fn as_u32(self) -> u32 {
        match self {
            Self::Hard => 0,
            Self::Smooth => 1,
            Self::Round => 2,
            Self::Chamfer => 3,
            Self::Groove => 4,
        }
    }
}

pub struct ShapeBooleanParams {
    pub op: BoolOp,
    pub mode: BoolMode,
    /// Blend size (Smooth/Round/Chamfer) or groove width (Groove), tile
    /// units.
    pub radius: f32,
    /// Groove mode only: how far the groove cuts into A.
    pub groove_depth: f32,
    /// Rebuilds an exact field from the result's own zero contour after the
    /// GPU pass above — booleans built from min/max keep the inputs' hidden
    /// interior edges, which a later Bevel would otherwise show as a false
    /// groove where two shapes meet (HARDSURFACE_NODES.md §3.2). Only the
    /// `Sdf` output is affected; the four masks come from the raw `A`/`B`
    /// sign comparison either way.
    pub exact: bool,
}

impl Default for ShapeBooleanParams {
    fn default() -> Self {
        Self {
            op: BoolOp::Union,
            mode: BoolMode::Hard,
            radius: 0.02,
            groove_depth: 0.02,
            exact: true,
        }
    }
}

impl ShapeBooleanParams {
    pub fn from_graph(params: &BTreeMap<String, ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            op: params
                .get("op")
                .and_then(|v| match v {
                    ParamValue::Enum(name) => BoolOp::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.op),
            mode: params
                .get("mode")
                .and_then(|v| match v {
                    ParamValue::Enum(name) => BoolMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.mode),
            radius: params
                .get("radius")
                .and_then(ParamValue::as_float)
                .unwrap_or(defaults.radius),
            groove_depth: params
                .get("groove_depth")
                .and_then(ParamValue::as_float)
                .unwrap_or(defaults.groove_depth),
            exact: params
                .get("exact")
                .and_then(ParamValue::as_bool)
                .unwrap_or(defaults.exact),
        }
    }
}

/// Where `dispatch` stores each output — `sdf` must stay first in the
/// `NodeDef`'s own `outputs` list (same "first declared output is the
/// default preview" convention every other multi-output node relies on,
/// e.g. Thread's `height`).
pub struct ShapeBooleanOutputKeys<'a> {
    pub sdf: &'a str,
    pub a_only: &'a str,
    pub b_only: &'a str,
    pub overlap: &'a str,
    pub blend: &'a str,
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    a_key: &str,
    b_key: &str,
    keys: ShapeBooleanOutputKeys,
    resolution: u32,
    params: &ShapeBooleanParams,
) {
    let a_view = pool
        .view(a_key)
        .unwrap_or_else(|| panic!("shape_boolean: no texture at A key {a_key:?}"))
        .clone();
    let b_view = pool
        .view(b_key)
        .unwrap_or_else(|| panic!("shape_boolean: no texture at B key {b_key:?}"))
        .clone();

    let shader_source = full_source("shape_boolean.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shape_boolean_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    // `Sdf` sockets are `Rgba32Float` (HARDSURFACE_NODES.md §2) — the four
    // masks below stay `Rgba16Float` like any other `Gray` output; only
    // this one needs the extra precision.
    let sdf_view = pool
        .get_or_create(
            &gpu.device,
            keys.sdf,
            resolution,
            wgpu::TextureFormat::Rgba32Float,
        )
        .clone();
    let a_only_view = pool
        .get_or_create(
            &gpu.device,
            keys.a_only,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let b_only_view = pool
        .get_or_create(
            &gpu.device,
            keys.b_only,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let overlap_view = pool
        .get_or_create(
            &gpu.device,
            keys.overlap,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let blend_view = pool
        .get_or_create(
            &gpu.device,
            keys.blend,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shape_boolean_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&a_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&b_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&sdf_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&a_only_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&b_only_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&overlap_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&blend_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shape_boolean_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shape_boolean_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));

    if params.exact {
        redistance_sdf_output(gpu, pool, keys.sdf, resolution);
    }
}

/// The `Exact` option's CPU pass: reads the just-computed `Sdf` output
/// back, turns it into a coverage mask, re-runs Step 2's exact distance
/// transform, and uploads the result back over the same key — same
/// deliberate, temporary CPU round-trip `distance_field::dispatch` and
/// `redistance::dispatch` already use, not a new architecture.
fn redistance_sdf_output(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    sdf_key: &str,
    resolution: u32,
) {
    let texture = pool
        .texture(sdf_key)
        .unwrap_or_else(|| panic!("shape_boolean: no texture at sdf key {sdf_key:?}"));
    let raw = gpu.read_rgba32f(texture, resolution);
    let res_f = resolution as f32;
    let mask: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| crate::distance_field::coverage(p[0], res_f))
        .collect();
    let sdf = crate::distance_field::compute_distance_field(
        &mask,
        resolution as usize,
        &crate::distance_field::DistanceFieldParams {
            wrap: true,
            threshold: 0.5,
            max_distance: 0.0,
        },
    );
    let pixels: Vec<[f32; 4]> = sdf.iter().map(|&d| [d, d, d, 1.0]).collect();
    gpu.write_rgba32f(pool, sdf_key, resolution, &pixels);
}

fn pack_params(resolution: u32, params: &ShapeBooleanParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.op.as_u32().to_le_bytes());
    bytes[8..12].copy_from_slice(&params.mode.as_u32().to_le_bytes());
    bytes[12..16].copy_from_slice(&params.radius.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.groove_depth.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{BoolMode, BoolOp, ShapeBooleanOutputKeys, ShapeBooleanParams};
    use crate::test_support::{golden_path, read_npy_f32_2d};

    const RES: u32 = 512;

    fn load_shapes(gpu: &mc_gpu::Context, pool: &mut mc_gpu::TexturePool) {
        let a = read_npy_f32_2d(&golden_path("hardsurface", "shape_a.npy"));
        let b = read_npy_f32_2d(&golden_path("hardsurface", "shape_b.npy"));
        let to_pixels =
            |v: &[f32]| -> Vec<[f32; 4]> { v.iter().map(|&d| [d, d, d, 1.0]).collect() };
        // Real `Sdf` inputs are `Rgba32Float` — using `Rgba16Float` here
        // would quantize the goldens' own exact values before the node
        // under test ever sees them.
        gpu.write_rgba32f(pool, "a", RES, &to_pixels(&a));
        gpu.write_rgba32f(pool, "b", RES, &to_pixels(&b));
    }

    type FiveChannels = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    fn run(
        gpu: &mut mc_gpu::Context,
        pool: &mut mc_gpu::TexturePool,
        params: &ShapeBooleanParams,
    ) -> FiveChannels {
        super::dispatch(
            gpu,
            pool,
            "a",
            "b",
            ShapeBooleanOutputKeys {
                sdf: "sdf",
                a_only: "a_only",
                b_only: "b_only",
                overlap: "overlap",
                blend: "blend",
            },
            RES,
            params,
        );
        let read = |key: &str| -> Vec<f32> {
            gpu.read_rgba16f(pool.texture(key).unwrap(), RES)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| p[0])
                .collect()
        };
        let sdf: Vec<f32> = gpu
            .read_rgba32f(pool.texture("sdf").unwrap(), RES)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| p[0])
            .collect();
        (
            sdf,
            read("a_only"),
            read("b_only"),
            read("overlap"),
            read("blend"),
        )
    }

    fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
        got.iter()
            .zip(want.iter())
            .map(|(&g, &w)| (g - w).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn matches_every_op_mode_golden() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        load_shapes(&gpu, &mut pool);

        for (op, op_name) in [
            (BoolOp::Union, "union"),
            (BoolOp::Subtract, "subtract"),
            (BoolOp::Intersect, "intersect"),
        ] {
            for (mode, mode_name) in [
                (BoolMode::Hard, "hard"),
                (BoolMode::Smooth, "smooth"),
                (BoolMode::Round, "round"),
                (BoolMode::Chamfer, "chamfer"),
            ] {
                let (sdf, ..) = run(
                    &mut gpu,
                    &mut pool,
                    &ShapeBooleanParams {
                        op,
                        mode,
                        radius: 0.04,
                        groove_depth: 0.02,
                        exact: false,
                    },
                );
                let want = read_npy_f32_2d(&golden_path(
                    "hardsurface",
                    &format!("bool_{op_name}_{mode_name}.npy"),
                ));
                let diff = max_abs_diff(&sdf, &want);
                assert!(
                    diff < 1e-5,
                    "{op_name}_{mode_name}: max abs diff {diff} exceeds 1e-5 tile"
                );
            }
        }
    }

    #[test]
    fn matches_groove_golden() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        load_shapes(&gpu, &mut pool);

        let (sdf, ..) = run(
            &mut gpu,
            &mut pool,
            &ShapeBooleanParams {
                op: BoolOp::Union, // ignored by groove mode
                mode: BoolMode::Groove,
                radius: 0.01,
                groove_depth: 0.02,
                exact: false,
            },
        );
        let want = read_npy_f32_2d(&golden_path("hardsurface", "bool_groove.npy"));
        let diff = max_abs_diff(&sdf, &want);
        assert!(diff < 1e-5, "groove: max abs diff {diff} exceeds 1e-5 tile");
    }

    /// HARDSURFACE_NODES.md §3.2's own motivation for `Exact`: a hard union
    /// of two touching/overlapping shapes keeps their hidden interior
    /// edges, which a later Bevel would show as a false groove — `exact`
    /// should measurably change the raw boolean's own output.
    #[test]
    fn exact_changes_the_sdf_output() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        load_shapes(&gpu, &mut pool);

        let (raw, ..) = run(
            &mut gpu,
            &mut pool,
            &ShapeBooleanParams {
                op: BoolOp::Union,
                mode: BoolMode::Hard,
                radius: 0.0,
                groove_depth: 0.0,
                exact: false,
            },
        );
        let (exact, ..) = run(
            &mut gpu,
            &mut pool,
            &ShapeBooleanParams {
                op: BoolOp::Union,
                mode: BoolMode::Hard,
                radius: 0.0,
                groove_depth: 0.0,
                exact: true,
            },
        );
        assert_ne!(
            raw, exact,
            "Exact should change the hard union's hidden interior edge"
        );
    }

    /// Redistancing the hard union (what `exact: true` does internally)
    /// should match `redistance_union.npy` within Distance Field's own
    /// px-based tolerance — confirms the two features agree with each
    /// other, not just with themselves.
    #[test]
    fn exact_union_matches_the_redistance_golden() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        load_shapes(&gpu, &mut pool);

        let (sdf, ..) = run(
            &mut gpu,
            &mut pool,
            &ShapeBooleanParams {
                op: BoolOp::Union,
                mode: BoolMode::Hard,
                radius: 0.0,
                groove_depth: 0.0,
                exact: true,
            },
        );
        let want = read_npy_f32_2d(&golden_path("hardsurface", "redistance_union.npy"));
        let mut errs: Vec<f64> = sdf
            .iter()
            .zip(want.iter())
            .map(|(&g, &w)| ((g - w) as f64 * RES as f64).abs())
            .collect();
        let mean = errs.iter().sum::<f64>() / errs.len() as f64;
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99 = errs[(errs.len() as f64 * 0.99) as usize];
        assert!(mean < 0.05, "mean error {mean} px");
        assert!(p99 < 0.3, "p99 error {p99} px");
    }

    #[test]
    fn masks_match_raw_sign_of_inputs() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        load_shapes(&gpu, &mut pool);
        let a = read_npy_f32_2d(&golden_path("hardsurface", "shape_a.npy"));
        let b = read_npy_f32_2d(&golden_path("hardsurface", "shape_b.npy"));

        let (_, a_only, b_only, overlap, _) = run(
            &mut gpu,
            &mut pool,
            &ShapeBooleanParams {
                op: BoolOp::Union,
                mode: BoolMode::Hard,
                radius: 0.0,
                groove_depth: 0.0,
                exact: false,
            },
        );
        for i in 0..a.len() {
            let (ia, ib) = (a[i] < 0.0, b[i] < 0.0);
            assert_eq!(a_only[i] > 0.5, ia && !ib, "a_only mismatch at {i}");
            assert_eq!(b_only[i] > 0.5, ib && !ia, "b_only mismatch at {i}");
            assert_eq!(overlap[i] > 0.5, ia && ib, "overlap mismatch at {i}");
        }
    }

    #[test]
    fn blend_mask_is_empty_for_hard_mode_and_nonempty_for_smooth() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        load_shapes(&gpu, &mut pool);

        let (_, _, _, _, blend_hard) = run(
            &mut gpu,
            &mut pool,
            &ShapeBooleanParams {
                op: BoolOp::Union,
                mode: BoolMode::Hard,
                radius: 0.04,
                groove_depth: 0.0,
                exact: false,
            },
        );
        assert!(
            blend_hard.iter().all(|&v| v < 0.5),
            "hard mode's result equals the hard comparison everywhere, so blend should be empty"
        );

        let (_, _, _, _, blend_smooth) = run(
            &mut gpu,
            &mut pool,
            &ShapeBooleanParams {
                op: BoolOp::Union,
                mode: BoolMode::Smooth,
                radius: 0.04,
                groove_depth: 0.0,
                exact: false,
            },
        );
        assert!(
            blend_smooth.iter().any(|&v| v > 0.5),
            "smooth mode should mark some pixels changed by the fillet"
        );
    }

    #[test]
    fn from_graph_round_trips() {
        let mut params = std::collections::BTreeMap::new();
        params.insert(
            "op".to_string(),
            mc_core::ParamValue::Enum("Subtract".to_string()),
        );
        params.insert(
            "mode".to_string(),
            mc_core::ParamValue::Enum("Round".to_string()),
        );
        params.insert("radius".to_string(), mc_core::ParamValue::Float(0.05));
        params.insert("groove_depth".to_string(), mc_core::ParamValue::Float(0.1));
        params.insert("exact".to_string(), mc_core::ParamValue::Bool(false));
        let parsed = ShapeBooleanParams::from_graph(&params);
        assert_eq!(parsed.op, BoolOp::Subtract);
        assert_eq!(parsed.mode, BoolMode::Round);
        assert_eq!(parsed.radius, 0.05);
        assert_eq!(parsed.groove_depth, 0.1);
        assert!(!parsed.exact);

        let defaults = ShapeBooleanParams::from_graph(&std::collections::BTreeMap::new());
        assert_eq!(defaults.op, BoolOp::Union);
        assert_eq!(defaults.mode, BoolMode::Hard);
        assert!(defaults.exact);
    }
}
