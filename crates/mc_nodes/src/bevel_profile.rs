// SPDX-License-Identifier: MIT OR Apache-2.0
//! Converters / Bevel Profile.
//!
//! Turns an `Sdf` into a real height map through a chosen edge profile
//! (`docs/hardsurface_nodes/HARDSURFACE_NODES.md` §3.4) — replaces the
//! "bevel, then curve, then blur, then histogram" chain hard-surface
//! artists currently need, since the real distance to an edge is already
//! known instead of reconstructed from a blurred height map after the
//! fact.

use std::collections::BTreeMap;

use mc_core::ParamValue;

use crate::shaders::full_source;

const NODE_KIND: &str = "bevel_profile";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BevelPreset {
    Chamfer,
    Round,
    Cove,
    Ogee,
    Step,
    Lip,
    Flat,
    Custom,
}

impl BevelPreset {
    pub const NAMES: [&'static str; 8] = [
        "Chamfer", "Round", "Cove", "Ogee", "Step", "Lip", "Flat", "Custom",
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Chamfer" => Some(Self::Chamfer),
            "Round" => Some(Self::Round),
            "Cove" => Some(Self::Cove),
            "Ogee" => Some(Self::Ogee),
            "Step" => Some(Self::Step),
            "Lip" => Some(Self::Lip),
            "Flat" => Some(Self::Flat),
            "Custom" => Some(Self::Custom),
            _ => None,
        }
    }

    fn as_u32(self) -> u32 {
        match self {
            Self::Chamfer => 0,
            Self::Round => 1,
            Self::Cove => 2,
            Self::Ogee => 3,
            Self::Step => 4,
            Self::Lip => 5,
            Self::Flat => 6,
            Self::Custom => 7,
        }
    }
}

pub struct BevelProfileParams {
    pub preset: BevelPreset,
    /// How far the bevel reaches from the zero contour, tile units.
    pub width: f32,
    /// 0..1 — a vertical drop before the bevel starts (a crisp AA step).
    pub wall: f32,
    pub depth: f32,
    pub base: f32,
    /// `false` flips the sdf's sign first, bevelling outward (a moat or
    /// raised surround) instead of the normal inward case.
    pub inward: bool,
    /// `Custom` preset only: y-values of a monotone cubic through the
    /// fixed x-positions `[0, 0.2, 0.6, 1.0]` — same "fixed x, adjustable
    /// y" shape the Curve node's own v0..v4 params already use.
    pub custom_y: [f32; 4],
}

impl Default for BevelProfileParams {
    fn default() -> Self {
        Self {
            preset: BevelPreset::Round,
            width: 0.05,
            wall: 0.0,
            depth: 1.0,
            base: 0.0,
            inward: true,
            custom_y: [0.0, 0.6, 0.7, 1.0],
        }
    }
}

impl BevelProfileParams {
    pub fn from_graph(params: &BTreeMap<String, ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            preset: params
                .get("preset")
                .and_then(|v| match v {
                    ParamValue::Enum(name) => BevelPreset::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.preset),
            width: params
                .get("width")
                .and_then(ParamValue::as_float)
                .unwrap_or(defaults.width),
            wall: params
                .get("wall")
                .and_then(ParamValue::as_float)
                .unwrap_or(defaults.wall),
            depth: params
                .get("depth")
                .and_then(ParamValue::as_float)
                .unwrap_or(defaults.depth),
            base: params
                .get("base")
                .and_then(ParamValue::as_float)
                .unwrap_or(defaults.base),
            inward: params
                .get("inward")
                .and_then(ParamValue::as_bool)
                .unwrap_or(defaults.inward),
            custom_y: [
                params
                    .get("custom_y0")
                    .and_then(ParamValue::as_float)
                    .unwrap_or(defaults.custom_y[0]),
                params
                    .get("custom_y1")
                    .and_then(ParamValue::as_float)
                    .unwrap_or(defaults.custom_y[1]),
                params
                    .get("custom_y2")
                    .and_then(ParamValue::as_float)
                    .unwrap_or(defaults.custom_y[2]),
                params
                    .get("custom_y3")
                    .and_then(ParamValue::as_float)
                    .unwrap_or(defaults.custom_y[3]),
            ],
        }
    }
}

/// Where `dispatch` stores each output — `height` must stay first in the
/// `NodeDef`'s own `outputs` list (same "first declared output is the
/// default preview" convention every other multi-output node relies on).
pub struct BevelProfileOutputKeys<'a> {
    pub height: &'a str,
    pub edge: &'a str,
    pub top: &'a str,
    pub u: &'a str,
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    sdf_key: &str,
    keys: BevelProfileOutputKeys,
    resolution: u32,
    params: &BevelProfileParams,
) {
    let sdf_view = pool
        .view(sdf_key)
        .unwrap_or_else(|| panic!("bevel_profile: no texture at sdf key {sdf_key:?}"))
        .clone();

    let shader_source = full_source("bevel_profile.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bevel_profile_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    // All four outputs are plain `Gray`, not `Sdf` — no float32 precision
    // requirement (unlike Shape Boolean's `sdf` output).
    let height_view = pool
        .get_or_create(
            &gpu.device,
            keys.height,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let edge_view = pool
        .get_or_create(
            &gpu.device,
            keys.edge,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let top_view = pool
        .get_or_create(
            &gpu.device,
            keys.top,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let u_view = pool
        .get_or_create(
            &gpu.device,
            keys.u,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bevel_profile_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&sdf_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&height_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&edge_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&top_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&u_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bevel_profile_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bevel_profile_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

fn pack_params(resolution: u32, params: &BevelProfileParams) -> [u8; 48] {
    let mut bytes = [0u8; 48];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.preset.as_u32().to_le_bytes());
    bytes[8..12].copy_from_slice(&(params.inward as u32).to_le_bytes());
    bytes[12..16].copy_from_slice(&params.width.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.wall.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.depth.to_le_bytes());
    bytes[24..28].copy_from_slice(&params.base.to_le_bytes());
    bytes[28..32].copy_from_slice(&params.custom_y[0].to_le_bytes());
    bytes[32..36].copy_from_slice(&params.custom_y[1].to_le_bytes());
    bytes[36..40].copy_from_slice(&params.custom_y[2].to_le_bytes());
    bytes[40..44].copy_from_slice(&params.custom_y[3].to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{BevelPreset, BevelProfileOutputKeys, BevelProfileParams};
    use crate::test_support::{golden_path, read_npy_f32_2d, read_png16_as_unit_floats};

    const RES: u32 = 512;

    fn load_shape_a(gpu: &mc_gpu::Context, pool: &mut mc_gpu::TexturePool) {
        let a = read_npy_f32_2d(&golden_path("hardsurface", "shape_a.npy"));
        let pixels: Vec<[f32; 4]> = a.iter().map(|&d| [d, d, d, 1.0]).collect();
        // Real `Sdf` inputs are `Rgba32Float` — matching Shape Boolean's own
        // test convention so this test exercises the same precision the
        // real graph would feed this node.
        gpu.write_rgba32f(pool, "sdf", RES, &pixels);
    }

    struct Outputs {
        height: Vec<f32>,
        edge: Vec<f32>,
        top: Vec<f32>,
        u: Vec<f32>,
    }

    fn run(
        gpu: &mut mc_gpu::Context,
        pool: &mut mc_gpu::TexturePool,
        params: &BevelProfileParams,
    ) -> Outputs {
        super::dispatch(
            gpu,
            pool,
            "sdf",
            BevelProfileOutputKeys {
                height: "height",
                edge: "edge",
                top: "top",
                u: "u",
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
        Outputs {
            height: read("height"),
            edge: read("edge"),
            top: read("top"),
            u: read("u"),
        }
    }

    /// HARDSURFACE_NODES.md §6: mean abs under 0.002, max under 0.02 — max
    /// may be exceeded only on 1-px AA edges, at most 0.5% of pixels.
    fn assert_matches_golden(got: &[f32], want: &[f32], label: &str) {
        let diffs: Vec<f32> = got
            .iter()
            .zip(want.iter())
            .map(|(&g, &w)| (g - w).abs())
            .collect();
        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        let exceeding = diffs.iter().filter(|&&d| d > 0.02).count();
        let frac = exceeding as f32 / diffs.len() as f32;
        assert!(mean < 0.002, "{label}: mean abs diff {mean} exceeds 0.002");
        assert!(
            frac < 0.005,
            "{label}: {frac} of pixels exceed 0.02 abs diff (limit 0.005)"
        );
    }

    #[test]
    fn matches_every_preset_golden() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        load_shape_a(&gpu, &mut pool);

        for (preset, name) in [
            (BevelPreset::Chamfer, "chamfer"),
            (BevelPreset::Round, "round"),
            (BevelPreset::Cove, "cove"),
            (BevelPreset::Ogee, "ogee"),
            (BevelPreset::Step, "step"),
            (BevelPreset::Lip, "lip"),
            (BevelPreset::Flat, "flat"),
        ] {
            let out = run(
                &mut gpu,
                &mut pool,
                &BevelProfileParams {
                    preset,
                    width: 0.05,
                    wall: 0.0,
                    depth: 1.0,
                    base: 0.0,
                    inward: true,
                    ..BevelProfileParams::default()
                },
            );
            let (want, _) = read_png16_as_unit_floats(&golden_path(
                "hardsurface",
                &format!("bevel_{name}.png"),
            ));
            assert_matches_golden(&out.height, &want, name);
        }
    }

    #[test]
    fn wall_parameter_matches_golden() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        load_shape_a(&gpu, &mut pool);

        let out = run(
            &mut gpu,
            &mut pool,
            &BevelProfileParams {
                preset: BevelPreset::Round,
                width: 0.05,
                wall: 0.35,
                depth: 1.0,
                base: 0.0,
                inward: true,
                ..BevelProfileParams::default()
            },
        );
        let (want, _) =
            read_png16_as_unit_floats(&golden_path("hardsurface", "bevel_round_wall35.png"));
        assert_matches_golden(&out.height, &want, "round_wall35");
    }

    #[test]
    fn custom_preset_matches_golden() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        load_shape_a(&gpu, &mut pool);

        let out = run(
            &mut gpu,
            &mut pool,
            &BevelProfileParams {
                preset: BevelPreset::Custom,
                width: 0.05,
                wall: 0.0,
                depth: 1.0,
                base: 0.0,
                inward: true,
                custom_y: [0.0, 0.6, 0.7, 1.0],
            },
        );
        let (want, _) = read_png16_as_unit_floats(&golden_path("hardsurface", "bevel_custom.png"));
        assert_matches_golden(&out.height, &want, "custom");
    }

    /// `edge` is AA coverage gated to the bevel band, `top` is a strict
    /// binary mask of the flat top, and the two are mutually exclusive by
    /// construction (`u < 1.0` vs `u >= 1.0`) — checked against the real
    /// `round` golden run above, across every pixel, not just a few
    /// hand-picked ones.
    #[test]
    fn edge_and_top_masks_are_consistent_with_u() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        load_shape_a(&gpu, &mut pool);
        let sdf = read_npy_f32_2d(&golden_path("hardsurface", "shape_a.npy"));

        let out = run(
            &mut gpu,
            &mut pool,
            &BevelProfileParams {
                preset: BevelPreset::Round,
                width: 0.05,
                wall: 0.0,
                depth: 1.0,
                base: 0.0,
                inward: true,
                ..BevelProfileParams::default()
            },
        );

        let res_f = RES as f32;
        for (i, (((&u, &top), &edge), &sdf_v)) in out
            .u
            .iter()
            .zip(out.top.iter())
            .zip(out.edge.iter())
            .zip(sdf.iter())
            .enumerate()
        {
            assert!((0.0..=1.0).contains(&u), "u out of [0,1] at {i}: {u}");
            assert!(
                top == 0.0 || top == 1.0,
                "top not strictly binary at {i}: {top}"
            );
            let expect_top = u >= 1.0;
            assert_eq!(top == 1.0, expect_top, "top/u mismatch at {i}");

            let cov = (0.5 - sdf_v * res_f).clamp(0.0, 1.0);
            let expect_edge = if u < 1.0 { cov } else { 0.0 };
            assert!(
                (edge - expect_edge).abs() < 1e-3,
                "edge mismatch at {i}: got {edge} want {expect_edge}"
            );
        }
    }

    /// HARDSURFACE_NODES.md §3.4: `inward = false` bevels outward instead
    /// — a real, periodic Sdf (not a golden, built the same way
    /// `distance_field.rs`'s own `sign_is_negative_inside_positive_outside`
    /// test does) should flip which side of the zero contour reads as the
    /// "flat top" (`u >= 1`).
    #[test]
    fn inward_false_flips_which_side_is_the_flat_top() {
        const SDF_RES: usize = 64;
        let mut mask = vec![0.0f32; SDF_RES * SDF_RES];
        let (cx, cy, r) = (32i64, 32i64, 20i64);
        for y in 0..SDF_RES {
            for x in 0..SDF_RES {
                let dx = x as i64 - cx;
                let dy = y as i64 - cy;
                if dx * dx + dy * dy <= r * r {
                    mask[y * SDF_RES + x] = 1.0;
                }
            }
        }
        let sdf = crate::distance_field::compute_distance_field(
            &mask,
            SDF_RES,
            &crate::distance_field::DistanceFieldParams::default(),
        );
        let pixels: Vec<[f32; 4]> = sdf.iter().map(|&d| [d, d, d, 1.0]).collect();

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        gpu.write_rgba32f(&mut pool, "sdf", SDF_RES as u32, &pixels);

        // Deep inside the disc (well past `width`) and deep outside it
        // (near a corner, far from the disc), by index into the flat array.
        let inside_idx = cy as usize * SDF_RES + cx as usize;
        let outside_idx = 2 * SDF_RES + 2;
        assert!(sdf[inside_idx] < -0.05, "test setup: should be deep inside");
        assert!(
            sdf[outside_idx] > 0.05,
            "test setup: should be deep outside"
        );

        let base_params = BevelProfileParams {
            preset: BevelPreset::Round,
            width: 0.02,
            wall: 0.0,
            depth: 1.0,
            base: 0.0,
            ..BevelProfileParams::default()
        };

        let inward_out = {
            let mut params = BevelProfileParams {
                inward: true,
                ..BevelProfileParams::default()
            };
            params.preset = base_params.preset;
            params.width = base_params.width;
            super::dispatch(
                &mut gpu,
                &mut pool,
                "sdf",
                BevelProfileOutputKeys {
                    height: "h",
                    edge: "e",
                    top: "t",
                    u: "u_in",
                },
                SDF_RES as u32,
                &params,
            );
            gpu.read_rgba16f(pool.texture("u_in").unwrap(), SDF_RES as u32)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| p[0])
                .collect::<Vec<f32>>()
        };
        let outward_out = {
            let mut params = BevelProfileParams {
                inward: false,
                ..BevelProfileParams::default()
            };
            params.preset = base_params.preset;
            params.width = base_params.width;
            super::dispatch(
                &mut gpu,
                &mut pool,
                "sdf",
                BevelProfileOutputKeys {
                    height: "h",
                    edge: "e",
                    top: "t",
                    u: "u_out",
                },
                SDF_RES as u32,
                &params,
            );
            gpu.read_rgba16f(pool.texture("u_out").unwrap(), SDF_RES as u32)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| p[0])
                .collect::<Vec<f32>>()
        };

        assert!(
            inward_out[inside_idx] > 0.9 && inward_out[outside_idx] < 0.1,
            "inward=true: inside should be the flat top, outside should not \
             (got inside={}, outside={})",
            inward_out[inside_idx],
            inward_out[outside_idx]
        );
        assert!(
            outward_out[inside_idx] < 0.1 && outward_out[outside_idx] > 0.9,
            "inward=false: outside should be the flat top, inside should not \
             (got inside={}, outside={})",
            outward_out[inside_idx],
            outward_out[outside_idx]
        );
    }

    #[test]
    fn from_graph_round_trips() {
        let mut params = std::collections::BTreeMap::new();
        params.insert(
            "preset".to_string(),
            mc_core::ParamValue::Enum("Chamfer".to_string()),
        );
        params.insert("width".to_string(), mc_core::ParamValue::Float(0.1));
        params.insert("wall".to_string(), mc_core::ParamValue::Float(0.2));
        params.insert("depth".to_string(), mc_core::ParamValue::Float(0.5));
        params.insert("base".to_string(), mc_core::ParamValue::Float(0.1));
        params.insert("inward".to_string(), mc_core::ParamValue::Bool(false));
        params.insert("custom_y0".to_string(), mc_core::ParamValue::Float(0.1));
        params.insert("custom_y1".to_string(), mc_core::ParamValue::Float(0.2));
        params.insert("custom_y2".to_string(), mc_core::ParamValue::Float(0.3));
        params.insert("custom_y3".to_string(), mc_core::ParamValue::Float(0.4));
        let parsed = BevelProfileParams::from_graph(&params);
        assert_eq!(parsed.preset, BevelPreset::Chamfer);
        assert_eq!(parsed.width, 0.1);
        assert_eq!(parsed.wall, 0.2);
        assert_eq!(parsed.depth, 0.5);
        assert_eq!(parsed.base, 0.1);
        assert!(!parsed.inward);
        assert_eq!(parsed.custom_y, [0.1, 0.2, 0.3, 0.4]);

        let defaults = BevelProfileParams::from_graph(&std::collections::BTreeMap::new());
        assert_eq!(defaults.preset, BevelPreset::Round);
        assert!(defaults.inward);
    }
}
