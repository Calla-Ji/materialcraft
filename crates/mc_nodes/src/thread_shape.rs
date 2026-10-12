// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Thread.
//!
//! Display name: "Thread". Tooltip: "A single fiber's cross-section and
//! twist — wire into Fabric Generator's Thread input to control what its
//! threads look like."
//!
//! One shape, not tiled — the same relationship to Fabric Generator that
//! Shape has to Pattern Generator's own Shape input: a dedicated shape
//! generator feeding an arranger node, rather than the arranger owning the
//! shape itself.
//!
//! Redesigned around real yarn-construction references (the owner's own
//! textile reference book, not assumed from a generic "rope" mental image):
//! a twisted/plied yarn's visual identity comes from the strands' helical
//! path *along* the thread's length (the Yarn Twist S/Z-direction and
//! twist-level concepts every ply-yarn reference covers), not from its
//! cross-section silhouette alone — the first version of this node only
//! varied cross-section and, correctly, didn't read as "rope" at all. See
//! the shader's own doc comment for the periodicity requirement this
//! drives.
//!
//! A first pass at twist (each strand just orbiting the center, combined
//! with a plain `max()` of brightness) still didn't read as rope — it only
//! captured the strands' sideways motion, which for similarly-sized,
//! evenly-spaced strands produces a moire/lattice interference pattern
//! instead of distinct plies (confirmed against a screenshot, not assumed).
//! Fixed with two changes, both in the shader: strands are sized to just
//! touch their neighbors around the orbit instead of mostly overlapping,
//! and they're combined with depth-ordered occlusion (`cos` of each
//! strand's own orbit angle standing in for toward/away-from-viewer depth,
//! the other half of the same sin/cos helix projection `sin` already
//! provided for the sideways position) rather than a flat `max()` — so a
//! front strand fully hides a back one where they overlap, instead of the
//! two blending together. That occlusion step also resolved an earlier,
//! documented-and-now-stale limitation where Twist Direction had no visible
//! effect at exactly `ply_count` 2 (confirmed by test, see
//! `twist_direction_also_differs_at_two_ply_now_that_strands_occlude`).

use crate::shaders::full_source;

const NODE_KIND: &str = "thread_shape";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwistDirection {
    S,
    Z,
}

impl TwistDirection {
    pub const NAMES: [&'static str; 2] = ["S", "Z"];

    fn as_u32(self) -> u32 {
        match self {
            TwistDirection::S => 0,
            TwistDirection::Z => 1,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "S" => Some(TwistDirection::S),
            "Z" => Some(TwistDirection::Z),
            _ => None,
        }
    }
}

pub struct ThreadShapeParams {
    pub ply_count: u32,
    pub thickness: f32,
    pub roundness: f32,
    pub twist_cycles: u32,
    pub twist_direction: TwistDirection,
}

impl Default for ThreadShapeParams {
    fn default() -> Self {
        Self {
            ply_count: 1,
            thickness: 0.5,
            roundness: 0.3,
            twist_cycles: 2,
            twist_direction: TwistDirection::Z,
        }
    }
}

impl ThreadShapeParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            ply_count: params
                .get("ply_count")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.ply_count),
            thickness: params
                .get("thickness")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.thickness),
            roundness: params
                .get("roundness")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.roundness),
            twist_cycles: params
                .get("twist_cycles")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(0) as u32)
                .unwrap_or(defaults.twist_cycles),
            twist_direction: params
                .get("twist_direction")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => TwistDirection::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.twist_direction),
        }
    }
}

fn pack_params(resolution: u32, params: &ThreadShapeParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.ply_count.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.thickness.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.roundness.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.twist_cycles.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.twist_direction.as_u32().to_le_bytes());
    bytes
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &ThreadShapeParams,
) {
    let shader_source = full_source("thread_shape.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("thread_shape_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let view = pool.get_or_create(
        &gpu.device,
        key,
        resolution,
        wgpu::TextureFormat::Rgba8Unorm,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("thread_shape_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("thread_shape_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("thread_shape_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

#[cfg(test)]
mod tests {
    use super::{dispatch, ThreadShapeParams, TwistDirection};
    use crate::test_support::RESOLUTION;

    fn render(params: &ThreadShapeParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("thread_shape texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn is_deterministic() {
        let params = ThreadShapeParams::default();
        assert_eq!(render(&params), render(&params));
    }

    #[test]
    fn centerline_is_brighter_than_the_edge() {
        let pixels = render(&ThreadShapeParams {
            ply_count: 1,
            ..ThreadShapeParams::default()
        });
        let r = RESOLUTION as usize;
        let center = pixels[(r / 2 * r + r / 2) * 4];
        let edge = pixels[(0 * r + r / 2) * 4];
        assert!(
            center > edge,
            "fiber centerline ({center}) should be brighter than the tile edge ({edge})"
        );
    }

    #[test]
    fn zero_thickness_is_blank() {
        let pixels = render(&ThreadShapeParams {
            ply_count: 1,
            thickness: 0.0,
            ..ThreadShapeParams::default()
        });
        assert!(
            pixels.iter().step_by(4).all(|&v| v == 0),
            "zero thickness should produce an entirely black fiber"
        );
    }

    #[test]
    fn roundness_changes_the_profile() {
        let flat = render(&ThreadShapeParams {
            ply_count: 1,
            roundness: 0.0,
            ..ThreadShapeParams::default()
        });
        let round = render(&ThreadShapeParams {
            ply_count: 1,
            roundness: 1.0,
            ..ThreadShapeParams::default()
        });
        assert_ne!(flat, round);
    }

    #[test]
    fn single_ply_is_constant_along_its_length() {
        // Fabric Generator relies on sampling at any along-length position
        // when there's nothing to twist (a single strand has no partner to
        // orbit around).
        let pixels = render(&ThreadShapeParams {
            ply_count: 1,
            ..ThreadShapeParams::default()
        });
        let r = RESOLUTION as usize;
        let row = r / 4;
        let left = pixels[(row * r) * 4];
        let right = pixels[(row * r + r - 1) * 4];
        assert_eq!(left, right);
    }

    #[test]
    fn multi_ply_varies_along_its_length_when_twisted() {
        // The whole point of twist: unlike a single ply, a twisted 2+ ply
        // thread's cross-section at a fixed height genuinely differs as you
        // move along its length. Checks the full spread across a row
        // rather than two specific columns — an orbiting strand's period
        // for N plies is 2*PI/N, not 2*PI, so two arbitrarily-chosen
        // columns can land exactly half a strand-period apart and
        // coincidentally read identical without the pattern actually being
        // constant (hit this for real while writing this test).
        let pixels = render(&ThreadShapeParams {
            ply_count: 2,
            twist_cycles: 2,
            thickness: 0.6,
            ..ThreadShapeParams::default()
        });
        let r = RESOLUTION as usize;
        let row = r / 2 + r / 6; // off-center, where an orbiting strand sweeps through
        let row_values: Vec<u8> = (0..r).map(|x| pixels[(row * r + x) * 4]).collect();
        let min = *row_values.iter().min().unwrap();
        let max = *row_values.iter().max().unwrap();
        assert!(
            max - min > 20,
            "a twisted multi-ply thread should vary along its own length (min={min}, max={max})"
        );
    }

    #[test]
    fn zero_twist_cycles_is_constant_along_its_length_even_with_multiple_plies() {
        let pixels = render(&ThreadShapeParams {
            ply_count: 3,
            twist_cycles: 0,
            ..ThreadShapeParams::default()
        });
        let r = RESOLUTION as usize;
        let row = r / 2;
        let left = pixels[(row * r) * 4];
        let right = pixels[(row * r + r - 1) * 4];
        assert_eq!(left, right);
    }

    #[test]
    fn is_seamless_along_its_own_length() {
        // twist_cycles must be a whole number of rotations across this
        // texture's own width, or Fabric Generator's per-cell wraparound
        // sampling would show a visible seam where the twist jumps. Uses
        // the same calibrated seam-vs-interior check every other periodic
        // node's tiling test uses, not a flat small-delta threshold — a
        // sharp strand edge crossing the seam at a slightly different pixel
        // center than its interior neighbors is the same ordinary texel-
        // quantization noise every periodic generator here has, not a real
        // discontinuity.
        let pixels = render(&ThreadShapeParams {
            ply_count: 3,
            twist_cycles: 3,
            thickness: 0.7,
            ..ThreadShapeParams::default()
        });
        crate::test_support::assert_tiles_seamlessly(&pixels, RESOLUTION);
    }

    #[test]
    fn twist_direction_mirrors_the_spiral() {
        let s_twist = render(&ThreadShapeParams {
            ply_count: 3,
            twist_cycles: 2,
            twist_direction: TwistDirection::S,
            thickness: 0.6,
            ..ThreadShapeParams::default()
        });
        let z_twist = render(&ThreadShapeParams {
            ply_count: 3,
            twist_cycles: 2,
            twist_direction: TwistDirection::Z,
            thickness: 0.6,
            ..ThreadShapeParams::default()
        });
        assert_ne!(s_twist, z_twist);
    }

    /// The old plain-`max()` combine was symmetric at exactly 2 plies (see
    /// the module doc comment's now-superseded note): flipping direction
    /// just swapped which strand was which, with no "which one is on top"
    /// cue to tell the difference. The depth-ordered occlusion added to fix
    /// the moire-lattice bug happens to also resolve this — a strand's
    /// depth (`cos` of its orbit angle) and its position (`sin` of the same
    /// angle) don't swap together under a direction flip the way position
    /// alone did, so which strand occludes which now differs too. Confirmed
    /// here rather than assumed.
    #[test]
    fn twist_direction_also_differs_at_two_ply_now_that_strands_occlude() {
        let s_twist = render(&ThreadShapeParams {
            ply_count: 2,
            twist_cycles: 2,
            twist_direction: TwistDirection::S,
            thickness: 0.6,
            ..ThreadShapeParams::default()
        });
        let z_twist = render(&ThreadShapeParams {
            ply_count: 2,
            twist_cycles: 2,
            twist_direction: TwistDirection::Z,
            thickness: 0.6,
            ..ThreadShapeParams::default()
        });
        assert_ne!(s_twist, z_twist);
    }

    #[test]
    fn every_twist_direction_name_round_trips() {
        for name in TwistDirection::NAMES {
            assert!(
                TwistDirection::from_name(name).is_some(),
                "{name:?} should parse"
            );
        }
        assert!(TwistDirection::from_name("Nonsense").is_none());
    }
}
