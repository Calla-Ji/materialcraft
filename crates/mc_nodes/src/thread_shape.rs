// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Thread.
//!
//! Display name: "Thread". Tooltip: "A single fiber cross-section — wire
//! into Fabric Generator's Fiber input to control what its threads look
//! like."
//!
//! One shape, not tiled — the same relationship to Fabric Generator that
//! Shape has to Pattern Generator's own Shape input: a dedicated shape
//! generator feeding an arranger node, rather than the arranger owning the
//! shape itself.

use crate::shaders::full_source;

const NODE_KIND: &str = "thread_shape";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiberType {
    /// A round (or, at `roundness` 0, flat-ribbon) cross-section.
    Tube,
    /// Three overlapping strand-sized Tubes side by side — the scalloped
    /// silhouette a 3-strand twisted rope has end-on. Not a literal twist
    /// running along the thread's length — see `rope_profile`'s own doc
    /// comment in the shader for why this is the closest fit within
    /// Thread's "one cross-section, constant along its length" shape.
    Rope,
    // More kinds that fit this same "one cross-section" shape can go here
    // the way Shape grew its own kinds over time. Loop/knit-stitch shapes
    // don't fit it at all (they're not a straight thread's cross-section),
    // and are planned as their own, separate node instead.
}

impl FiberType {
    pub const NAMES: [&'static str; 2] = ["Tube", "Rope"];

    fn as_u32(self) -> u32 {
        match self {
            FiberType::Tube => 0,
            FiberType::Rope => 1,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Tube" => Some(FiberType::Tube),
            "Rope" => Some(FiberType::Rope),
            _ => None,
        }
    }
}

pub struct ThreadShapeParams {
    pub fiber_type: FiberType,
    pub thickness: f32,
    pub roundness: f32,
}

impl Default for ThreadShapeParams {
    fn default() -> Self {
        Self {
            fiber_type: FiberType::Tube,
            // Deliberately *not* the same as Fabric Generator's own fallback
            // defaults (0.85/0.6) — a freshly-connected Thread at its
            // defaults used to look pixel-identical to not connecting one at
            // all, which read as "the connection isn't doing anything" even
            // though it genuinely was (confirmed via a diagnostic test).
            // Thinner and flatter here makes the very first connection
            // visibly different without the artist needing to tweak a
            // slider first just to confirm the wiring works.
            thickness: 0.5,
            roundness: 0.3,
        }
    }
}

impl ThreadShapeParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            fiber_type: params
                .get("fiber_type")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => FiberType::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.fiber_type),
            thickness: params
                .get("thickness")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.thickness),
            roundness: params
                .get("roundness")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.roundness),
        }
    }
}

fn pack_params(resolution: u32, params: &ThreadShapeParams) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.fiber_type.as_u32().to_le_bytes());
    bytes[8..12].copy_from_slice(&params.thickness.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.roundness.to_le_bytes());
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
    use super::{dispatch, FiberType, ThreadShapeParams};
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
        let pixels = render(&ThreadShapeParams::default());
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
            roundness: 0.0,
            ..ThreadShapeParams::default()
        });
        let round = render(&ThreadShapeParams {
            roundness: 1.0,
            ..ThreadShapeParams::default()
        });
        assert_ne!(flat, round);
    }

    #[test]
    fn constant_across_the_horizontal_axis() {
        // The fiber is meant to be a bar spanning the full tile width, with
        // no variation left-to-right — Fabric Generator relies on this to
        // sample it at any arbitrary x.
        let pixels = render(&ThreadShapeParams::default());
        let r = RESOLUTION as usize;
        let row = r / 4;
        let left = pixels[(row * r) * 4];
        let right = pixels[(row * r + r - 1) * 4];
        assert_eq!(left, right);
    }

    #[test]
    fn every_fiber_type_name_round_trips() {
        for name in FiberType::NAMES {
            assert!(
                FiberType::from_name(name).is_some(),
                "{name:?} should parse"
            );
        }
        assert!(FiberType::from_name("Nonsense").is_none());
    }

    #[test]
    fn rope_looks_different_from_tube() {
        let tube = render(&ThreadShapeParams {
            fiber_type: FiberType::Tube,
            ..ThreadShapeParams::default()
        });
        let rope = render(&ThreadShapeParams {
            fiber_type: FiberType::Rope,
            ..ThreadShapeParams::default()
        });
        assert_ne!(tube, rope);
    }

    #[test]
    fn rope_is_constant_across_the_horizontal_axis() {
        // Same requirement as Tube — Fabric Generator samples at a single
        // fixed x regardless of fiber type.
        let pixels = render(&ThreadShapeParams {
            fiber_type: FiberType::Rope,
            ..ThreadShapeParams::default()
        });
        let r = RESOLUTION as usize;
        let row = r / 2;
        let left = pixels[(row * r) * 4];
        let right = pixels[(row * r + r - 1) * 4];
        assert_eq!(left, right);
    }

    /// A rope's 3 strands should show up as 3 separate bright bands with a
    /// visible dip between them near the fiber's center, not one smooth
    /// bump the way Tube reads.
    #[test]
    fn rope_has_a_dip_between_strands() {
        let pixels = render(&ThreadShapeParams {
            fiber_type: FiberType::Rope,
            thickness: 1.0,
            roundness: 1.0,
            ..ThreadShapeParams::default()
        });
        let r = RESOLUTION as usize;
        let col = r / 2;
        let center = pixels[(r / 2 * r + col) * 4] as i32;
        let between_strands = pixels[((r / 2 + r / 8) * r + col) * 4] as i32;
        let strand_peak = pixels[((r / 2 + r / 4) * r + col) * 4] as i32;
        assert!(
            between_strands < center.min(strand_peak),
            "expected a dip between strands (center={center}, between={between_strands}, \
             strand_peak={strand_peak})"
        );
    }
}
