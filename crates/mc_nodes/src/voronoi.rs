// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Noise / Voronoi.
//!
//! Display name: "Voronoi". Tooltip: "Cellular noise — distances to a
//! scattered set of points, good for rock, stone, and cracked patterns."

use crate::shaders::full_source;

const NODE_KIND: &str = "voronoi";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoronoiMode {
    F1,
    F2,
    F2MinusF1,
    Edges,
}

impl VoronoiMode {
    pub const NAMES: [&'static str; 4] = ["F1", "F2", "F2-F1", "Edges"];

    fn as_u32(self) -> u32 {
        match self {
            VoronoiMode::F1 => 0,
            VoronoiMode::F2 => 1,
            VoronoiMode::F2MinusF1 => 2,
            VoronoiMode::Edges => 3,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "F1" => Some(VoronoiMode::F1),
            "F2" => Some(VoronoiMode::F2),
            "F2-F1" => Some(VoronoiMode::F2MinusF1),
            "Edges" => Some(VoronoiMode::Edges),
            _ => None,
        }
    }
}

pub struct VoronoiParams {
    pub scale: u32,
    pub jitter: f32,
    pub seed: u32,
    pub mode: VoronoiMode,
}

impl Default for VoronoiParams {
    fn default() -> Self {
        Self {
            scale: 4,
            jitter: 1.0,
            seed: 0,
            mode: VoronoiMode::F1,
        }
    }
}

impl VoronoiParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            scale: params
                .get("scale")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.scale),
            jitter: params
                .get("jitter")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.jitter),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
            mode: params
                .get("mode")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => VoronoiMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.mode),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &VoronoiParams,
) {
    let shader_source = full_source("voronoi.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voronoi_params"),
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
        label: Some("voronoi_bind_group"),
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
            label: Some("voronoi_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voronoi_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

fn pack_params(resolution: u32, params: &VoronoiParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.scale.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.mode.as_u32().to_le_bytes());
    bytes[12..16].copy_from_slice(&params.seed.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.jitter.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, VoronoiMode, VoronoiParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &VoronoiParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("voronoi texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn tiles_seamlessly() {
        assert_tiles_seamlessly(&render(&VoronoiParams::default()), RESOLUTION);
    }

    #[test]
    fn tiles_seamlessly_in_every_mode() {
        for mode in [
            VoronoiMode::F1,
            VoronoiMode::F2,
            VoronoiMode::F2MinusF1,
            VoronoiMode::Edges,
        ] {
            let params = VoronoiParams {
                mode,
                ..VoronoiParams::default()
            };
            assert_tiles_seamlessly(&render(&params), RESOLUTION);
        }
    }

    #[test]
    fn is_deterministic() {
        let params = VoronoiParams {
            seed: 7,
            ..VoronoiParams::default()
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same params produced different pixels"
        );
    }

    #[test]
    fn seed_changes_output() {
        let a = render(&VoronoiParams {
            seed: 1,
            ..VoronoiParams::default()
        });
        let b = render(&VoronoiParams {
            seed: 2,
            ..VoronoiParams::default()
        });
        assert_ne!(a, b, "different seeds produced identical pixels");
    }

    #[test]
    fn every_declared_mode_name_parses() {
        for name in VoronoiMode::NAMES {
            assert!(
                VoronoiMode::from_name(name).is_some(),
                "{name:?} should parse"
            );
        }
        assert!(VoronoiMode::from_name("Nonsense").is_none());
    }
}
