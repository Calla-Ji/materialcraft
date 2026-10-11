// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Noise / Fractal Sum.
//!
//! Display name: "Fractal Sum". Tooltip: "Several octaves of a base noise,
//! summed (fBm) — more control than Perlin Noise's built-in octaves."

use crate::shaders::full_source;

const NODE_KIND: &str = "fractal_sum";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FractalBase {
    Perlin,
    Voronoi,
}

impl FractalBase {
    pub const NAMES: [&'static str; 2] = ["Perlin", "Voronoi"];

    fn as_u32(self) -> u32 {
        match self {
            FractalBase::Perlin => 0,
            FractalBase::Voronoi => 1,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Perlin" => Some(FractalBase::Perlin),
            "Voronoi" => Some(FractalBase::Voronoi),
            _ => None,
        }
    }
}

pub struct FractalSumParams {
    pub base: FractalBase,
    pub scale: u32,
    pub octaves: u32,
    pub lacunarity: f32,
    pub gain: f32,
    pub seed: u32,
}

impl Default for FractalSumParams {
    fn default() -> Self {
        Self {
            base: FractalBase::Perlin,
            scale: 4,
            octaves: 4,
            lacunarity: 2.0,
            gain: 0.5,
            seed: 0,
        }
    }
}

impl FractalSumParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            base: params
                .get("base")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => FractalBase::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.base),
            scale: params
                .get("scale")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.scale),
            octaves: params
                .get("octaves")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.octaves),
            lacunarity: params
                .get("lacunarity")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.lacunarity),
            gain: params
                .get("gain")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.gain),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &FractalSumParams,
) {
    let shader_source = full_source("fractal_sum.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fractal_sum_params"),
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
        label: Some("fractal_sum_bind_group"),
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
            label: Some("fractal_sum_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fractal_sum_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

fn pack_params(resolution: u32, params: &FractalSumParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.base.as_u32().to_le_bytes());
    bytes[8..12].copy_from_slice(&params.scale.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.octaves.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.lacunarity.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.gain.to_le_bytes());
    bytes[24..28].copy_from_slice(&params.seed.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, FractalBase, FractalSumParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &FractalSumParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("fractal_sum texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn tiles_seamlessly_with_perlin_base() {
        let params = FractalSumParams {
            base: FractalBase::Perlin,
            ..FractalSumParams::default()
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    #[test]
    fn tiles_seamlessly_with_voronoi_base() {
        let params = FractalSumParams {
            base: FractalBase::Voronoi,
            ..FractalSumParams::default()
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    /// Non-integer lacunarity (so each octave's period must be rounded to
    /// stay tileable) is exactly the case the rounding exists for.
    #[test]
    fn tiles_seamlessly_with_fractional_lacunarity() {
        let params = FractalSumParams {
            lacunarity: 2.3,
            octaves: 5,
            ..FractalSumParams::default()
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = FractalSumParams {
            seed: 9,
            ..FractalSumParams::default()
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same params produced different pixels"
        );
    }

    #[test]
    fn base_name_round_trips() {
        for name in FractalBase::NAMES {
            assert!(
                FractalBase::from_name(name).is_some(),
                "{name:?} should parse"
            );
        }
    }
}
