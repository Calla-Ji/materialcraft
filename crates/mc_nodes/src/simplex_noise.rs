// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Noise / Simplex Noise.
//!
//! Display name: "Simplex Noise". Tooltip: "Gradient noise on a triangular
//! lattice — fewer directional artifacts than Perlin at a similar cost."

use crate::shaders::full_source;
use crate::tiling::TilingMode;

const NODE_KIND: &str = "simplex_noise";

pub struct SimplexParams {
    pub scale: u32,
    pub octaves: u32,
    pub persistence: f32,
    pub seed: u32,
    pub tiling_mode: TilingMode,
}

impl Default for SimplexParams {
    fn default() -> Self {
        Self {
            scale: 4,
            octaves: 4,
            persistence: 0.5,
            seed: 0,
            tiling_mode: TilingMode::default(),
        }
    }
}

impl SimplexParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
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
            persistence: params
                .get("persistence")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.persistence),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
            tiling_mode: params
                .get("tiling_mode")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => TilingMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.tiling_mode),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &SimplexParams,
) {
    let shader_source = full_source("simplex_noise.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("simplex_noise_params"),
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
        label: Some("simplex_noise_bind_group"),
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
            label: Some("simplex_noise_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("simplex_noise_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

/// Packs params into the exact byte layout of `SimplexParams` in
/// `simplex_noise.wgsl` (8 x 4-byte fields = 32 bytes).
fn pack_params(resolution: u32, params: &SimplexParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.scale.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.octaves.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.seed.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.persistence.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.tiling_mode.as_u32().to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, SimplexParams};
    use crate::test_support::{
        assert_tiles_seamlessly, tiles_seamlessly as check_tiles_seamlessly, RESOLUTION,
    };
    use crate::tiling::TilingMode;

    fn render(params: &SimplexParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("simplex texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn tiles_seamlessly() {
        assert_tiles_seamlessly(&render(&SimplexParams::default()), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = SimplexParams {
            seed: 42,
            ..SimplexParams::default()
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same params + seed produced different pixels"
        );
    }

    #[test]
    fn seed_changes_output() {
        let a = render(&SimplexParams {
            seed: 1,
            ..SimplexParams::default()
        });
        let b = render(&SimplexParams {
            seed: 2,
            ..SimplexParams::default()
        });
        assert_ne!(a, b, "different seeds produced identical pixels");
    }

    #[test]
    fn tiling_off_is_no_longer_seamless() {
        let params = SimplexParams {
            tiling_mode: TilingMode::Off,
            ..SimplexParams::default()
        };
        assert!(
            check_tiles_seamlessly(&render(&params), RESOLUTION).is_err(),
            "disabling tiling should break the forced seamless lattice wrap"
        );
    }

    #[test]
    fn horizontal_only_still_wraps_vertically_but_not_horizontally() {
        let both = render(&SimplexParams::default());
        let horizontal_only = render(&SimplexParams {
            tiling_mode: TilingMode::HorizontalOnly,
            ..SimplexParams::default()
        });
        assert_ne!(
            both, horizontal_only,
            "turning off the vertical axis alone should change the output"
        );
    }

    /// Simplex and Perlin are different algorithms (triangular vs. square
    /// lattice) — if this node were accidentally calling `perlin_2d` instead
    /// of `simplex_2d`, this test would still pass every other test here but
    /// silently not be testing what it claims to.
    #[test]
    fn differs_from_perlin_noise() {
        use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(
            &mut gpu,
            &mut pool,
            "simplex",
            RESOLUTION,
            &SimplexParams::default(),
        );
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "perlin",
            RESOLUTION,
            &PerlinParams::default(),
        );
        let simplex_pixels = gpu.read_rgba8(pool.texture("simplex").unwrap(), RESOLUTION);
        let perlin_pixels = gpu.read_rgba8(pool.texture("perlin").unwrap(), RESOLUTION);
        assert_ne!(
            simplex_pixels, perlin_pixels,
            "Simplex Noise produced byte-identical output to Perlin Noise"
        );
    }
}
