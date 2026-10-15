// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Spatial / Push Warp.
//!
//! Display name: "Push Warp". Tooltip: "Shift pixels along one fixed
//! direction, by how much an intensity map's value says at that point —
//! unlike Warp, which pushes along the intensity map's gradient instead."

use crate::shaders::full_source;
use crate::tiling::TilingMode;

const NODE_KIND: &str = "directional_warp";

pub struct DirectionalWarpParams {
    pub angle: f32,
    pub amount: f32,
    pub tiling_mode: TilingMode,
}

impl Default for DirectionalWarpParams {
    fn default() -> Self {
        Self {
            angle: 0.0,
            amount: 0.1,
            tiling_mode: TilingMode::default(),
        }
    }
}

impl DirectionalWarpParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let float = |name: &str, default: f32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(default)
        };
        Self {
            angle: float("angle", defaults.angle),
            amount: float("amount", defaults.amount),
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
    input_key: &str,
    intensity_key: &str,
    output_key: &str,
    resolution: u32,
    params: &DirectionalWarpParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("directional_warp: no texture at input key {input_key:?}"))
        .clone();
    let intensity_view = pool
        .view(intensity_key)
        .unwrap_or_else(|| {
            panic!("directional_warp: no texture at intensity key {intensity_key:?}")
        })
        .clone();

    let shader_source = full_source("directional_warp.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&params.angle.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&params.amount.to_le_bytes());
    param_bytes[12..16].copy_from_slice(&params.tiling_mode.as_u32().to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("directional_warp_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let output_view = pool.get_or_create(
        &gpu.device,
        output_key,
        resolution,
        wgpu::TextureFormat::Rgba16Float,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("directional_warp_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&input_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&intensity_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("directional_warp_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("directional_warp_pass"),
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
    use super::{dispatch, DirectionalWarpParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, tiles_seamlessly, RESOLUTION};
    use crate::tiling::TilingMode;

    fn render(params: &DirectionalWarpParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams {
                seed: 1,
                ..PerlinParams::default()
            },
        );
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "intensity",
            RESOLUTION,
            &PerlinParams {
                seed: 2,
                ..PerlinParams::default()
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "intensity",
            "output",
            RESOLUTION,
            params,
        );
        let texture = pool.texture("output").expect("directional_warp texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_inputs() {
        let params = DirectionalWarpParams {
            angle: 0.25,
            amount: 0.08,
            tiling_mode: TilingMode::Both,
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = DirectionalWarpParams {
            angle: 0.1,
            amount: 0.05,
            tiling_mode: TilingMode::Both,
        };
        assert_eq!(render(&params), render(&params));
    }

    #[test]
    fn zero_amount_is_a_no_op() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams::default(),
        );
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "intensity",
            RESOLUTION,
            &PerlinParams {
                seed: 2,
                ..PerlinParams::default()
            },
        );
        let before = gpu.read_rgba8(pool.texture("input").unwrap(), RESOLUTION);

        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "intensity",
            "output",
            RESOLUTION,
            &DirectionalWarpParams {
                angle: 0.3,
                amount: 0.0,
                tiling_mode: TilingMode::Both,
            },
        );
        let after = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        assert_eq!(before, after, "amount = 0 must not change pixels");
    }

    /// A smooth, periodic (no sawtooth) cosine bump, peaking at x=0 — the
    /// strongest possible push exactly at the edge this test cares about,
    /// rather than hoping a particular Perlin seed happens to push hard
    /// enough there (confirmed by hand that it doesn't reliably).
    fn upload_cosine_intensity(
        gpu: &mut mc_gpu::Context,
        pool: &mut mc_gpu::TexturePool,
        key: &str,
    ) {
        let r = RESOLUTION;
        let mut pixels = vec![0u8; (r * r * 4) as usize];
        for y in 0..r {
            for x in 0..r {
                let v = (std::f32::consts::TAU * x as f32 / r as f32).cos() * 0.5 + 0.5;
                let byte = (v * 255.0).round() as u8;
                let i = ((y * r + x) * 4) as usize;
                pixels[i..i + 4].copy_from_slice(&[byte, byte, byte, 255]);
            }
        }
        pool.get_or_create(&gpu.device, key, r, wgpu::TextureFormat::Rgba8Unorm);
        let texture = pool.texture(key).expect("just created above");
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(r * 4),
                rows_per_image: Some(r),
            },
            wgpu::Extent3d {
                width: r,
                height: r,
                depth_or_array_layers: 1,
            },
        );
    }

    #[test]
    fn tiling_off_is_no_longer_seamless() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams::default(),
        );
        upload_cosine_intensity(&mut gpu, &mut pool, "intensity");
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "intensity",
            "output",
            RESOLUTION,
            &DirectionalWarpParams {
                angle: 0.0,
                amount: 0.1,
                tiling_mode: TilingMode::Off,
            },
        );
        let pixels = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
        assert!(
            tiles_seamlessly(&pixels, RESOLUTION).is_err(),
            "disabling tiling should introduce a real seam"
        );
    }
}
