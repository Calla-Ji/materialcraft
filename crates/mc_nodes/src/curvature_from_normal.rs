// SPDX-License-Identifier: MIT OR Apache-2.0
//! Converters / Curvature from Normal.
//!
//! Display name: "Curvature from Normal". Tooltip: "Convex edges read
//! bright, concave crevices read dark, flat areas sit at mid-gray."

use crate::shaders::full_source;
use crate::tiling::TilingMode;

const NODE_KIND: &str = "curvature_from_normal";

pub struct CurvatureFromNormalParams {
    pub intensity: f32,
    pub tiling_mode: TilingMode,
}

impl Default for CurvatureFromNormalParams {
    fn default() -> Self {
        Self {
            intensity: 4.0,
            tiling_mode: TilingMode::default(),
        }
    }
}

impl CurvatureFromNormalParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            intensity: params
                .get("intensity")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.intensity),
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
    output_key: &str,
    resolution: u32,
    params: &CurvatureFromNormalParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("curvature_from_normal: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("curvature_from_normal.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&params.intensity.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&params.tiling_mode.as_u32().to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("curvature_from_normal_params"),
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
        label: Some("curvature_from_normal_bind_group"),
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
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("curvature_from_normal_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("curvature_from_normal_pass"),
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
    use super::{dispatch, CurvatureFromNormalParams};
    use crate::height_to_normal::{dispatch as height_to_normal_dispatch, HeightToNormalParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};
    use crate::tiling::TilingMode;
    use crate::uniform_color::{dispatch as uniform_color_dispatch, UniformColorParams};

    /// Height to Normal is the natural upstream of this node in a real
    /// graph, so tests build a normal map that way rather than from a flat
    /// uniform color (which would only exercise the flat/no-op case).
    fn render(params: &CurvatureFromNormalParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "height",
            RESOLUTION,
            &PerlinParams::default(),
        );
        height_to_normal_dispatch(
            &mut gpu,
            &mut pool,
            "height",
            "normal",
            RESOLUTION,
            &HeightToNormalParams::default(),
        );
        dispatch(&mut gpu, &mut pool, "normal", "output", RESOLUTION, params);
        let texture = pool
            .texture("output")
            .expect("curvature_from_normal texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(&render(&CurvatureFromNormalParams::default()), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = CurvatureFromNormalParams {
            intensity: 6.0,
            tiling_mode: TilingMode::Both,
        };
        assert_eq!(render(&params), render(&params));
    }

    /// A perfectly flat normal map (pointing straight up everywhere) has
    /// zero curvature everywhere, so the output should sit at flat mid-gray.
    #[test]
    fn flat_normal_is_mid_gray() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        uniform_color_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &UniformColorParams {
                color: [0.5, 0.5, 1.0, 1.0],
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output",
            RESOLUTION,
            &CurvatureFromNormalParams::default(),
        );
        let pixels = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
        for px in pixels.as_chunks::<4>().0 {
            assert!(
                (px[0] as i32 - 128).abs() <= 2,
                "a flat normal map must produce flat mid-gray curvature, got {}",
                px[0]
            );
        }
    }

    /// A normal map with a deliberate asymmetry between column 0 and the
    /// last column (not a smooth/periodic signal run through Height to
    /// Normal first, which turned out to wash this signal out either by
    /// being too smooth for an 8-bit quantized difference to survive, or
    /// by saturating identically at both edges for a hard-edged pattern
    /// like Checker — confirmed by hand) — guarantees column 0's "left"
    /// neighbor genuinely differs between wrapping to the last column and
    /// clamping to itself.
    fn upload_asymmetric_normal(
        gpu: &mut mc_gpu::Context,
        pool: &mut mc_gpu::TexturePool,
        key: &str,
    ) {
        let r = RESOLUTION;
        let mut pixels = vec![128u8; (r * r * 4) as usize];
        for y in 0..r {
            let first = ((y * r) * 4) as usize;
            let last = ((y * r + r - 1) * 4) as usize;
            pixels[first] = 255;
            pixels[last] = 0;
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
    fn tiling_off_changes_the_edge_reading() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        upload_asymmetric_normal(&mut gpu, &mut pool, "normal");
        dispatch(
            &mut gpu,
            &mut pool,
            "normal",
            "wrapped",
            RESOLUTION,
            &CurvatureFromNormalParams::default(),
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "normal",
            "clamped",
            RESOLUTION,
            &CurvatureFromNormalParams {
                tiling_mode: TilingMode::Off,
                ..CurvatureFromNormalParams::default()
            },
        );
        let wrapped = gpu.read_rgba8(pool.texture("wrapped").unwrap(), RESOLUTION);
        let clamped = gpu.read_rgba8(pool.texture("clamped").unwrap(), RESOLUTION);
        assert_ne!(
            wrapped[0], clamped[0],
            "disabling tiling should change column 0's reading, since its \"left\" neighbor \
             now clamps to itself instead of wrapping to the deliberately different last column"
        );
    }
}
