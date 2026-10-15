// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Spatial / Blur.
//!
//! Display name: "Blur". Tooltip: "Gaussian blur — softens detail. Wrap-
//! addressed so it stays tileable."

use crate::shaders::full_source;
use crate::tiling::TilingMode;

const NODE_KIND: &str = "blur";

pub struct BlurParams {
    /// In UV space (0..1), per CLAUDE.md §6.2 — converted to a pixel radius
    /// at dispatch time since the shader's sampling loop needs texel steps.
    pub radius: f32,
    pub tiling_mode: TilingMode,
}

impl Default for BlurParams {
    fn default() -> Self {
        Self {
            radius: 0.01,
            tiling_mode: TilingMode::default(),
        }
    }
}

impl BlurParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            radius: params
                .get("radius")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.radius),
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

/// Runs the separable blur pass twice (horizontal into a temp texture, then
/// vertical from the temp texture into `output_key`) to get a full 2D blur.
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &BlurParams,
) {
    let radius_px = params.radius * resolution as f32;
    let temp_key = format!("{output_key}__blur_temp");

    dispatch_pass(
        gpu,
        pool,
        input_key,
        &temp_key,
        resolution,
        0,
        radius_px,
        params.tiling_mode.wrap_x(),
    );
    dispatch_pass(
        gpu,
        pool,
        &temp_key,
        output_key,
        resolution,
        1,
        radius_px,
        params.tiling_mode.wrap_y(),
    );
}

#[allow(clippy::too_many_arguments)]
fn dispatch_pass(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    direction: u32,
    radius_px: f32,
    wrap: bool,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("blur: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("blur.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&direction.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&radius_px.to_le_bytes());
    param_bytes[12..16].copy_from_slice(&(wrap as u32).to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("blur_params"),
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
        label: Some("blur_bind_group"),
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
            label: Some("blur_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("blur_pass"),
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
    use super::{dispatch, BlurParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, max_channel_diff, pixel, RESOLUTION};
    use crate::tiling::TilingMode;

    /// The largest left/right-edge mismatch anywhere along the seam —
    /// `assert_tiles_seamlessly` checks this stays within the interior
    /// baseline; these tests check the opposite, that it doesn't.
    fn left_right_seam_diff(pixels: &[u8]) -> i32 {
        (0..RESOLUTION)
            .map(|y| {
                max_channel_diff(
                    pixel(pixels, RESOLUTION, 0, y),
                    pixel(pixels, RESOLUTION, RESOLUTION - 1, y),
                )
            })
            .max()
            .unwrap()
    }

    fn top_bottom_seam_diff(pixels: &[u8]) -> i32 {
        (0..RESOLUTION)
            .map(|x| {
                max_channel_diff(
                    pixel(pixels, RESOLUTION, x, 0),
                    pixel(pixels, RESOLUTION, x, RESOLUTION - 1),
                )
            })
            .max()
            .unwrap()
    }

    fn render(params: &BlurParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams::default(),
        );
        dispatch(&mut gpu, &mut pool, "input", "output", RESOLUTION, params);
        let texture = pool.texture("output").expect("blur texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    /// Wrap-addressed sampling is what keeps a blur of a tileable input
    /// tileable — without it, edge pixels would blend in black/garbage from
    /// outside the texture.
    #[test]
    fn preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(
            &render(&BlurParams {
                radius: 0.03,
                tiling_mode: TilingMode::Both,
            }),
            RESOLUTION,
        );
    }

    #[test]
    fn is_deterministic() {
        let params = BlurParams {
            radius: 0.02,
            tiling_mode: TilingMode::Both,
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same input + params produced different pixels"
        );
    }

    #[test]
    fn increases_smoothness() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams::default(),
        );
        let before = gpu.read_rgba8(pool.texture("input").unwrap(), RESOLUTION);

        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output",
            RESOLUTION,
            &BlurParams {
                radius: 0.05,
                tiling_mode: TilingMode::Both,
            },
        );
        let after = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        let roughness = |pixels: &[u8]| -> i64 {
            let r = RESOLUTION as usize;
            let mut total = 0i64;
            for y in 0..r {
                for x in 0..r - 1 {
                    let a = pixels[(y * r + x) * 4] as i64;
                    let b = pixels[(y * r + x + 1) * 4] as i64;
                    total += (a - b).abs();
                }
            }
            total
        };
        assert!(
            roughness(&after) < roughness(&before),
            "blurred output should be smoother (less pixel-to-pixel variation) than the input"
        );
    }

    #[test]
    fn horizontal_tiling_off_introduces_a_left_right_seam() {
        let wrapped = render(&BlurParams {
            radius: 0.1,
            tiling_mode: TilingMode::Both,
        });
        let clamped = render(&BlurParams {
            radius: 0.1,
            tiling_mode: TilingMode::VerticalOnly,
        });
        assert!(
            left_right_seam_diff(&clamped) > left_right_seam_diff(&wrapped),
            "disabling horizontal tiling should make the left/right edges blur independently \
             instead of blending across the seam"
        );
    }

    #[test]
    fn vertical_tiling_off_introduces_a_top_bottom_seam() {
        let wrapped = render(&BlurParams {
            radius: 0.1,
            tiling_mode: TilingMode::Both,
        });
        let clamped = render(&BlurParams {
            radius: 0.1,
            tiling_mode: TilingMode::HorizontalOnly,
        });
        assert!(
            top_bottom_seam_diff(&clamped) > top_bottom_seam_diff(&wrapped),
            "disabling vertical tiling should make the top/bottom edges blur independently \
             instead of blending across the seam"
        );
    }
}
