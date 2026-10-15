// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Spatial / Flow Blur.
//!
//! Display name: "Flow Blur". Tooltip: "Blur along a control map's local
//! gradient, strength scaled by its steepness — key for organic, eroded
//! looks."

use crate::shaders::full_source;
use crate::tiling::TilingMode;

const NODE_KIND: &str = "slope_blur";

pub struct SlopeBlurParams {
    /// In UV space (0..1), like `BlurParams::radius` — the blur radius
    /// reached at the control map's steepest point.
    pub radius: f32,
    pub tiling_mode: TilingMode,
}

impl Default for SlopeBlurParams {
    fn default() -> Self {
        Self {
            radius: 0.03,
            tiling_mode: TilingMode::default(),
        }
    }
}

impl SlopeBlurParams {
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

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    control_key: &str,
    output_key: &str,
    resolution: u32,
    params: &SlopeBlurParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("slope_blur: no texture at input key {input_key:?}"))
        .clone();
    let control_view = pool
        .view(control_key)
        .unwrap_or_else(|| panic!("slope_blur: no texture at control key {control_key:?}"))
        .clone();

    let shader_source = full_source("slope_blur.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let radius_px = params.radius * resolution as f32;
    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&radius_px.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&params.tiling_mode.as_u32().to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("slope_blur_params"),
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
        label: Some("slope_blur_bind_group"),
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
                resource: wgpu::BindingResource::TextureView(&control_view),
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
            label: Some("slope_blur_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("slope_blur_pass"),
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
    use super::{dispatch, SlopeBlurParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, max_channel_diff, pixel, RESOLUTION};
    use crate::tiling::TilingMode;
    use crate::uniform_color::{dispatch as uniform_color_dispatch, UniformColorParams};

    fn render(params: &SlopeBlurParams) -> Vec<u8> {
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
            "control",
            RESOLUTION,
            &PerlinParams {
                seed: 2,
                ..PerlinParams::default()
            },
        );
        dispatch(
            &mut gpu, &mut pool, "input", "control", "output", RESOLUTION, params,
        );
        let texture = pool.texture("output").expect("slope_blur texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_inputs() {
        assert_tiles_seamlessly(
            &render(&SlopeBlurParams {
                radius: 0.05,
                tiling_mode: TilingMode::Both,
            }),
            RESOLUTION,
        );
    }

    #[test]
    fn is_deterministic() {
        let params = SlopeBlurParams {
            radius: 0.03,
            tiling_mode: TilingMode::Both,
        };
        assert_eq!(render(&params), render(&params));
    }

    /// A perfectly flat control map has zero slope everywhere, so the input
    /// should pass through completely unblurred.
    #[test]
    fn flat_control_is_a_no_op() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams::default(),
        );
        uniform_color_dispatch(
            &mut gpu,
            &mut pool,
            "control",
            RESOLUTION,
            &UniformColorParams {
                color: [0.5, 0.5, 0.5, 1.0],
            },
        );
        let before = gpu.read_rgba8(pool.texture("input").unwrap(), RESOLUTION);

        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "control",
            "output",
            RESOLUTION,
            &SlopeBlurParams {
                radius: 0.05,
                tiling_mode: TilingMode::Both,
            },
        );
        let after = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        assert_eq!(
            before, after,
            "a flat control map must not blur the input at all"
        );
    }

    #[test]
    fn tiling_off_introduces_a_seam() {
        let wrapped = render(&SlopeBlurParams {
            radius: 0.1,
            tiling_mode: TilingMode::Both,
        });
        let clamped = render(&SlopeBlurParams {
            radius: 0.1,
            tiling_mode: TilingMode::Off,
        });
        let seam = |pixels: &[u8]| -> i32 {
            (0..RESOLUTION)
                .map(|y| {
                    max_channel_diff(
                        pixel(pixels, RESOLUTION, 0, y),
                        pixel(pixels, RESOLUTION, RESOLUTION - 1, y),
                    )
                })
                .max()
                .unwrap()
        };
        assert!(
            seam(&clamped) > seam(&wrapped),
            "disabling tiling should make the left/right edges blur independently"
        );
    }
}
