//! Filters / Spatial / Directional Blur.
//!
//! Display name: "Directional Blur". Tooltip: "A motion-blur-like smear
//! along one fixed direction, rather than Blur's even spread."

use crate::shaders::full_source;

const NODE_KIND: &str = "directional_blur";

pub struct DirectionalBlurParams {
    /// In UV space (0..1), like `BlurParams::radius`.
    pub radius: f32,
    /// Fraction of a full turn.
    pub angle: f32,
}

impl Default for DirectionalBlurParams {
    fn default() -> Self {
        Self {
            radius: 0.02,
            angle: 0.0,
        }
    }
}

impl DirectionalBlurParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let float = |name: &str, default: f32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(default)
        };
        Self {
            radius: float("radius", defaults.radius),
            angle: float("angle", defaults.angle),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &DirectionalBlurParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("directional_blur: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("directional_blur.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let radius_px = params.radius * resolution as f32;
    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&params.angle.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&radius_px.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("directional_blur_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let output_view = pool.get_or_create(
        &gpu.device,
        output_key,
        resolution,
        wgpu::TextureFormat::Rgba8Unorm,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("directional_blur_bind_group"),
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
            label: Some("directional_blur_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("directional_blur_pass"),
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
    use super::{dispatch, DirectionalBlurParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &DirectionalBlurParams) -> Vec<u8> {
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
        let texture = pool.texture("output").expect("directional_blur texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_input() {
        let params = DirectionalBlurParams {
            radius: 0.04,
            angle: 0.2,
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = DirectionalBlurParams {
            radius: 0.03,
            angle: 0.1,
        };
        assert_eq!(render(&params), render(&params));
    }

    /// A blur along 0 degrees (pure horizontal) should smooth horizontal
    /// pixel-to-pixel variation far more than vertical — otherwise this
    /// isn't actually *directional*.
    #[test]
    fn horizontal_blur_smooths_rows_more_than_columns() {
        let pixels = render(&DirectionalBlurParams {
            radius: 0.06,
            angle: 0.0,
        });
        let r = RESOLUTION as usize;
        let row_diff: i64 = (0..r)
            .map(|y| {
                (0..r - 1)
                    .map(|x| {
                        (pixels[(y * r + x) * 4] as i64 - pixels[(y * r + x + 1) * 4] as i64).abs()
                    })
                    .sum::<i64>()
            })
            .sum();
        let col_diff: i64 = (0..r)
            .map(|x| {
                (0..r - 1)
                    .map(|y| {
                        (pixels[(y * r + x) * 4] as i64 - pixels[((y + 1) * r + x) * 4] as i64)
                            .abs()
                    })
                    .sum::<i64>()
            })
            .sum();
        assert!(
            row_diff < col_diff,
            "a horizontal directional blur should smooth left-right variation \
             ({row_diff}) more than top-bottom variation ({col_diff})"
        );
    }
}
