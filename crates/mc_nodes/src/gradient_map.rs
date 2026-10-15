// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Adjustments / Gradient Map.
//!
//! Display name: "Gradient Map". Tooltip: "Maps a Gray input through a color
//! gradient." Gray -> Color, the mirror image of Grayscale Conversion.

use crate::shaders::full_source;

const NODE_KIND: &str = "gradient_map";
/// Matches `gradient_map.wgsl`'s fixed-size `positions`/`colors` arrays.
const MAX_STOPS: usize = 8;

pub struct GradientMapParams {
    /// Always sorted by position ascending, always at least one stop —
    /// `from_graph` and `default` both guarantee this so `dispatch` never
    /// has to handle an empty gradient.
    pub stops: Vec<mc_core::GradientStop>,
}

impl Default for GradientMapParams {
    fn default() -> Self {
        Self {
            stops: vec![
                mc_core::GradientStop {
                    position: 0.0,
                    color: [0.0, 0.0, 0.0, 1.0],
                },
                mc_core::GradientStop {
                    position: 1.0,
                    color: [1.0, 1.0, 1.0, 1.0],
                },
            ],
        }
    }
}

impl GradientMapParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let mut stops = match params.get("gradient") {
            Some(mc_core::ParamValue::Gradient(stops)) if !stops.is_empty() => stops.clone(),
            _ => defaults.stops,
        };
        stops.sort_by(|a, b| a.position.total_cmp(&b.position));
        stops.truncate(MAX_STOPS);
        Self { stops }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &GradientMapParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("gradient_map: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("gradient_map.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gradient_map_params"),
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
        label: Some("gradient_map_bind_group"),
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
            label: Some("gradient_map_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gradient_map_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

/// Packs params into the exact byte layout of `GradientMapParams` in
/// `gradient_map.wgsl`: a 16-byte header (resolution, stop_count, 2 padding
/// u32s), then `positions: array<vec4<f32>, 2>` (32 bytes, 4 positions per
/// vec4), then `colors: array<vec4<f32>, 8>` (128 bytes) = 176 bytes total.
fn pack_params(resolution: u32, params: &GradientMapParams) -> [u8; 176] {
    let mut bytes = [0u8; 176];
    let stop_count = params.stops.len().clamp(1, MAX_STOPS) as u32;
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&stop_count.to_le_bytes());

    for (i, stop) in params.stops.iter().enumerate().take(MAX_STOPS) {
        let pos_offset = 16 + i * 4;
        bytes[pos_offset..pos_offset + 4].copy_from_slice(&stop.position.to_le_bytes());

        let color_offset = 48 + i * 16;
        for (c, channel) in stop.color.iter().enumerate() {
            let o = color_offset + c * 4;
            bytes[o..o + 4].copy_from_slice(&channel.to_le_bytes());
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, GradientMapParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};
    use mc_core::GradientStop;

    fn render(params: &GradientMapParams) -> Vec<u8> {
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
        let texture = pool.texture("output").expect("gradient_map texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(&render(&GradientMapParams::default()), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = GradientMapParams::default();
        assert_eq!(render(&params), render(&params));
    }

    #[test]
    fn two_stop_black_to_white_is_close_to_a_gray_broadcast() {
        let pixels = render(&GradientMapParams::default());
        for px in pixels.as_chunks::<4>().0 {
            assert!(
                px[0].abs_diff(px[1]) <= 2 && px[1].abs_diff(px[2]) <= 2,
                "a black-to-white gradient map of a Gray input should read close to \
                 grayscale, got {px:?}"
            );
        }
    }

    #[test]
    fn three_stops_hit_the_middle_color_at_its_position() {
        let params = GradientMapParams {
            stops: vec![
                GradientStop {
                    position: 0.0,
                    color: [1.0, 0.0, 0.0, 1.0],
                },
                GradientStop {
                    position: 0.5,
                    color: [0.0, 1.0, 0.0, 1.0],
                },
                GradientStop {
                    position: 1.0,
                    color: [0.0, 0.0, 1.0, 1.0],
                },
            ],
        };
        // A flat 0.5 Gray input hits the middle stop (green) exactly.
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::uniform_color::dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &crate::uniform_color::UniformColorParams {
                color: [0.5, 0.5, 0.5, 1.0],
            },
        );
        dispatch(&mut gpu, &mut pool, "input", "output", RESOLUTION, &params);
        let pixels = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
        let px = &pixels[0..4];
        assert!(px[0] < 10, "red channel should be ~0 at the green stop");
        assert!(
            px[1] > 245,
            "green channel should be ~255 at the green stop"
        );
        assert!(px[2] < 10, "blue channel should be ~0 at the green stop");
    }

    /// Guards the `stop_count` underflow this shader would hit if an empty
    /// gradient ever reached the GPU: `from_graph` and `Default` both
    /// guarantee at least one stop, so this exercises that guarantee rather
    /// than the shader itself.
    #[test]
    fn from_graph_never_produces_an_empty_gradient() {
        let params = GradientMapParams::from_graph(&std::collections::BTreeMap::new());
        assert!(!params.stops.is_empty());
    }
}
