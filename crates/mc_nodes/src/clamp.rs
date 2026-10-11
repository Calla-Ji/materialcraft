// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Adjustments / Clamp.
//!
//! Display name: "Clamp". Tooltip: "Restricts values to a [low, high] range."

use crate::shaders::full_source;

const NODE_KIND: &str = "clamp";

pub struct ClampParams {
    pub low: f32,
    pub high: f32,
}

impl Default for ClampParams {
    fn default() -> Self {
        Self {
            low: 0.0,
            high: 1.0,
        }
    }
}

impl ClampParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let float = |name: &str, default: f32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(default)
        };
        Self {
            low: float("low", defaults.low),
            high: float("high", defaults.high),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &ClampParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("clamp: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("clamp.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("clamp_params"),
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
        label: Some("clamp_bind_group"),
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
            label: Some("clamp_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("clamp_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

/// Packs params into the exact byte layout of `ClampParams` in `clamp.wgsl`
/// (resolution, low, high, padding = 16 bytes).
fn pack_params(resolution: u32, params: &ClampParams) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.low.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.high.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, ClampParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &ClampParams) -> Vec<u8> {
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
        let texture = pool.texture("output").expect("clamp texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(&render(&ClampParams::default()), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = ClampParams {
            low: 0.2,
            high: 0.8,
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same input + params produced different pixels"
        );
    }

    #[test]
    fn clamps_every_channel_into_range() {
        let params = ClampParams {
            low: 0.3,
            high: 0.6,
        };
        let pixels = render(&params);
        let low_u8 = (0.3f32 * 255.0).round() as i32;
        let high_u8 = (0.6f32 * 255.0).round() as i32;
        for px in pixels.as_chunks::<4>().0 {
            for c in 0..3 {
                let v = px[c] as i32;
                assert!(
                    v >= low_u8 - 1 && v <= high_u8 + 1,
                    "channel {v} outside [{low_u8}, {high_u8}]"
                );
            }
        }
    }
}
