// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Uniform Color.
//!
//! Display name: "Uniform Color". Tooltip: "A flat, solid color fill."

use crate::shaders::full_source;

const NODE_KIND: &str = "uniform_color";

pub struct UniformColorParams {
    pub color: [f32; 4],
}

impl Default for UniformColorParams {
    fn default() -> Self {
        Self {
            color: [0.5, 0.5, 0.5, 1.0],
        }
    }
}

impl UniformColorParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let color = match params.get("color") {
            Some(mc_core::ParamValue::Color(c)) => *c,
            _ => defaults.color,
        };
        Self { color }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &UniformColorParams,
) {
    let shader_source = full_source("uniform_color.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform_color_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let view = pool.get_or_create(
        &gpu.device,
        key,
        resolution,
        wgpu::TextureFormat::Rgba16Float,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("uniform_color_bind_group"),
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
            label: Some("uniform_color_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("uniform_color_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

/// Packs params into the exact byte layout of `UniformColorParams` in
/// `uniform_color.wgsl` (resolution + 3 padding u32s, then a `vec4<f32>`
/// color at its required 16-byte-aligned offset = 32 bytes).
fn pack_params(resolution: u32, params: &UniformColorParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.color[0].to_le_bytes());
    bytes[20..24].copy_from_slice(&params.color[1].to_le_bytes());
    bytes[24..28].copy_from_slice(&params.color[2].to_le_bytes());
    bytes[28..32].copy_from_slice(&params.color[3].to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, UniformColorParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &UniformColorParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("uniform_color texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn tiles_seamlessly() {
        assert_tiles_seamlessly(&render(&UniformColorParams::default()), RESOLUTION);
    }

    #[test]
    fn fills_with_the_requested_color() {
        let params = UniformColorParams {
            color: [0.2, 0.4, 0.6, 1.0],
        };
        let pixels = render(&params);
        let expected = [51u8, 102, 153, 255];
        for (channel, expected) in pixels[0..4].iter().zip(expected) {
            assert!(
                (*channel as i32 - expected as i32).abs() <= 1,
                "channel {channel} not close to expected {expected}"
            );
        }
        // Every pixel should match, not just the first.
        assert!(pixels.as_chunks::<4>().0.iter().all(|p| p == &pixels[0..4]));
    }
}
