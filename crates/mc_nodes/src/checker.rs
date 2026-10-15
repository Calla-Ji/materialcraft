// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Checker.
//!
//! Display name: "Checker". Tooltip: "An alternating black/white grid."

use crate::shaders::full_source;

const NODE_KIND: &str = "checker";

pub struct CheckerParams {
    pub scale: i32,
}

impl Default for CheckerParams {
    fn default() -> Self {
        Self { scale: 8 }
    }
}

impl CheckerParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            scale: params
                .get("scale")
                .and_then(mc_core::ParamValue::as_int)
                .unwrap_or(defaults.scale),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &CheckerParams,
) {
    let shader_source = full_source("checker.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    // Even-ize the cell count before it ever reaches the shader: an odd
    // count flips the checker's parity exactly at the wrap edge, breaking
    // seamless tiling (same reasoning as Pattern Generator's "rows must be
    // even" — see its own doc comment).
    let even_scale = params.scale + (params.scale & 1);
    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&even_scale.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("checker_params"),
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
        label: Some("checker_bind_group"),
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
            label: Some("checker_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("checker_pass"),
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
    use super::{dispatch, CheckerParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &CheckerParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("checker texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn tiles_seamlessly() {
        assert_tiles_seamlessly(&render(&CheckerParams::default()), RESOLUTION);
    }

    /// An odd requested scale must still tile — `dispatch` rounds it up to
    /// even before it ever reaches the shader.
    #[test]
    fn odd_scale_still_tiles_seamlessly() {
        assert_tiles_seamlessly(&render(&CheckerParams { scale: 7 }), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = CheckerParams { scale: 6 };
        assert_eq!(render(&params), render(&params));
    }

    #[test]
    fn output_is_pure_black_and_white() {
        let pixels = render(&CheckerParams::default());
        for px in pixels.as_chunks::<4>().0 {
            assert!(
                (px[0] == 0 || px[0] == 255) && px[0] == px[1] && px[1] == px[2],
                "checker must be pure black or white, broadcast to every channel"
            );
        }
    }
}
