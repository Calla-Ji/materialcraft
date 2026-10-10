//! Generators / Patterns / Gradient Linear.
//!
//! Display name: "Gradient Linear". Tooltip: "A left-to-right ramp from
//! black to white."

use crate::shaders::full_source;

const NODE_KIND: &str = "gradient_linear";

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
) {
    let shader_source = full_source("gradient_linear.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gradient_linear_params"),
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
        label: Some("gradient_linear_bind_group"),
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
            label: Some("gradient_linear_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gradient_linear_pass"),
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
    use super::dispatch;
    use crate::test_support::RESOLUTION;

    fn render() -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION);
        let texture = pool.texture("test").expect("gradient_linear texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    // No `assert_tiles_seamlessly` here on purpose: this is an intentionally
    // discontinuous sawtooth ramp (see the shader's doc comment) — it IS
    // periodic (f(uv+1) = f(uv)), just not seam-free, same as every other
    // texture tool's Linear Gradient. What's worth checking instead is that
    // it actually ramps as expected.

    #[test]
    fn ramps_from_black_to_white_left_to_right() {
        let pixels = render();
        let r = RESOLUTION as usize;
        let left = pixels[0];
        let right = pixels[(r - 1) * 4];
        assert!(left < 10, "left edge should be near black, got {left}");
        assert!(right > 245, "right edge should be near white, got {right}");

        // Monotonically increasing along a row.
        let row: Vec<u8> = (0..r).map(|x| pixels[x * 4]).collect();
        assert!(
            row.windows(2).all(|w| w[1] >= w[0]),
            "gradient must be monotonic"
        );
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(
            render(),
            render(),
            "same (parameterless) node produced different pixels"
        );
    }
}
