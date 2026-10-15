// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Adjustments / Grayscale Conversion.
//!
//! Display name: "Grayscale Conversion". Tooltip: "Converts Color to Gray by
//! luminance — the one explicit way to make that connection (CLAUDE.md §5)."

use crate::shaders::full_source;

const NODE_KIND: &str = "grayscale_conversion";

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("grayscale_conversion: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("grayscale_conversion.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let mut padded = [0u8; 16];
    padded[0..4].copy_from_slice(&resolution.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grayscale_conversion_params"),
        size: padded.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &padded);

    let output_view = pool.get_or_create(
        &gpu.device,
        output_key,
        resolution,
        wgpu::TextureFormat::Rgba16Float,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("grayscale_conversion_bind_group"),
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
            label: Some("grayscale_conversion_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("grayscale_conversion_pass"),
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
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};
    use crate::uniform_color::{dispatch as uniform_color_dispatch, UniformColorParams};

    fn render_from_color(color: [f32; 4]) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        uniform_color_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &UniformColorParams { color },
        );
        dispatch(&mut gpu, &mut pool, "input", "output", RESOLUTION);
        let texture = pool
            .texture("output")
            .expect("grayscale_conversion texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(&render_from_color([0.3, 0.6, 0.9, 1.0]), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let a = render_from_color([0.1, 0.5, 0.9, 1.0]);
        let b = render_from_color([0.1, 0.5, 0.9, 1.0]);
        assert_eq!(a, b, "same input produced different pixels");
    }

    /// Rec. 709 luminance of pure green should read much brighter than pure
    /// blue — if this were a naive average instead, they'd match (both
    /// 1/3), which would be the easiest implementation bug to make here.
    #[test]
    fn weighs_green_more_than_blue() {
        let green = render_from_color([0.0, 1.0, 0.0, 1.0]);
        let blue = render_from_color([0.0, 0.0, 1.0, 1.0]);
        assert!(
            green[0] > blue[0] * 2,
            "green ({}) should read far brighter than blue ({}) under Rec. 709 luminance",
            green[0],
            blue[0]
        );
    }

    #[test]
    fn output_broadcasts_gray_to_rgb_with_opaque_alpha() {
        let pixels = render_from_color([0.2, 0.4, 0.6, 1.0]);
        for px in pixels.as_chunks::<4>().0 {
            assert_eq!(
                px[0], px[1],
                "gray must be broadcast equally to every channel"
            );
            assert_eq!(
                px[1], px[2],
                "gray must be broadcast equally to every channel"
            );
            assert_eq!(
                px[3], 255,
                "alpha must be fully opaque regardless of input alpha"
            );
        }
    }
}
