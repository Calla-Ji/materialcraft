// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Gradient Radial.
//!
//! Display name: "Gradient Radial". Tooltip: "A ramp from black at the
//! center to white at the given radius."

use crate::shaders::full_source;

const NODE_KIND: &str = "gradient_radial";

pub struct GradientRadialParams {
    pub radius: f32,
}

impl Default for GradientRadialParams {
    fn default() -> Self {
        Self { radius: 0.5 }
    }
}

impl GradientRadialParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            radius: params
                .get("radius")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.radius),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &GradientRadialParams,
) {
    let shader_source = full_source("gradient_radial.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&params.radius.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gradient_radial_params"),
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
        label: Some("gradient_radial_bind_group"),
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
            label: Some("gradient_radial_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gradient_radial_pass"),
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
    use super::{dispatch, GradientRadialParams};
    use crate::test_support::RESOLUTION;

    fn render(params: &GradientRadialParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("gradient_radial texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    // No `assert_tiles_seamlessly` here either, for the same reason as
    // Gradient Linear: it's a ramp by design, checked by shape instead.

    #[test]
    fn ramps_from_black_at_center_to_white_at_radius() {
        let params = GradientRadialParams { radius: 0.4 };
        let pixels = render(&params);
        let r = RESOLUTION as usize;
        let center = pixels[(r / 2 * r + r / 2) * 4];
        let corner = pixels[0];
        assert!(center < 20, "center should be near black, got {center}");
        assert!(
            corner > 235,
            "corner (beyond radius) should be clamped to white, got {corner}"
        );
    }

    #[test]
    fn is_deterministic() {
        let params = GradientRadialParams::default();
        assert_eq!(
            render(&params),
            render(&params),
            "same params produced different pixels"
        );
    }
}
