//! Generators / Noise / White Noise.
//!
//! Display name: "White Noise". Tooltip: "Fully uncorrelated random cells —
//! no smooth interpolation, unlike Perlin or Simplex."

use crate::shaders::full_source;

const NODE_KIND: &str = "white_noise";

pub struct WhiteNoiseParams {
    pub scale: u32,
    pub seed: u32,
}

impl Default for WhiteNoiseParams {
    fn default() -> Self {
        Self { scale: 16, seed: 0 }
    }
}

impl WhiteNoiseParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            scale: params
                .get("scale")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.scale),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &WhiteNoiseParams,
) {
    let shader_source = full_source("white_noise.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&params.scale.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&params.seed.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("white_noise_params"),
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
        label: Some("white_noise_bind_group"),
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
            label: Some("white_noise_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("white_noise_pass"),
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
    use super::{dispatch, WhiteNoiseParams};
    use crate::test_support::{assert_tiles_seamlessly, pixel, RESOLUTION};

    fn render(params: &WhiteNoiseParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("white_noise texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn tiles_seamlessly() {
        assert_tiles_seamlessly(&render(&WhiteNoiseParams::default()), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = WhiteNoiseParams { scale: 8, seed: 7 };
        assert_eq!(render(&params), render(&params));
    }

    #[test]
    fn seed_changes_output() {
        let a = render(&WhiteNoiseParams { scale: 8, seed: 1 });
        let b = render(&WhiteNoiseParams { scale: 8, seed: 2 });
        assert_ne!(a, b, "different seeds produced identical pixels");
    }

    /// Every pixel within one cell shares the exact same random value — a
    /// smoothly-interpolated result here would mean this accidentally calls
    /// a continuous noise function instead of a per-cell hash.
    #[test]
    fn pixels_within_a_cell_are_identical() {
        let pixels = render(&WhiteNoiseParams { scale: 4, seed: 3 });
        let cell_px = RESOLUTION / 4;
        let a = pixel(&pixels, RESOLUTION, 0, 0);
        let b = pixel(&pixels, RESOLUTION, cell_px - 1, cell_px - 1);
        assert_eq!(
            a, b,
            "pixels in the same white-noise cell must be identical"
        );
    }
}
