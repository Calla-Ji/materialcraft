//! Generators / Noise / Clouds.
//!
//! Display name: "Clouds". Tooltip: "An fBm preset tuned for a cloud-like
//! look — fewer knobs than Fractal Sum, a softer, more contrasty result."
//!
//! A thin, artist-facing wrapper around the same `perlin_2d` fBm summation
//! `perlin_noise.wgsl`/`fractal_sum.wgsl` use, fixed at 5 octaves and with a
//! contrast curve applied afterward — CLAUDE.md §7 calls this out as its own
//! node ("Clouds (fBm preset)") rather than just a Fractal Sum preset value,
//! so it gets its own shader rather than forwarding into Fractal Sum's.

use crate::shaders::full_source;

const NODE_KIND: &str = "clouds";

pub struct CloudsParams {
    pub scale: u32,
    /// How soft/fluffy the result looks — internally remapped to the fBm's
    /// persistence (see `persistence`), not exposed directly.
    pub softness: f32,
    pub seed: u32,
}

impl Default for CloudsParams {
    fn default() -> Self {
        Self {
            scale: 4,
            softness: 0.5,
            seed: 0,
        }
    }
}

impl CloudsParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            scale: params
                .get("scale")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.scale),
            softness: params
                .get("softness")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.softness),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
        }
    }

    fn persistence(&self) -> f32 {
        0.3 + self.softness.clamp(0.0, 1.0) * 0.35
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &CloudsParams,
) {
    let shader_source = full_source("clouds.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&params.scale.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&params.seed.to_le_bytes());
    param_bytes[12..16].copy_from_slice(&params.persistence().to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("clouds_params"),
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
        label: Some("clouds_bind_group"),
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
            label: Some("clouds_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("clouds_pass"),
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
    use super::{dispatch, CloudsParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &CloudsParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("clouds texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn tiles_seamlessly() {
        assert_tiles_seamlessly(&render(&CloudsParams::default()), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = CloudsParams {
            seed: 9,
            ..CloudsParams::default()
        };
        assert_eq!(render(&params), render(&params));
    }

    #[test]
    fn seed_changes_output() {
        let a = render(&CloudsParams {
            seed: 1,
            ..CloudsParams::default()
        });
        let b = render(&CloudsParams {
            seed: 2,
            ..CloudsParams::default()
        });
        assert_ne!(a, b, "different seeds produced identical pixels");
    }

    #[test]
    fn softness_changes_output() {
        let a = render(&CloudsParams {
            softness: 0.0,
            ..CloudsParams::default()
        });
        let b = render(&CloudsParams {
            softness: 1.0,
            ..CloudsParams::default()
        });
        assert_ne!(a, b, "different softness produced identical pixels");
    }
}
