//! Generators / Noise / Perlin Noise.
//!
//! Display name: "Perlin Noise". Tooltip: "Classic gradient noise — smooth,
//! organic variation."
//!
//! Output is stored as an `Rgba8Unorm` texture (gray value broadcast to RGB,
//! alpha = 1) so it can be shown directly in egui's 2D view and read back for
//! PNG export without a separate conversion step.

use crate::shaders::full_source;

const NODE_KIND: &str = "perlin_noise";

/// Perlin Noise's parameters, in artist-friendly terms.
pub struct PerlinParams {
    /// How many noise cells tile across the texture. Higher = finer detail.
    pub scale: u32,
    /// How many layers of noise are summed (fractal Brownian motion).
    pub octaves: u32,
    /// How much each successive octave's contribution shrinks, in (0, 1].
    pub persistence: f32,
    /// Changes the random pattern without changing its statistical character.
    pub seed: u32,
}

impl Default for PerlinParams {
    fn default() -> Self {
        Self {
            scale: 4,
            octaves: 4,
            persistence: 0.5,
            seed: 0,
        }
    }
}

impl PerlinParams {
    /// Reads a graph node's generic param map into typed `PerlinParams`,
    /// falling back to defaults for anything missing or the wrong type.
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            scale: params
                .get("scale")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.scale),
            octaves: params
                .get("octaves")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.octaves),
            persistence: params
                .get("persistence")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.persistence),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
        }
    }
}

/// Evaluate the Perlin Noise node at `resolution x resolution` into the
/// texture pool slot `key`. Fetch the result with `pool.view(key)` /
/// `pool.texture(key)` afterwards.
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &PerlinParams,
) {
    let shader_source = full_source("perlin_noise.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("perlin_noise_params"),
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
        label: Some("perlin_noise_bind_group"),
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
            label: Some("perlin_noise_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("perlin_noise_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

/// Packs params into the exact byte layout of `PerlinParams` in
/// `perlin_noise.wgsl` (8 x 4-byte fields = 32 bytes; see that file for the
/// struct definition this must stay in sync with).
fn pack_params(resolution: u32, params: &PerlinParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.scale.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.octaves.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.seed.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.persistence.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &PerlinParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("perlin texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn tiles_seamlessly() {
        assert_tiles_seamlessly(&render(&PerlinParams::default()), RESOLUTION);
    }

    /// Hard rule #3 (CLAUDE.md §6): same graph + same seed = bit-identical
    /// output on the same GPU.
    #[test]
    fn is_deterministic() {
        let params = PerlinParams {
            seed: 42,
            ..PerlinParams::default()
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same params + seed produced different pixels"
        );
    }

    /// Changing the seed should actually change the output, otherwise the
    /// determinism test above would pass trivially.
    #[test]
    fn seed_changes_output() {
        let a = render(&PerlinParams {
            seed: 1,
            ..PerlinParams::default()
        });
        let b = render(&PerlinParams {
            seed: 2,
            ..PerlinParams::default()
        });
        assert_ne!(a, b, "different seeds produced identical pixels");
    }
}
