// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Adjustments / Levels.
//!
//! Display name: "Levels". Tooltip: "Remap an input/output range, with a
//! gamma curve in between — brighten, darken, or boost contrast."
//!
//! Unlike Perlin Noise, this node has an input: it reads another node's
//! texture via a plain sampled binding (`textureLoad`, no sampler object
//! needed) rather than a read-write storage texture, which would need a
//! native-only wgpu feature we'd otherwise have to request.

use crate::shaders::full_source;

const NODE_KIND: &str = "levels";

/// Levels' parameters, in artist-friendly terms.
pub struct LevelsParams {
    pub in_low: f32,
    pub in_high: f32,
    pub gamma: f32,
    pub out_low: f32,
    pub out_high: f32,
}

impl Default for LevelsParams {
    fn default() -> Self {
        Self {
            in_low: 0.0,
            in_high: 1.0,
            gamma: 1.0,
            out_low: 0.0,
            out_high: 1.0,
        }
    }
}

impl LevelsParams {
    /// Reads a graph node's generic param map into typed `LevelsParams`,
    /// falling back to defaults for anything missing or the wrong type.
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let float = |name: &str, default: f32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(default)
        };
        Self {
            in_low: float("in_low", defaults.in_low),
            in_high: float("in_high", defaults.in_high),
            gamma: float("gamma", defaults.gamma),
            out_low: float("out_low", defaults.out_low),
            out_high: float("out_high", defaults.out_high),
        }
    }
}

/// Evaluate Levels at `resolution x resolution`, reading `input_key`'s
/// texture from the pool and writing the result into `output_key`.
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &LevelsParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("levels: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("levels.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("levels_params"),
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
        label: Some("levels_bind_group"),
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
            label: Some("levels_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("levels_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

/// Packs params into the exact byte layout of `LevelsParams` in
/// `levels.wgsl` (8 x 4-byte fields = 32 bytes; see that file for the struct
/// definition this must stay in sync with).
fn pack_params(resolution: u32, params: &LevelsParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.in_low.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.in_high.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.gamma.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.out_low.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.out_high.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, LevelsParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &LevelsParams) -> Vec<u8> {
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
        let texture = pool.texture("output").expect("levels texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    /// Levels is a pointwise remap (no neighborhood sampling), so it can't
    /// introduce a seam on its own — but a real implementation bug (e.g.
    /// sampling the wrong pixel) could, so this is still worth checking.
    #[test]
    fn preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(&render(&LevelsParams::default()), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = LevelsParams {
            gamma: 1.8,
            ..LevelsParams::default()
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same input + params produced different pixels"
        );
    }

    #[test]
    fn default_params_are_a_no_op() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams::default(),
        );
        let before = gpu.read_rgba8(pool.texture("input").unwrap(), RESOLUTION);

        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output",
            RESOLUTION,
            &LevelsParams::default(),
        );
        let after = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        assert_eq!(
            before, after,
            "identity levels params must not change pixels"
        );
    }
}
