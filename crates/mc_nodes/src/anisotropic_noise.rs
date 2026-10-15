// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Noise / Anisotropic Noise.
//!
//! Display name: "Anisotropic Noise". Tooltip: "Directional gradient noise
//! — different frequency per axis for streaky, brushed-metal or
//! wood-grain-like variation."
//!
//! Same gradient-noise shape as Perlin Noise, except `scale_x`/`scale_y`
//! are independent instead of one shared `scale` — stretching one axis
//! relative to the other is what produces the streaky look. Output is
//! stored as an `Rgba16Float` texture (gray value broadcast to RGB, alpha =
//! 1), same as every other noise node.

use crate::shaders::full_source;
use crate::tiling::TilingMode;

const NODE_KIND: &str = "anisotropic_noise";

/// Anisotropic Noise's parameters, in artist-friendly terms.
pub struct AnisotropicNoiseParams {
    /// How many noise cells tile across the texture along X. Higher = finer detail.
    pub scale_x: u32,
    /// How many noise cells tile across the texture along Y. Higher = finer detail.
    pub scale_y: u32,
    /// How many layers of noise are summed (fractal Brownian motion).
    pub octaves: u32,
    /// How much each successive octave's contribution shrinks, in (0, 1].
    pub persistence: f32,
    /// Changes the random pattern without changing its statistical character.
    pub seed: u32,
    pub tiling_mode: TilingMode,
    /// Contrast boost around mid-gray; 1 = unchanged, higher pushes values
    /// away from 0.5 so the noise reads as crisper and less blurry.
    pub sharpness: f32,
}

impl Default for AnisotropicNoiseParams {
    fn default() -> Self {
        Self {
            scale_x: 4,
            scale_y: 16,
            octaves: 4,
            persistence: 0.5,
            seed: 0,
            tiling_mode: TilingMode::default(),
            sharpness: 1.0,
        }
    }
}

impl AnisotropicNoiseParams {
    /// Reads a graph node's generic param map into typed
    /// `AnisotropicNoiseParams`, falling back to defaults for anything
    /// missing or the wrong type.
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            scale_x: params
                .get("scale_x")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.scale_x),
            scale_y: params
                .get("scale_y")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.scale_y),
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
            tiling_mode: params
                .get("tiling_mode")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => TilingMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.tiling_mode),
            sharpness: params
                .get("sharpness")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.sharpness),
        }
    }
}

/// Evaluate the Anisotropic Noise node at `resolution x resolution` into
/// the texture pool slot `key`. Fetch the result with `pool.view(key)` /
/// `pool.texture(key)` afterwards.
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &AnisotropicNoiseParams,
) {
    let shader_source = full_source("anisotropic_noise.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("anisotropic_noise_params"),
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
        label: Some("anisotropic_noise_bind_group"),
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
            label: Some("anisotropic_noise_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("anisotropic_noise_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

/// Packs params into the exact byte layout of `AnisotropicNoiseParams` in
/// `anisotropic_noise.wgsl` (8 x 4-byte fields = 32 bytes; see that file
/// for the struct definition this must stay in sync with).
fn pack_params(resolution: u32, params: &AnisotropicNoiseParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.scale_x.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.scale_y.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.octaves.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.seed.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.persistence.to_le_bytes());
    bytes[24..28].copy_from_slice(&params.tiling_mode.as_u32().to_le_bytes());
    bytes[28..32].copy_from_slice(&params.sharpness.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, AnisotropicNoiseParams};
    use crate::test_support::{
        assert_tiles_seamlessly, tiles_seamlessly as check_tiles_seamlessly, RESOLUTION,
    };
    use crate::tiling::TilingMode;

    fn render(params: &AnisotropicNoiseParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("anisotropic noise texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn tiles_seamlessly() {
        assert_tiles_seamlessly(&render(&AnisotropicNoiseParams::default()), RESOLUTION);
    }

    /// `scale_x`/`scale_y` are independent axes of the same lattice, so
    /// each must keep wrapping exactly at the texture edge even when the
    /// other is wildly different — not just at matched scales, which
    /// `tiles_seamlessly` above alone wouldn't catch.
    #[test]
    fn tiles_seamlessly_with_very_different_scales() {
        let params = AnisotropicNoiseParams {
            scale_x: 2,
            scale_y: 24,
            ..AnisotropicNoiseParams::default()
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    /// Hard rule #3 (CLAUDE.md §6): same graph + same seed = bit-identical
    /// output on the same GPU.
    #[test]
    fn is_deterministic() {
        let params = AnisotropicNoiseParams {
            seed: 42,
            ..AnisotropicNoiseParams::default()
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
        let a = render(&AnisotropicNoiseParams {
            seed: 1,
            ..AnisotropicNoiseParams::default()
        });
        let b = render(&AnisotropicNoiseParams {
            seed: 2,
            ..AnisotropicNoiseParams::default()
        });
        assert_ne!(a, b, "different seeds produced identical pixels");
    }

    /// The entire point of the node: an uneven `scale_x`/`scale_y` split
    /// must actually look different from the isotropic case (equal
    /// scales), otherwise the two axes aren't really independent.
    #[test]
    fn uneven_scales_differ_from_isotropic() {
        let isotropic = render(&AnisotropicNoiseParams {
            scale_x: 8,
            scale_y: 8,
            ..AnisotropicNoiseParams::default()
        });
        let anisotropic = render(&AnisotropicNoiseParams {
            scale_x: 2,
            scale_y: 24,
            ..AnisotropicNoiseParams::default()
        });
        assert_ne!(
            isotropic, anisotropic,
            "a stretched scale_x/scale_y split produced the same pixels as the isotropic case"
        );
    }

    /// The entire point of the param: a higher sharpness should push
    /// pixel values further from mid-gray (more extreme blacks/whites),
    /// which is what "crisper, less blurry" cashes out to for a noise
    /// field.
    #[test]
    fn higher_sharpness_increases_contrast() {
        fn mean_abs_deviation_from_mid_gray(pixels: &[u8]) -> f64 {
            let samples: Vec<f64> = pixels
                .as_chunks::<4>()
                .0
                .iter()
                .map(|px| (px[0] as f64 - 127.5).abs())
                .collect();
            samples.iter().sum::<f64>() / samples.len() as f64
        }

        let soft = render(&AnisotropicNoiseParams {
            sharpness: 1.0,
            ..AnisotropicNoiseParams::default()
        });
        let sharp = render(&AnisotropicNoiseParams {
            sharpness: 4.0,
            ..AnisotropicNoiseParams::default()
        });
        assert!(
            mean_abs_deviation_from_mid_gray(&sharp) > mean_abs_deviation_from_mid_gray(&soft),
            "a higher sharpness should push pixels further from mid-gray on average"
        );
    }

    #[test]
    fn tiling_off_is_no_longer_seamless() {
        let params = AnisotropicNoiseParams {
            tiling_mode: TilingMode::Off,
            ..AnisotropicNoiseParams::default()
        };
        assert!(
            check_tiles_seamlessly(&render(&params), RESOLUTION).is_err(),
            "disabling tiling should break the forced seamless lattice wrap"
        );
    }

    #[test]
    fn horizontal_only_still_wraps_vertically_but_not_horizontally() {
        let both = render(&AnisotropicNoiseParams::default());
        let horizontal_only = render(&AnisotropicNoiseParams {
            tiling_mode: TilingMode::HorizontalOnly,
            ..AnisotropicNoiseParams::default()
        });
        assert_ne!(
            both, horizontal_only,
            "turning off the vertical axis alone should change the output"
        );
    }
}
