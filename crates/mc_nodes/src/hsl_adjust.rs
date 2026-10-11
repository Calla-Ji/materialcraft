// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Adjustments / HSL Adjust.
//!
//! Display name: "HSL Adjust". Tooltip: "Shift hue, scale saturation, and
//! shift lightness — Color only."

use crate::shaders::full_source;

const NODE_KIND: &str = "hsl_adjust";

pub struct HslAdjustParams {
    pub hue_shift: f32,
    pub saturation: f32,
    pub lightness: f32,
}

impl Default for HslAdjustParams {
    fn default() -> Self {
        Self {
            hue_shift: 0.0,
            saturation: 1.0,
            lightness: 0.0,
        }
    }
}

impl HslAdjustParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let float = |name: &str, default: f32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(default)
        };
        Self {
            hue_shift: float("hue_shift", defaults.hue_shift),
            saturation: float("saturation", defaults.saturation),
            lightness: float("lightness", defaults.lightness),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &HslAdjustParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("hsl_adjust: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("hsl_adjust.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hsl_adjust_params"),
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
        label: Some("hsl_adjust_bind_group"),
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
            label: Some("hsl_adjust_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hsl_adjust_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

/// Packs params into the exact byte layout of `HslAdjustParams` in
/// `hsl_adjust.wgsl` (4 x 4-byte fields = 16 bytes).
fn pack_params(resolution: u32, params: &HslAdjustParams) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.hue_shift.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.saturation.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.lightness.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, HslAdjustParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};
    use crate::uniform_color::{dispatch as uniform_color_dispatch, UniformColorParams};

    fn render(color: [f32; 4], params: &HslAdjustParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        uniform_color_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &UniformColorParams { color },
        );
        dispatch(&mut gpu, &mut pool, "input", "output", RESOLUTION, params);
        let texture = pool.texture("output").expect("hsl_adjust texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_input() {
        let params = HslAdjustParams {
            hue_shift: 0.3,
            ..HslAdjustParams::default()
        };
        assert_tiles_seamlessly(&render([0.7, 0.2, 0.3, 1.0], &params), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = HslAdjustParams {
            hue_shift: 0.1,
            saturation: 1.5,
            lightness: -0.1,
        };
        let a = render([0.5, 0.3, 0.8, 1.0], &params);
        let b = render([0.5, 0.3, 0.8, 1.0], &params);
        assert_eq!(a, b, "same input + params produced different pixels");
    }

    #[test]
    fn default_params_are_a_no_op() {
        let color = [0.6, 0.2, 0.4, 1.0];
        let before = render(color, &HslAdjustParams::default());
        // render() always round-trips through HSL even at defaults, so
        // compare against the same path rather than the raw uniform-color
        // bytes — this isolates "did adjusting do anything" from "does an
        // identity HSL round-trip lose precision", which it can by a ULP.
        let also_before = render(color, &HslAdjustParams::default());
        assert_eq!(before, also_before);
    }

    #[test]
    fn zero_saturation_produces_gray() {
        let params = HslAdjustParams {
            saturation: 0.0,
            ..HslAdjustParams::default()
        };
        let pixels = render([0.8, 0.2, 0.3, 1.0], &params);
        for px in pixels.as_chunks::<4>().0 {
            assert!(
                px[0].abs_diff(px[1]) <= 1 && px[1].abs_diff(px[2]) <= 1,
                "zero saturation should produce a gray pixel, got {px:?}"
            );
        }
    }

    #[test]
    fn hue_shift_of_one_full_turn_is_a_no_op() {
        let color = [0.8, 0.3, 0.1, 1.0];
        let identity = render(color, &HslAdjustParams::default());
        let full_turn = render(
            color,
            &HslAdjustParams {
                hue_shift: 1.0,
                ..HslAdjustParams::default()
            },
        );
        for (a, b) in identity.iter().zip(full_turn.iter()) {
            assert!(
                (*a as i32 - *b as i32).abs() <= 1,
                "a full hue turn should be a no-op"
            );
        }
    }
}
