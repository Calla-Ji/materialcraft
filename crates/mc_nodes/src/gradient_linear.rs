// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Gradient Linear.
//!
//! Display name: "Gradient Linear". Tooltip: "A left-to-right ramp from
//! black to white, rotatable around the tile's center."

use crate::shaders::full_source;

const NODE_KIND: &str = "gradient_linear";

/// A dropdown rather than a free slider, and deliberately limited to these
/// four — the only angles where the ramp's two tile edges still line up
/// with its neighbor's. A continuous slider would just invite seams.
pub const ROTATION_OPTIONS: [&str; 4] = ["0°", "90°", "180°", "270°"];

fn rotation_turns(option: &str) -> f32 {
    match option {
        "90°" => 0.25,
        "180°" => 0.5,
        "270°" => 0.75,
        _ => 0.0,
    }
}

pub struct GradientLinearParams {
    /// Fraction of a full turn — always one of 0.0/0.25/0.5/0.75 when it
    /// comes from `from_graph`'s enum, but kept as a plain f32 internally
    /// since that's all the shader needs.
    pub rotation: f32,
}

impl Default for GradientLinearParams {
    fn default() -> Self {
        Self { rotation: 0.0 }
    }
}

impl GradientLinearParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            rotation: params
                .get("rotation")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => Some(rotation_turns(name)),
                    _ => None,
                })
                .unwrap_or(defaults.rotation),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &GradientLinearParams,
) {
    let shader_source = full_source("gradient_linear.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&params.rotation.to_le_bytes());
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
        wgpu::TextureFormat::Rgba16Float,
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
    use super::{dispatch, rotation_turns, GradientLinearParams, ROTATION_OPTIONS};
    use crate::test_support::RESOLUTION;

    fn render(params: &GradientLinearParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("gradient_linear texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    // No `assert_tiles_seamlessly` for the default (unrotated) case on
    // purpose: this is an intentionally discontinuous sawtooth ramp (see the
    // shader's doc comment) — it IS periodic (f(uv+1) = f(uv)), just not
    // seam-free, same as every other texture tool's Linear Gradient. What's
    // worth checking instead is that it actually ramps as expected.

    #[test]
    fn ramps_from_black_to_white_left_to_right() {
        let pixels = render(&GradientLinearParams::default());
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
        let params = GradientLinearParams { rotation: 0.1 };
        assert_eq!(
            render(&params),
            render(&params),
            "same params produced different pixels"
        );
    }

    /// A 90° rotation just turns the same sawtooth ramp to run top-to-bottom
    /// instead of left-to-right — still periodic with the same seam, same as
    /// the unrotated case, and still a quarter turn that keeps every other
    /// rotatable node (e.g. Transform 2D) exactly tileable.
    #[test]
    fn quarter_turn_rotation_ramps_top_to_bottom() {
        let pixels = render(&GradientLinearParams { rotation: 0.25 });
        let r = RESOLUTION as usize;
        let top = pixels[0];
        let bottom = pixels[(r - 1) * r * 4];
        assert!(top < 10, "top edge should be near black, got {top}");
        assert!(
            bottom > 245,
            "bottom edge should be near white, got {bottom}"
        );
    }

    #[test]
    fn every_rotation_option_parses_to_a_quarter_turn_multiple() {
        for option in ROTATION_OPTIONS {
            let turns = rotation_turns(option);
            assert!(
                (turns * 4.0).round() == turns * 4.0,
                "{option:?} parsed to {turns}, not a multiple of a quarter turn"
            );
        }
    }
}
