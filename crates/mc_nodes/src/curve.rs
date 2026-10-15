// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Adjustments / Curve.
//!
//! Display name: "Curve". Tooltip: "A 5-point piecewise-linear curve at
//! x = 0, 0.25, 0.5, 0.75, 1.0 — drag each point's output value."

use crate::shaders::full_source;

const NODE_KIND: &str = "curve";

pub struct CurveParams {
    pub v0: f32,
    pub v1: f32,
    pub v2: f32,
    pub v3: f32,
    pub v4: f32,
}

impl Default for CurveParams {
    fn default() -> Self {
        Self {
            v0: 0.0,
            v1: 0.25,
            v2: 0.5,
            v3: 0.75,
            v4: 1.0,
        }
    }
}

impl CurveParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let float = |name: &str, default: f32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(default)
        };
        Self {
            v0: float("v0", defaults.v0),
            v1: float("v1", defaults.v1),
            v2: float("v2", defaults.v2),
            v3: float("v3", defaults.v3),
            v4: float("v4", defaults.v4),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &CurveParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("curve: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("curve.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("curve_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let output_view = pool.get_or_create(
        &gpu.device,
        output_key,
        resolution,
        wgpu::TextureFormat::Rgba16Float,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("curve_bind_group"),
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
            label: Some("curve_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("curve_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

/// Packs params into the exact byte layout of `CurveParams` in `curve.wgsl`
/// (resolution + 5 control points + 2 padding floats = 32 bytes).
fn pack_params(resolution: u32, params: &CurveParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.v0.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.v1.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.v2.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.v3.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.v4.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, CurveParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &CurveParams) -> Vec<u8> {
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
        let texture = pool.texture("output").expect("curve texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(&render(&CurveParams::default()), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = CurveParams {
            v2: 0.9,
            ..CurveParams::default()
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same input + params produced different pixels"
        );
    }

    #[test]
    fn identity_curve_is_a_no_op() {
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
            &CurveParams::default(),
        );
        let after = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        assert_eq!(
            before, after,
            "the default identity curve must not change pixels"
        );
    }

    #[test]
    fn flat_curve_flattens_every_channel() {
        let params = CurveParams {
            v0: 0.5,
            v1: 0.5,
            v2: 0.5,
            v3: 0.5,
            v4: 0.5,
        };
        let pixels = render(&params);
        let expected = (0.5f32 * 255.0).round() as i32;
        for px in pixels.as_chunks::<4>().0 {
            for &channel in &px[..3] {
                assert!(
                    (channel as i32 - expected).abs() <= 1,
                    "a flat curve at 0.5 must flatten every pixel to ~128"
                );
            }
        }
    }
}
