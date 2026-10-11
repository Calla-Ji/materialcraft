// SPDX-License-Identifier: MIT OR Apache-2.0
//! Converters / Height to Normal.
//!
//! Display name: "Height to Normal". Tooltip: "Converts a height map to a
//! tangent-space normal map via a Sobel gradient."

use crate::shaders::full_source;
use crate::tiling::TilingMode;

const NODE_KIND: &str = "height_to_normal";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalFormat {
    OpenGl,
    DirectX,
}

impl NormalFormat {
    pub const NAMES: [&'static str; 2] = ["OpenGL", "DirectX"];

    fn as_u32(self) -> u32 {
        match self {
            NormalFormat::OpenGl => 0,
            NormalFormat::DirectX => 1,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "OpenGL" => Some(NormalFormat::OpenGl),
            "DirectX" => Some(NormalFormat::DirectX),
            _ => None,
        }
    }
}

pub struct HeightToNormalParams {
    pub intensity: f32,
    pub format: NormalFormat,
    pub tiling_mode: TilingMode,
}

impl Default for HeightToNormalParams {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            format: NormalFormat::OpenGl,
            tiling_mode: TilingMode::default(),
        }
    }
}

impl HeightToNormalParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            intensity: params
                .get("intensity")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.intensity),
            format: params
                .get("format")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => NormalFormat::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.format),
            tiling_mode: params
                .get("tiling_mode")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => TilingMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.tiling_mode),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &HeightToNormalParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("height_to_normal: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("height_to_normal.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&params.intensity.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&params.format.as_u32().to_le_bytes());
    param_bytes[12..16].copy_from_slice(&params.tiling_mode.as_u32().to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("height_to_normal_params"),
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
        label: Some("height_to_normal_bind_group"),
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
            label: Some("height_to_normal_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("height_to_normal_pass"),
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
    use super::{dispatch, HeightToNormalParams, NormalFormat};
    use crate::checker::{dispatch as checker_dispatch, CheckerParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, tiles_seamlessly, RESOLUTION};
    use crate::tiling::TilingMode;

    fn render(params: &HeightToNormalParams) -> Vec<u8> {
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
        let texture = pool.texture("output").expect("height_to_normal texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(&render(&HeightToNormalParams::default()), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = HeightToNormalParams {
            intensity: 2.0,
            ..HeightToNormalParams::default()
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same input + params produced different pixels"
        );
    }

    #[test]
    fn flat_input_is_a_flat_normal() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        // A constant-height input (scale/octaves don't matter if persistence
        // collapses amplitude to ~0 — simplest is just reading an unused key,
        // but we don't have a flat generator here, so build one directly via
        // uniform_color's dispatch instead of Perlin.
        crate::uniform_color::dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &crate::uniform_color::UniformColorParams {
                color: [0.5, 0.5, 0.5, 1.0],
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output",
            RESOLUTION,
            &HeightToNormalParams::default(),
        );
        let pixels = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        for pixel in pixels.as_chunks::<4>().0 {
            assert!(
                (pixel[0] as i32 - 128).abs() <= 2 && (pixel[1] as i32 - 128).abs() <= 2 && pixel[2] > 250,
                "flat height should produce a flat up-facing normal (128,128,255-ish), got {pixel:?}"
            );
        }
    }

    #[test]
    fn tiling_off_is_no_longer_seamless() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        checker_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &CheckerParams { scale: 2 },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output",
            RESOLUTION,
            &HeightToNormalParams {
                tiling_mode: TilingMode::Off,
                ..HeightToNormalParams::default()
            },
        );
        let pixels = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
        assert!(
            tiles_seamlessly(&pixels, RESOLUTION).is_err(),
            "disabling tiling should introduce a real seam"
        );
    }

    #[test]
    fn format_name_round_trips() {
        for name in NormalFormat::NAMES {
            assert!(
                NormalFormat::from_name(name).is_some(),
                "{name:?} should parse"
            );
        }
    }
}
