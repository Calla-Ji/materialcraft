// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Glitch / Scanline Jitter.
//!
//! Display name: "Scanline Jitter". Tooltip: "Shifts random horizontal or
//! vertical bands sideways — the classic 'bad video signal' glitch."

use crate::shaders::full_source;

const NODE_KIND: &str = "scanline_jitter";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanlineOrientation {
    Horizontal,
    Vertical,
}

impl ScanlineOrientation {
    pub const NAMES: [&'static str; 2] = ["Horizontal", "Vertical"];

    fn as_u32(self) -> u32 {
        match self {
            ScanlineOrientation::Horizontal => 0,
            ScanlineOrientation::Vertical => 1,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Horizontal" => Some(ScanlineOrientation::Horizontal),
            "Vertical" => Some(ScanlineOrientation::Vertical),
            _ => None,
        }
    }
}

pub struct ScanlineJitterParams {
    pub lines: i32,
    pub amount: f32,
    pub max_offset: f32,
    pub orientation: ScanlineOrientation,
    pub seed: u32,
}

impl Default for ScanlineJitterParams {
    fn default() -> Self {
        Self {
            lines: 32,
            amount: 0.3,
            max_offset: 0.05,
            orientation: ScanlineOrientation::Horizontal,
            seed: 0,
        }
    }
}

impl ScanlineJitterParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            lines: params
                .get("lines")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1))
                .unwrap_or(defaults.lines),
            amount: params
                .get("amount")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.amount),
            max_offset: params
                .get("max_offset")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.max_offset),
            orientation: params
                .get("orientation")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => ScanlineOrientation::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.orientation),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
        }
    }
}

fn pack_params(resolution: u32, params: &ScanlineJitterParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.lines.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.orientation.as_u32().to_le_bytes());
    bytes[12..16].copy_from_slice(&params.amount.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.max_offset.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.seed.to_le_bytes());
    bytes
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    image_key: &str,
    mask_key: Option<&str>,
    output_key: &str,
    resolution: u32,
    params: &ScanlineJitterParams,
) {
    let default_mask_key = format!("{output_key}__default_mask");
    let mask_key = match mask_key {
        Some(key) => key,
        None => {
            gpu.fill_rgba8(pool, &default_mask_key, resolution, [255, 255, 255, 255]);
            &default_mask_key
        }
    };

    let image_view = pool
        .view(image_key)
        .unwrap_or_else(|| panic!("scanline_jitter: no texture at image key {image_key:?}"))
        .clone();
    let mask_view = pool
        .view(mask_key)
        .unwrap_or_else(|| panic!("scanline_jitter: no texture at mask key {mask_key:?}"))
        .clone();

    let shader_source = full_source("scanline_jitter.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scanline_jitter_params"),
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
        label: Some("scanline_jitter_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&image_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&mask_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scanline_jitter_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scanline_jitter_pass"),
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
    use super::{dispatch, ScanlineJitterParams, ScanlineOrientation};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render_from_noise(params: &ScanlineJitterParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::perlin::dispatch(
            &mut gpu,
            &mut pool,
            "image",
            RESOLUTION,
            &crate::perlin::PerlinParams::default(),
        );
        dispatch(
            &mut gpu, &mut pool, "image", None, "out", RESOLUTION, params,
        );
        let texture = pool.texture("out").expect("scanline jitter texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    fn original_noise() -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::perlin::dispatch(
            &mut gpu,
            &mut pool,
            "image",
            RESOLUTION,
            &crate::perlin::PerlinParams::default(),
        );
        gpu.read_rgba8(pool.texture("image").expect("image texture"), RESOLUTION)
    }

    #[test]
    fn tiles_seamlessly_both_orientations() {
        for orientation in [
            ScanlineOrientation::Horizontal,
            ScanlineOrientation::Vertical,
        ] {
            assert_tiles_seamlessly(
                &render_from_noise(&ScanlineJitterParams {
                    orientation,
                    ..ScanlineJitterParams::default()
                }),
                RESOLUTION,
            );
        }
    }

    #[test]
    fn is_deterministic() {
        let params = ScanlineJitterParams::default();
        assert_eq!(render_from_noise(&params), render_from_noise(&params));
    }

    #[test]
    fn zero_amount_is_a_no_op() {
        let params = ScanlineJitterParams {
            amount: 0.0,
            ..ScanlineJitterParams::default()
        };
        assert_eq!(render_from_noise(&params), original_noise());
    }

    #[test]
    fn nonzero_amount_changes_the_output() {
        let params = ScanlineJitterParams::default();
        assert_ne!(render_from_noise(&params), original_noise());
    }

    #[test]
    fn every_orientation_name_round_trips() {
        for name in ScanlineOrientation::NAMES {
            assert!(
                ScanlineOrientation::from_name(name).is_some(),
                "{name} should parse back to a ScanlineOrientation"
            );
        }
    }
}
