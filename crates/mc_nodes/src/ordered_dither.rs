// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Stylize / Ordered Dither.
//!
//! Display name: "Ordered Dither". Tooltip: "Reduces an image to a few
//! tones with an ordered (Bayer or blue noise) dither pattern."

use crate::dither_pattern::DitherPattern;
use crate::shaders::full_source;

const NODE_KIND: &str = "ordered_dither";

/// The 4 patterns the brief scopes for Ordered Dither — a subset of
/// [`DitherPattern`]'s full set (no Dot, which is Dither Pattern's own
/// addition with no obvious "dither with a halftone dot" meaning here).
pub const PATTERN_NAMES: [&str; 4] = ["Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Blue Noise"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderedDitherMode {
    PerChannel,
    Luminance,
}

impl OrderedDitherMode {
    pub const NAMES: [&'static str; 2] = ["Per Channel", "Luminance"];

    fn as_u32(self) -> u32 {
        match self {
            OrderedDitherMode::PerChannel => 0,
            OrderedDitherMode::Luminance => 1,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Per Channel" => Some(OrderedDitherMode::PerChannel),
            "Luminance" => Some(OrderedDitherMode::Luminance),
            _ => None,
        }
    }
}

pub struct OrderedDitherParams {
    pub pattern: DitherPattern,
    pub cells: i32,
    pub levels: i32,
    pub spread: f32,
    pub bias: f32,
    pub mode: OrderedDitherMode,
    pub seed: u32,
}

impl Default for OrderedDitherParams {
    fn default() -> Self {
        Self {
            pattern: DitherPattern::Bayer4x4,
            cells: 64,
            levels: 4,
            spread: 1.0,
            bias: 0.0,
            mode: OrderedDitherMode::Luminance,
            seed: 0,
        }
    }
}

impl OrderedDitherParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            pattern: params
                .get("pattern")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => DitherPattern::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.pattern),
            cells: params
                .get("cells")
                .and_then(mc_core::ParamValue::as_int)
                .unwrap_or(defaults.cells),
            levels: params
                .get("levels")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(2))
                .unwrap_or(defaults.levels),
            spread: params
                .get("spread")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.spread),
            bias: params
                .get("bias")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.bias),
            mode: params
                .get("mode")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => OrderedDitherMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.mode),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
        }
    }
}

fn pack_params(resolution: u32, cells: i32, params: &OrderedDitherParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&cells.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.pattern.as_u32().to_le_bytes());
    bytes[12..16].copy_from_slice(&params.levels.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.spread.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.bias.to_le_bytes());
    bytes[24..28].copy_from_slice(&params.mode.as_u32().to_le_bytes());
    bytes[28..32].copy_from_slice(&params.seed.to_le_bytes());
    bytes
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    image_key: &str,
    output_key: &str,
    resolution: u32,
    params: &OrderedDitherParams,
) {
    let image_view = pool
        .view(image_key)
        .unwrap_or_else(|| panic!("ordered_dither: no texture at image key {image_key:?}"))
        .clone();
    let blue_noise_view = crate::dither_common::upload_blue_noise_mask(gpu, pool);

    let shader_source = full_source("ordered_dither.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let cells =
        crate::dither_common::round_up_to_multiple(params.cells, params.pattern.tile_size());
    let param_bytes = pack_params(resolution, cells, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ordered_dither_params"),
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
        label: Some("ordered_dither_bind_group"),
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
                resource: wgpu::BindingResource::TextureView(&blue_noise_view),
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
            label: Some("ordered_dither_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ordered_dither_pass"),
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
    use super::{dispatch, DitherPattern, OrderedDitherMode, OrderedDitherParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render_from_noise(params: &OrderedDitherParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::perlin::dispatch(
            &mut gpu,
            &mut pool,
            "image",
            RESOLUTION,
            &crate::perlin::PerlinParams::default(),
        );
        dispatch(&mut gpu, &mut pool, "image", "out", RESOLUTION, params);
        let texture = pool.texture("out").expect("ordered dither texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    fn render_flat_gray(params: &OrderedDitherParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        gpu.fill_rgba8(&mut pool, "image", RESOLUTION, [128, 128, 128, 255]);
        dispatch(&mut gpu, &mut pool, "image", "out", RESOLUTION, params);
        let texture = pool.texture("out").expect("ordered dither texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn tiles_seamlessly_for_every_pattern() {
        for &pattern in &[
            DitherPattern::Bayer2x2,
            DitherPattern::Bayer4x4,
            DitherPattern::Bayer8x8,
            DitherPattern::BlueNoise,
        ] {
            assert_tiles_seamlessly(
                &render_from_noise(&OrderedDitherParams {
                    pattern,
                    ..OrderedDitherParams::default()
                }),
                RESOLUTION,
            );
        }
    }

    #[test]
    fn is_deterministic() {
        let params = OrderedDitherParams::default();
        assert_eq!(render_from_noise(&params), render_from_noise(&params));
    }

    #[test]
    fn levels_two_produces_only_two_distinct_values() {
        let pixels = render_from_noise(&OrderedDitherParams {
            levels: 2,
            ..OrderedDitherParams::default()
        });
        let distinct: std::collections::BTreeSet<u8> =
            pixels.as_chunks::<4>().0.iter().map(|px| px[0]).collect();
        assert!(
            distinct.len() <= 2,
            "Levels = 2 should produce at most 2 distinct values, found {distinct:?}"
        );
    }

    #[test]
    fn zero_spread_matches_plain_posterize() {
        // With no dither noise, every texel's value only depends on its own
        // continuous input value — the same posterize a `round()` to
        // `levels` steps would give directly, regardless of which dither
        // pattern's threshold would otherwise be mixed in.
        let bayer = render_from_noise(&OrderedDitherParams {
            pattern: DitherPattern::Bayer4x4,
            spread: 0.0,
            ..OrderedDitherParams::default()
        });
        let blue_noise = render_from_noise(&OrderedDitherParams {
            pattern: DitherPattern::BlueNoise,
            spread: 0.0,
            ..OrderedDitherParams::default()
        });
        assert_eq!(
            bayer, blue_noise,
            "zero spread must ignore the dither pattern entirely"
        );
    }

    #[test]
    fn flat_half_gray_with_bayer_2x2_is_a_50_50_split() {
        let pixels = render_flat_gray(&OrderedDitherParams {
            pattern: DitherPattern::Bayer2x2,
            cells: RESOLUTION as i32,
            levels: 2,
            spread: 1.0,
            bias: 0.0,
            mode: OrderedDitherMode::PerChannel,
            seed: 0,
        });
        let white_count = pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[0] > 127)
            .count();
        let total = (RESOLUTION * RESOLUTION) as usize;
        assert_eq!(
            white_count,
            total / 2,
            "a flat ~50% gray input dithered with Bayer 2x2 at Levels=2 should split evenly"
        );
    }

    #[test]
    fn luminance_and_per_channel_agree_on_gray_input() {
        let per_channel = render_from_noise(&OrderedDitherParams {
            mode: OrderedDitherMode::PerChannel,
            ..OrderedDitherParams::default()
        });
        let luminance = render_from_noise(&OrderedDitherParams {
            mode: OrderedDitherMode::Luminance,
            ..OrderedDitherParams::default()
        });
        assert_eq!(
            per_channel, luminance,
            "Mode should have no visible effect on a Gray-sourced (R=G=B) image"
        );
    }

    #[test]
    fn every_mode_name_round_trips() {
        for name in OrderedDitherMode::NAMES {
            assert!(
                OrderedDitherMode::from_name(name).is_some(),
                "{name} should parse back to an OrderedDitherMode"
            );
        }
    }
}
