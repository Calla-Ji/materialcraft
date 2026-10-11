//! Filters / Stylize / Palette Dither.
//!
//! Display name: "Palette Dither". Tooltip: "Dithers a Color image down to a
//! fixed palette of 2-8 colors."

use crate::dither_pattern::DitherPattern;
use crate::shaders::full_source;

const NODE_KIND: &str = "palette_dither";
/// Matches `palette_dither.wgsl`'s fixed-size `colors` array.
const MAX_COLORS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMode {
    Rgb,
    Perceptual,
}

impl DistanceMode {
    pub const NAMES: [&'static str; 2] = ["RGB", "Perceptual"];

    fn as_u32(self) -> u32 {
        match self {
            DistanceMode::Rgb => 0,
            DistanceMode::Perceptual => 1,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "RGB" => Some(DistanceMode::Rgb),
            "Perceptual" => Some(DistanceMode::Perceptual),
            _ => None,
        }
    }
}

pub struct PaletteDitherParams {
    /// Always sorted by position ascending, always at least 2 colors —
    /// `from_graph` and `Default` both guarantee this so `dispatch` never
    /// has to handle a degenerate palette. Position itself is discarded
    /// after sorting — a stop's color is a palette entry, nothing more.
    pub colors: Vec<[f32; 4]>,
    pub pattern: DitherPattern,
    pub cells: i32,
    pub spread: f32,
    pub distance: DistanceMode,
    pub seed: u32,
}

impl Default for PaletteDitherParams {
    fn default() -> Self {
        Self {
            colors: vec![
                [0.0, 0.0, 0.0, 1.0],
                [0.5, 0.5, 0.5, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            pattern: DitherPattern::Bayer4x4,
            cells: 64,
            spread: 1.0,
            distance: DistanceMode::Perceptual,
            seed: 0,
        }
    }
}

impl PaletteDitherParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let mut colors = match params.get("palette") {
            Some(mc_core::ParamValue::Gradient(stops)) if stops.len() >= 2 => {
                let mut stops = stops.clone();
                stops.sort_by(|a, b| a.position.total_cmp(&b.position));
                stops.into_iter().map(|s| s.color).collect()
            }
            _ => defaults.colors,
        };
        colors.truncate(MAX_COLORS);

        Self {
            colors,
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
            spread: params
                .get("spread")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.spread),
            distance: params
                .get("distance")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => DistanceMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.distance),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
        }
    }
}

/// Packs params into the exact byte layout of `PaletteDitherParams` in
/// `palette_dither.wgsl`: a 32-byte header (resolution, cells, pattern,
/// color_count, distance_mode, spread, seed, padding), then
/// `colors: array<vec4<f32>, 8>` (128 bytes) = 160 bytes total.
fn pack_params(resolution: u32, cells: i32, params: &PaletteDitherParams) -> [u8; 160] {
    let mut bytes = [0u8; 160];
    let color_count = params.colors.len().clamp(1, MAX_COLORS) as u32;
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&cells.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.pattern.as_u32().to_le_bytes());
    bytes[12..16].copy_from_slice(&color_count.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.distance.as_u32().to_le_bytes());
    bytes[20..24].copy_from_slice(&params.spread.to_le_bytes());
    bytes[24..28].copy_from_slice(&params.seed.to_le_bytes());

    for (i, color) in params.colors.iter().enumerate().take(MAX_COLORS) {
        let offset = 32 + i * 16;
        for (c, channel) in color.iter().enumerate() {
            let o = offset + c * 4;
            bytes[o..o + 4].copy_from_slice(&channel.to_le_bytes());
        }
    }
    bytes
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    image_key: &str,
    output_key: &str,
    resolution: u32,
    params: &PaletteDitherParams,
) {
    let image_view = pool
        .view(image_key)
        .unwrap_or_else(|| panic!("palette_dither: no texture at image key {image_key:?}"))
        .clone();
    let blue_noise_view = crate::dither_common::upload_blue_noise_mask(gpu, pool);

    let shader_source = full_source("palette_dither.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let cells =
        crate::dither_common::round_up_to_multiple(params.cells, params.pattern.tile_size());
    let param_bytes = pack_params(resolution, cells, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("palette_dither_params"),
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
        label: Some("palette_dither_bind_group"),
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
            label: Some("palette_dither_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("palette_dither_pass"),
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
    use super::{dispatch, DistanceMode, PaletteDitherParams};
    use crate::dither_pattern::DitherPattern;
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render_from_noise(params: &PaletteDitherParams) -> Vec<u8> {
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
        let texture = pool.texture("out").expect("palette dither texture");
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
                &render_from_noise(&PaletteDitherParams {
                    pattern,
                    ..PaletteDitherParams::default()
                }),
                RESOLUTION,
            );
        }
    }

    #[test]
    fn is_deterministic() {
        let params = PaletteDitherParams::default();
        assert_eq!(render_from_noise(&params), render_from_noise(&params));
    }

    #[test]
    fn every_output_pixel_is_one_of_the_palette_colors() {
        let params = PaletteDitherParams::default();
        let pixels = render_from_noise(&params);
        let palette_u8: Vec<[u8; 3]> = params
            .colors
            .iter()
            .map(|c| {
                [
                    (c[0] * 255.0).round() as u8,
                    (c[1] * 255.0).round() as u8,
                    (c[2] * 255.0).round() as u8,
                ]
            })
            .collect();
        for px in pixels.as_chunks::<4>().0 {
            let rgb = [px[0], px[1], px[2]];
            assert!(
                palette_u8.contains(&rgb),
                "pixel {rgb:?} isn't one of the palette colors {palette_u8:?}"
            );
        }
    }

    #[test]
    fn every_distance_mode_name_round_trips() {
        for name in DistanceMode::NAMES {
            assert!(
                DistanceMode::from_name(name).is_some(),
                "{name} should parse back to a DistanceMode"
            );
        }
    }

    #[test]
    fn from_graph_never_produces_fewer_than_one_color() {
        let params = PaletteDitherParams::from_graph(&std::collections::BTreeMap::new());
        assert!(!params.colors.is_empty());
    }
}
