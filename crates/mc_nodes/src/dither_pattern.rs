//! Generators / Patterns / Dither Pattern.
//!
//! Display name: "Dither Pattern". Tooltip: "The raw ordered-dither
//! threshold mask (Bayer, blue noise, or a halftone dot grid), for use in
//! Blend, Threshold, and similar nodes."

use crate::shaders::full_source;

const NODE_KIND: &str = "dither_pattern";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitherPattern {
    Bayer2x2,
    Bayer4x4,
    Bayer8x8,
    BlueNoise,
    Dot,
}

impl DitherPattern {
    pub const NAMES: [&'static str; 5] =
        ["Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Blue Noise", "Dot"];

    pub(crate) fn as_u32(self) -> u32 {
        match self {
            DitherPattern::Bayer2x2 => 0,
            DitherPattern::Bayer4x4 => 1,
            DitherPattern::Bayer8x8 => 2,
            DitherPattern::BlueNoise => 3,
            DitherPattern::Dot => 4,
        }
    }

    /// The pattern's native tile size in texels — `Cells X`/`Cells Y` are
    /// each rounded up to the nearest multiple of this so the pattern
    /// always divides evenly into the texture and wraps seamlessly. Dot
    /// cells have no inherent matrix size, so any single cell already
    /// tiles; 1 is a no-op multiple for the rounding math.
    pub fn tile_size(self) -> i32 {
        match self {
            DitherPattern::Bayer2x2 => 2,
            DitherPattern::Bayer4x4 => 4,
            DitherPattern::Bayer8x8 => 8,
            DitherPattern::BlueNoise => crate::dither_common::BLUE_NOISE_SIZE as i32,
            DitherPattern::Dot => 1,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Bayer 2x2" => Some(DitherPattern::Bayer2x2),
            "Bayer 4x4" => Some(DitherPattern::Bayer4x4),
            "Bayer 8x8" => Some(DitherPattern::Bayer8x8),
            "Blue Noise" => Some(DitherPattern::BlueNoise),
            "Dot" => Some(DitherPattern::Dot),
            _ => None,
        }
    }
}

/// A classic-halftone "screen angle": rather than a continuous rotation
/// (which can't tile exactly except by accident), real halftone screens are
/// rotated by a small set of *rational* angles — atan2(b, a) for small
/// integers a, b — because transforming an already-periodic integer cell
/// coordinate by the integer matrix `[[a,-b],[b,a]]` is *automatically*
/// exactly periodic in the original coordinate, for any a and b (shifting
/// x by one period shifts the transformed x and y by `a` and `b` whole
/// periods respectively — still an exact multiple, so nothing misaligns at
/// the wrap edge). The tradeoff real screens accept too: the transform's
/// determinant (a² + b²) scales the effective cell size, so steeper angles
/// come out a bit denser.
pub const ROTATION_NAMES: [&str; 4] = ["0°", "18.4°", "26.6°", "45°"];

fn rotation_ab(name: &str) -> (i32, i32) {
    match name {
        "18.4°" => (3, 1),
        "26.6°" => (2, 1),
        "45°" => (1, 1),
        _ => (1, 0),
    }
}

pub struct DitherPatternParams {
    pub pattern: DitherPattern,
    pub cells_x: i32,
    pub cells_y: i32,
    pub rotation: (i32, i32),
    pub dot_size: f32,
    pub seed: u32,
}

impl Default for DitherPatternParams {
    fn default() -> Self {
        Self {
            pattern: DitherPattern::Bayer4x4,
            cells_x: 64,
            cells_y: 64,
            rotation: (1, 0),
            dot_size: 0.6,
            seed: 0,
        }
    }
}

impl DitherPatternParams {
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
            cells_x: params
                .get("cells_x")
                .and_then(mc_core::ParamValue::as_int)
                .unwrap_or(defaults.cells_x),
            cells_y: params
                .get("cells_y")
                .and_then(mc_core::ParamValue::as_int)
                .unwrap_or(defaults.cells_y),
            rotation: params
                .get("rotation")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => Some(rotation_ab(name)),
                    _ => None,
                })
                .unwrap_or(defaults.rotation),
            dot_size: params
                .get("dot_size")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.dot_size),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &DitherPatternParams,
) {
    let shader_source = full_source("dither_pattern.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    // Rounding each axis independently to a multiple of the pattern's tile
    // size is enough to keep the rotated pattern exactly tileable too (see
    // `ROTATION_NAMES`'s doc comment) — the rotation matrix doesn't add any
    // extra divisibility requirement of its own.
    let unit = params.pattern.tile_size();
    let cells_x = crate::dither_common::round_up_to_multiple(params.cells_x, unit);
    let cells_y = crate::dither_common::round_up_to_multiple(params.cells_y, unit);

    let mut param_bytes = [0u8; 32];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&cells_x.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&cells_y.to_le_bytes());
    param_bytes[12..16].copy_from_slice(&params.pattern.as_u32().to_le_bytes());
    param_bytes[16..20].copy_from_slice(&params.rotation.0.to_le_bytes());
    param_bytes[20..24].copy_from_slice(&params.rotation.1.to_le_bytes());
    param_bytes[24..28].copy_from_slice(&params.dot_size.to_le_bytes());
    param_bytes[28..32].copy_from_slice(&params.seed.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dither_pattern_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let blue_noise_view = crate::dither_common::upload_blue_noise_mask(gpu, pool);

    let view = pool.get_or_create(
        &gpu.device,
        key,
        resolution,
        wgpu::TextureFormat::Rgba8Unorm,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dither_pattern_bind_group"),
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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&blue_noise_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dither_pattern_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dither_pattern_pass"),
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
    use super::{dispatch, rotation_ab, DitherPattern, DitherPatternParams, ROTATION_NAMES};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &DitherPatternParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("dither pattern texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn tiles_seamlessly_for_every_pattern_and_rotation() {
        for &pattern in &[
            DitherPattern::Bayer2x2,
            DitherPattern::Bayer4x4,
            DitherPattern::Bayer8x8,
            DitherPattern::BlueNoise,
            DitherPattern::Dot,
        ] {
            for rotation_name in ROTATION_NAMES {
                assert_tiles_seamlessly(
                    &render(&DitherPatternParams {
                        pattern,
                        rotation: rotation_ab(rotation_name),
                        ..DitherPatternParams::default()
                    }),
                    RESOLUTION,
                );
            }
        }
    }

    /// A `Cells X`/`Cells Y` value that isn't already a multiple of the
    /// pattern's tile size must still tile — `dispatch` rounds it up before
    /// it ever reaches the shader. Deliberately anisotropic (different X and
    /// Y) to also cover that axes round independently.
    #[test]
    fn unrounded_anisotropic_cells_still_tiles_seamlessly() {
        assert_tiles_seamlessly(
            &render(&DitherPatternParams {
                pattern: DitherPattern::Bayer8x8,
                cells_x: 50,
                cells_y: 37,
                ..DitherPatternParams::default()
            }),
            RESOLUTION,
        );
    }

    #[test]
    fn is_deterministic() {
        let params = DitherPatternParams {
            pattern: DitherPattern::BlueNoise,
            seed: 3,
            ..DitherPatternParams::default()
        };
        assert_eq!(render(&params), render(&params));
    }

    #[test]
    fn blue_noise_seed_changes_output() {
        let base = DitherPatternParams {
            pattern: DitherPattern::BlueNoise,
            ..DitherPatternParams::default()
        };
        let reseeded = DitherPatternParams {
            pattern: DitherPattern::BlueNoise,
            seed: 5,
            ..DitherPatternParams::default()
        };
        assert_ne!(render(&base), render(&reseeded));
    }

    #[test]
    fn rotation_changes_bayer_output() {
        let unrotated = render(&DitherPatternParams {
            pattern: DitherPattern::Bayer4x4,
            ..DitherPatternParams::default()
        });
        let rotated = render(&DitherPatternParams {
            pattern: DitherPattern::Bayer4x4,
            rotation: rotation_ab("45°"),
            ..DitherPatternParams::default()
        });
        assert_ne!(unrotated, rotated);
    }

    #[test]
    fn bayer_2x2_on_a_flat_cell_grid_is_a_50_50_checker() {
        // At `cells = resolution` every texel is its own Bayer cell, so a
        // 2x2 matrix's 4 distinct threshold levels appear in a perfectly
        // regular, repeating pattern — in particular, exactly 2 of every 4
        // texels fall above the midpoint and 2 below, the ordered-dither
        // analogue of Threshold's "flat input still produces a 50/50 split"
        // test.
        let pixels = render(&DitherPatternParams {
            pattern: DitherPattern::Bayer2x2,
            cells_x: RESOLUTION as i32,
            cells_y: RESOLUTION as i32,
            ..DitherPatternParams::default()
        });
        let above_mid = pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[0] as f32 / 255.0 > 0.5)
            .count();
        let total = (RESOLUTION * RESOLUTION) as usize;
        assert_eq!(
            above_mid,
            total / 2,
            "Bayer 2x2 must split exactly half its texels above the midpoint"
        );
    }

    #[test]
    fn larger_dot_size_covers_more_of_the_texture() {
        // `cells_x`/`cells_y` deliberately well below `RESOLUTION` (64) so
        // each cell spans several texels — at the default 64 cells on a 64
        // texel test texture, every cell is exactly 1 texel and always
        // samples dead center, which can't tell dot sizes apart at all.
        let small = render(&DitherPatternParams {
            pattern: DitherPattern::Dot,
            cells_x: 8,
            cells_y: 8,
            dot_size: 0.1,
            ..DitherPatternParams::default()
        });
        let large = render(&DitherPatternParams {
            pattern: DitherPattern::Dot,
            cells_x: 8,
            cells_y: 8,
            dot_size: 0.9,
            ..DitherPatternParams::default()
        });
        let white_count = |pixels: &[u8]| {
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|px| px[0] as f32 / 255.0 > 0.5)
                .count()
        };
        assert!(
            white_count(&large) > white_count(&small),
            "a bigger Dot Size should cover more of the texture in white"
        );
    }

    #[test]
    fn every_pattern_name_round_trips() {
        for name in DitherPattern::NAMES {
            assert!(
                DitherPattern::from_name(name).is_some(),
                "{name} should parse back to a DitherPattern"
            );
        }
    }
}
