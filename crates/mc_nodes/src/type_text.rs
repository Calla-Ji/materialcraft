// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Type.
//!
//! Display name: "Type". Tooltip: "Renders text, in any font installed on
//! this machine, as a Gray mask."
//!
//! Unlike every other node, this one does real system font discovery and
//! rasterization at dispatch time instead of running a WGSL compute
//! shader — there's no portable way to do font shaping/rasterization on
//! the GPU from this codebase's own primitives, and `font-kit` already
//! wraps the real platform text engines (Core Text on macOS, DirectWrite
//! on Windows, FreeType/fontconfig on Linux) to do it correctly. Same
//! "real file I/O/decoding at dispatch time" precedent as Image Import.

use font_kit::canvas::{Canvas, Format, RasterizationOptions};
use font_kit::family_name::FamilyName;
use font_kit::hinting::HintingOptions;
use font_kit::properties::Properties;
use font_kit::source::SystemSource;
use pathfinder_geometry::transform2d::Transform2F;
use pathfinder_geometry::vector::{Vector2F, Vector2I};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HorizontalAlign {
    Left,
    Center,
    Right,
}

impl HorizontalAlign {
    pub const NAMES: [&'static str; 3] = ["Left", "Center", "Right"];

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "Left" => Some(Self::Left),
            "Center" => Some(Self::Center),
            "Right" => Some(Self::Right),
            _ => None,
        }
    }
}

pub struct TypeTextParams {
    pub text: String,
    /// Empty means "no specific family requested" — [`dispatch`] falls
    /// back to a generic sans-serif rather than failing, same as an
    /// unrecognized family name that isn't actually installed here (e.g.
    /// a graph saved on a different machine).
    pub font_family: String,
    /// Tile units (fraction of the tile's own height, same convention as
    /// every other size param in the app) — converted to a real point
    /// size at dispatch time via the current resolution.
    pub size: f32,
    /// Extra spacing added after every glyph, tile units.
    pub tracking: f32,
    pub align: HorizontalAlign,
}

impl Default for TypeTextParams {
    fn default() -> Self {
        Self {
            text: "Type".to_string(),
            font_family: String::new(),
            size: 0.2,
            tracking: 0.0,
            align: HorizontalAlign::Center,
        }
    }
}

impl TypeTextParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            text: params
                .get("text")
                .and_then(mc_core::ParamValue::as_text)
                .map(str::to_string)
                .unwrap_or(defaults.text),
            font_family: params
                .get("font_family")
                .and_then(mc_core::ParamValue::as_text)
                .map(str::to_string)
                .unwrap_or(defaults.font_family),
            size: params
                .get("size")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.size),
            tracking: params
                .get("tracking")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.tracking),
            align: params
                .get("align")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => HorizontalAlign::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.align),
        }
    }
}

/// Every installed font family name, sorted and deduplicated — for the
/// Type node's own font picker (`params_panel.rs`). `font-kit`'s own
/// `all_families` can legitimately fail (e.g. a sandboxed or headless
/// environment with no font database at all); an empty list degrades to
/// just the "(System Default)" option that picker already always shows,
/// not a crash.
pub fn list_font_families() -> Vec<String> {
    let mut families = SystemSource::new().all_families().unwrap_or_default();
    families.sort();
    families.dedup();
    families
}

/// Loads `family` (or, if empty or not found on this machine, a generic
/// sans-serif fallback via `font-kit`'s own CSS-style family matching —
/// the same "fall back to a sensible default instead of failing" rule
/// every other node's optional/malformed input already follows).
fn load_font(family: &str) -> Option<font_kit::font::Font> {
    let source = SystemSource::new();
    let mut candidates = Vec::new();
    if !family.is_empty() {
        candidates.push(FamilyName::Title(family.to_string()));
    }
    candidates.push(FamilyName::SansSerif);
    source
        .select_best_match(&candidates, &Properties::new())
        .ok()?
        .load()
        .ok()
}

/// Renders `params.text` into `key`'s texture as a Gray mask — centered
/// vertically in the tile (ascent/descent-balanced, the usual "centered
/// text" convention), horizontally per `params.align`. Antialiased
/// (`RasterizationOptions::GrayscaleAa`), same soft-edge-by-default
/// convention every other mask-producing node in this codebase already
/// uses (e.g. Shape's own default Softness).
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &TypeTextParams,
) {
    let Some(font) = load_font(&params.font_family) else {
        // No usable font at all on this machine — vanishingly unlikely
        // (even a minimal system has *some* installed font `font-kit`'s
        // own generic-family matching can find), but degrade to a blank
        // mask rather than panicking (CLAUDE.md §12).
        gpu.fill_rgba8(pool, key, resolution, [0, 0, 0, 255]);
        return;
    };

    let metrics = font.metrics();
    let units_per_em = (metrics.units_per_em.max(1)) as f32;
    let point_size = (params.size * resolution as f32).max(1.0);
    let scale = point_size / units_per_em;
    let tracking_px = params.tracking * resolution as f32;
    // A glyph `font-kit` can't find (an unsupported character, most
    // often) still needs *some* advance so the rest of the string doesn't
    // bunch up on top of it — approximated as a typical space width
    // rather than left at zero.
    let fallback_advance_px = point_size * 0.3;

    let mut glyphs: Vec<(u32, f32)> = Vec::with_capacity(params.text.chars().count());
    let mut cursor_x = 0.0f32;
    for ch in params.text.chars() {
        match font.glyph_for_char(ch) {
            Some(glyph_id) => {
                glyphs.push((glyph_id, cursor_x));
                let advance_px = font
                    .advance(glyph_id)
                    .map(|v| v.x() * scale)
                    .unwrap_or(fallback_advance_px);
                cursor_x += advance_px + tracking_px;
            }
            None => cursor_x += fallback_advance_px + tracking_px,
        }
    }
    // The last glyph's own trailing advance (plus one tracking gap that
    // never actually separates it from a *next* glyph) isn't part of the
    // text's own visible width — trimmed back off before centering/
    // right-aligning against it.
    let last_width = glyphs
        .last()
        .map(|&(glyph_id, x)| x + font.advance(glyph_id).map(|v| v.x() * scale).unwrap_or(0.0))
        .unwrap_or(0.0);
    let total_width = last_width.max(0.0);

    let align_offset = match params.align {
        HorizontalAlign::Left => 0.0,
        HorizontalAlign::Center => (resolution as f32 - total_width) / 2.0,
        HorizontalAlign::Right => resolution as f32 - total_width,
    };

    // font-kit's own `Metrics`: `ascent` is positive (above the
    // baseline), `descent` is *negative* (below it) — so the total
    // cap-to-descender height is `ascent - descent`, not `ascent +
    // descent`.
    let ascent_px = metrics.ascent * scale;
    let descent_px = metrics.descent * scale;
    let text_height = ascent_px - descent_px;
    let top_y = (resolution as f32 - text_height) / 2.0;
    let baseline_y = top_y + ascent_px;

    let mut canvas = Canvas::new(
        Vector2I::new(resolution as i32, resolution as i32),
        Format::A8,
    );
    for (glyph_id, x) in glyphs {
        let transform = Transform2F::from_translation(Vector2F::new(x + align_offset, baseline_y));
        // A glyph `font-kit` can't rasterize at this point (extremely
        // rare once `glyph_for_char` already succeeded) just doesn't
        // draw — not worth failing the whole node over one character.
        let _ = font.rasterize_glyph(
            &mut canvas,
            glyph_id,
            point_size,
            transform,
            HintingOptions::None,
            RasterizationOptions::GrayscaleAa,
        );
    }

    // `Canvas::new`'s own stride is tightly packed (`size.x() *
    // bytes_per_pixel`, no row padding) for `Format::A8` (1 byte/pixel),
    // so `canvas.pixels` is already exactly `resolution * resolution`
    // bytes, row-major, top-left origin (font-kit's own doc comment on
    // `rasterize_glyph`: deliberately flipped from Core Graphics' native
    // bottom-left convention "to make behavior consistent" across
    // backends) — matching every other node's own `(id.xy + 0.5) /
    // resolution` convention with no further flipping needed here.
    let pixels: Vec<[f32; 4]> = canvas
        .pixels
        .iter()
        .map(|&a| {
            let v = a as f32 / 255.0;
            [v, v, v, 1.0]
        })
        .collect();
    gpu.write_rgba16f(pool, key, resolution, &pixels);
}

#[cfg(test)]
mod tests {
    use super::{dispatch, HorizontalAlign, TypeTextParams};
    use crate::test_support::RESOLUTION;

    fn render(params: &TypeTextParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("type_text texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn empty_text_is_blank() {
        let params = TypeTextParams {
            text: String::new(),
            ..TypeTextParams::default()
        };
        let pixels = render(&params);
        assert!(
            pixels.as_chunks::<4>().0.iter().all(|p| p[0] == 0),
            "an empty string should draw nothing"
        );
    }

    #[test]
    fn real_text_draws_something() {
        let pixels = render(&TypeTextParams::default());
        assert!(
            pixels.as_chunks::<4>().0.iter().any(|p| p[0] > 10),
            "default text (\"Type\") should draw visible pixels"
        );
    }

    #[test]
    fn is_deterministic() {
        let params = TypeTextParams::default();
        assert_eq!(render(&params), render(&params));
    }

    #[test]
    fn left_right_and_center_align_actually_differ() {
        let base = TypeTextParams {
            text: "A".to_string(),
            size: 0.3,
            ..TypeTextParams::default()
        };
        let left = render(&TypeTextParams {
            align: HorizontalAlign::Left,
            ..TypeTextParams {
                text: base.text.clone(),
                font_family: base.font_family.clone(),
                size: base.size,
                tracking: base.tracking,
                align: base.align,
            }
        });
        let right = render(&TypeTextParams {
            align: HorizontalAlign::Right,
            ..TypeTextParams {
                text: base.text.clone(),
                font_family: base.font_family.clone(),
                size: base.size,
                tracking: base.tracking,
                align: base.align,
            }
        });
        assert_ne!(
            left, right,
            "left- and right-aligned single-character text should land in visibly different places"
        );
    }

    #[test]
    fn unknown_font_family_falls_back_instead_of_blanking() {
        let params = TypeTextParams {
            font_family: "Definitely Not A Real Font Family Name".to_string(),
            ..TypeTextParams::default()
        };
        let pixels = render(&params);
        assert!(
            pixels.as_chunks::<4>().0.iter().any(|p| p[0] > 10),
            "an unrecognized family should still fall back to a real font, not blank out"
        );
    }

    #[test]
    fn list_font_families_is_sorted_and_deduplicated() {
        let families = super::list_font_families();
        let mut sorted = families.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(families, sorted);
    }
}
