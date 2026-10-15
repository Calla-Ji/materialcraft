// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Path to Mask.
//!
//! Turns a `Path` into a real, usable `Gray` output — something Preview
//! Mask was never meant to be (it's a fixed-width editor affordance, not a
//! real node output). Two modes: Outline reuses `path::dispatch_stroke_mask`
//! directly (the exact rasterizer Preview Mask already uses, just with a
//! real, user-controlled width); Fill is new — a CPU scanline, nonzero-
//! winding polygon fill, since nothing in the codebase rasterizes a filled
//! region yet (everything else only strokes paths). The owner's own
//! request: a separate node rather than folding Fill/Outline into the Path
//! node itself, so Path stays purely about curve editing and stamp
//! distribution.

use mc_core::Path;

use crate::path::preview_polyline;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskMode {
    Fill,
    Outline,
}

impl MaskMode {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "Fill" => Some(Self::Fill),
            "Outline" => Some(Self::Outline),
            _ => None,
        }
    }
}

pub struct MaskParams {
    pub mode: MaskMode,
    /// Tile-UV units, like every other size param in the app — not
    /// pixels, so the outline stays the same relative width regardless of
    /// export resolution (CLAUDE.md §6, "resolution independent").
    pub outline_width: f32,
}

impl Default for MaskParams {
    fn default() -> Self {
        Self {
            mode: MaskMode::Fill,
            outline_width: 0.01,
        }
    }
}

impl MaskParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            mode: params
                .get("mode")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => MaskMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.mode),
            outline_width: params
                .get("outline_width")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.outline_width),
        }
    }
}

/// Dispatches `path` into `key`'s texture per `params.mode`. Outline is a
/// thin wrapper over `path::dispatch_stroke_mask`; Fill rasterizes on the
/// CPU and uploads directly — see [`fill_mask_cpu`] for the algorithm.
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    path: &Path,
    params: &MaskParams,
) {
    match params.mode {
        MaskMode::Outline => {
            crate::path::dispatch_stroke_mask(
                gpu,
                pool,
                key,
                resolution,
                path,
                params.outline_width,
            );
        }
        MaskMode::Fill => {
            let mask = fill_mask_cpu(path, resolution);
            let pixels: Vec<[f32; 4]> = mask.iter().map(|&v| [v, v, v, 1.0]).collect();
            gpu.write_rgba16f(pool, key, resolution, &pixels);
        }
    }
}

const FILL_SAMPLES_PER_SEGMENT: usize = 32;

/// Every subpath's polyline, duplicated across the 3x3 tile neighborhood
/// (`dx, dy` in `{-1, 0, 1}`) so a shape straddling the tile edge still
/// fills correctly on both sides — the same toroidal trick
/// `tube_raster`'s own `wrap_cell` binning achieves for strokes, done here
/// by brute-force replication instead, since the fill algorithm below has
/// no spatial grid of its own to wrap through. `path.wrap == false` isn't
/// specially handled, matching `dispatch_stroke_mask`'s own documented gap.
fn wrapped_polygons(path: &Path) -> Vec<Vec<[f64; 2]>> {
    let mut out = Vec::new();
    for sp in &path.subpaths {
        let base = preview_polyline(sp, FILL_SAMPLES_PER_SEGMENT);
        if base.len() < 2 {
            continue;
        }
        for dx in [-1.0, 0.0, 1.0] {
            for dy in [-1.0, 0.0, 1.0] {
                out.push(base.iter().map(|p| [p[0] + dx, p[1] + dy]).collect());
            }
        }
    }
    out
}

/// Nonzero-winding scanline fill. Every subpath is treated as implicitly
/// closed for fill purposes (the last point connects straight back to the
/// first) even if its own `closed` flag is false — the same convention
/// every vector tool uses when you fill an open path. Nonzero winding
/// (not even-odd) is the correct rule here, not an arbitrary pick: a
/// traced shape with a hole (Image Trace's own `decompose` stage) encodes
/// that hole purely as an oppositely-wound subpath, which nonzero winding
/// reconstructs correctly from geometry alone — no extra "is this a hole"
/// data needed.
fn fill_mask_cpu(path: &Path, resolution: u32) -> Vec<f32> {
    let polygons = wrapped_polygons(path);
    let size = resolution as usize;
    let mut mask = vec![0.0f32; size * size];
    if polygons.is_empty() {
        return mask;
    }

    for py in 0..size {
        let y = (py as f64 + 0.5) / resolution as f64;
        let mut crossings: Vec<(f64, i32)> = Vec::new();
        for poly in &polygons {
            let n = poly.len();
            for i in 0..n {
                let a = poly[i];
                let b = poly[(i + 1) % n];
                let (y0, y1) = (a[1], b[1]);
                if (y0 <= y) != (y1 <= y) {
                    let t = (y - y0) / (y1 - y0);
                    let x = a[0] + t * (b[0] - a[0]);
                    let dir = if y1 > y0 { 1 } else { -1 };
                    crossings.push((x, dir));
                }
            }
        }
        if crossings.is_empty() {
            continue;
        }
        crossings.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let row = &mut mask[py * size..(py + 1) * size];
        let mut winding = 0i32;
        for i in 0..crossings.len() {
            winding += crossings[i].1;
            if winding != 0 && i + 1 < crossings.len() {
                fill_span(row, resolution, crossings[i].0, crossings[i + 1].0);
            }
        }
    }
    mask
}

/// Fills pixel-center columns whose UV x falls in `[x0, x1)`.
fn fill_span(row: &mut [f32], resolution: u32, x0: f64, x1: f64) {
    let size = resolution as f64;
    let end_f = (x1 * size - 0.5).ceil();
    if end_f <= 0.0 {
        return;
    }
    let start = ((x0 * size - 0.5).ceil()).max(0.0) as usize;
    let end = (end_f as usize).min(row.len());
    for px in row.iter_mut().take(end).skip(start) {
        *px = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_core::{Point, PointKind, SubPath};

    fn square_path(x0: f64, y0: f64, x1: f64, y1: f64) -> Path {
        fn pt(x: f64, y: f64) -> Point {
            Point {
                pos: [x, y],
                h_in: [0.0, 0.0],
                h_out: [0.0, 0.0],
                kind: PointKind::Corner,
                width: 1.0,
            }
        }
        Path {
            subpaths: vec![SubPath::new(
                vec![pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1)],
                true,
            )],
            wrap: true,
        }
    }

    #[test]
    fn mode_names_round_trip() {
        assert_eq!(MaskMode::from_name("Fill"), Some(MaskMode::Fill));
        assert_eq!(MaskMode::from_name("Outline"), Some(MaskMode::Outline));
        assert_eq!(MaskMode::from_name("Bogus"), None);
    }

    #[test]
    fn fill_is_blank_for_an_empty_path() {
        let mask = fill_mask_cpu(&Path::default(), 16);
        assert!(mask.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fill_covers_the_interior_of_a_square_and_leaves_the_outside_blank() {
        let path = square_path(0.25, 0.25, 0.75, 0.75);
        let mask = fill_mask_cpu(&path, 64);
        let size = 64usize;
        let at = |px: usize, py: usize| mask[py * size + px];

        // Center: well inside the square.
        assert_eq!(at(32, 32), 1.0, "center should be filled");
        // Corners: well outside.
        assert_eq!(at(2, 2), 0.0, "corner should be blank");
        assert_eq!(at(61, 61), 0.0, "corner should be blank");
    }

    #[test]
    fn fill_leaves_a_hole_for_an_oppositely_wound_inner_subpath() {
        fn pt(x: f64, y: f64) -> Point {
            Point {
                pos: [x, y],
                h_in: [0.0, 0.0],
                h_out: [0.0, 0.0],
                kind: PointKind::Corner,
                width: 1.0,
            }
        }
        // Outer ring CCW, inner ring CW (opposite winding) — nonzero
        // winding should fill the ring but leave the inner hole blank.
        let outer = SubPath::new(
            vec![pt(0.1, 0.1), pt(0.9, 0.1), pt(0.9, 0.9), pt(0.1, 0.9)],
            true,
        );
        let inner = SubPath::new(
            vec![pt(0.4, 0.4), pt(0.4, 0.6), pt(0.6, 0.6), pt(0.6, 0.4)],
            true,
        );
        let path = Path {
            subpaths: vec![outer, inner],
            wrap: true,
        };
        let mask = fill_mask_cpu(&path, 64);
        let size = 64usize;
        let at = |px: usize, py: usize| mask[py * size + px];

        assert_eq!(at(32, 32), 0.0, "hole in the middle should stay blank");
        assert_eq!(at(10, 32), 1.0, "ring body should be filled");
    }

    #[test]
    fn fill_wraps_a_shape_straddling_the_tile_edge() {
        // A square centered on x=0 (straddling the left/right wrap
        // boundary once normalized into [0,1)) should fill on both the
        // far-left and far-right columns of the tile.
        let path = square_path(-0.1, 0.25, 0.1, 0.75);
        let mask = fill_mask_cpu(&path, 64);
        let size = 64usize;
        let at = |px: usize, py: usize| mask[py * size + px];

        assert_eq!(at(63, 32), 1.0, "right edge should be filled by the wrap");
        assert_eq!(at(0, 32), 1.0, "left edge should be filled directly");
    }

    #[test]
    fn fill_is_deterministic() {
        let path = square_path(0.2, 0.2, 0.8, 0.8);
        let a = fill_mask_cpu(&path, 32);
        let b = fill_mask_cpu(&path, 32);
        assert_eq!(a, b);
    }
}
