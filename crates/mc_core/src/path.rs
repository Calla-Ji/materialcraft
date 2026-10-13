// SPDX-License-Identifier: MIT OR Apache-2.0
//! The `Path` node's data: an editable vector ornament (PATH_NODE_SPEC.md
//! §2), ground-truthed by `path_reference.py`'s `Point`/`SubPath`/`Path`
//! dataclasses — field names here match that file's `to_json()` output
//! exactly (`pos`, `h_in`, `h_out`, `kind`, `width`; `points`, `closed`;
//! `subpaths`, `wrap`), since that JSON shape is this node's own saved
//! parameter and `golden/demo_path.json` round-trips through it.
//!
//! This module is deliberately just the data shape (mirrors how
//! [`crate::GradientStop`]/[`crate::ImageRef`] back their own
//! `ParamValue` variants) — the geometry that operates on it (`sections`,
//! `insert_point`, `drag_handle`, `set_kind`, `nearest_on_path`, all
//! `kurbo`-backed per PATH_NODE_SPEC.md §6) lives in `mc_nodes::path`,
//! same split as every other node's domain logic living in `mc_nodes`
//! rather than here.

use serde::{Deserialize, Serialize};

/// How a point's two handles move relative to each other while dragging
/// (PATH_NODE_SPEC.md §3) — enforced by `mc_nodes::path::drag_handle`/
/// `set_kind`, not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PointKind {
    /// Handles independent — a sharp angle.
    #[default]
    Corner,
    /// The other handle rotates to stay exactly opposite, keeping its own
    /// length.
    Smooth,
    /// The other handle is the exact mirror: same length, opposite
    /// direction.
    Symmetric,
}

/// One anchor on a [`SubPath`]. `pos`/`h_in`/`h_out` are tile units (`x`
/// right, `y` down, same convention as every raster output in this
/// codebase) — handles are stored as offsets *relative to* `pos`, never as
/// absolute positions, so moving a point carries its handles along for
/// free (PATH_NODE_SPEC.md §2's own stated reason). `width` is a
/// multiplier (default `1.0`) used by width-aware stitches (satin,
/// zigzag) — meaningless on its own without a downstream consumer, but
/// every point carries one regardless.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub pos: [f64; 2],
    #[serde(default)]
    pub h_in: [f64; 2],
    #[serde(default)]
    pub h_out: [f64; 2],
    #[serde(default)]
    pub kind: PointKind,
    #[serde(default = "default_width")]
    pub width: f32,
}

fn default_width() -> f32 {
    1.0
}

impl Point {
    pub fn new(pos: [f64; 2]) -> Self {
        Self {
            pos,
            h_in: [0.0, 0.0],
            h_out: [0.0, 0.0],
            kind: PointKind::Corner,
            width: 1.0,
        }
    }
}

// f64/f32 don't implement `Hash` (no reflexive NaN equality), so `Point`
// needs a manual impl — same reasoning and the same `to_bits()` technique
// `mc_core::ParamValue`'s own `Hash` impl already uses for `Color`/`Vec2`.
// `SubPath`/`Path` derive `Hash` straight through this, since `Vec<T>` is
// `Hash` whenever `T` is.
impl std::hash::Hash for Point {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.h_in[0].to_bits().hash(state);
        self.h_in[1].to_bits().hash(state);
        self.h_out[0].to_bits().hash(state);
        self.h_out[1].to_bits().hash(state);
        self.kind.hash(state);
        self.width.to_bits().hash(state);
    }
}

/// One contiguous run of points, open or closed. Segment `i` is the cubic
/// Bezier `P0 = points[i].pos`, `P1 = points[i].pos + points[i].h_out`,
/// `P2 = points[i+1].pos + points[i+1].h_in`, `P3 = points[i+1].pos`; a
/// closed subpath has one more segment running from the last point back to
/// the first. A zero handle makes that end of the segment straight (a
/// straight line if both ends are zero).
#[derive(Debug, Clone, PartialEq, Hash, Serialize, Deserialize)]
pub struct SubPath {
    pub points: Vec<Point>,
    #[serde(default)]
    pub closed: bool,
}

impl SubPath {
    pub fn new(points: Vec<Point>, closed: bool) -> Self {
        Self { points, closed }
    }
}

/// The Path node's entire output/parameter value (PATH_NODE_SPEC.md §1-2):
/// a list of subpaths plus `Wrap`. With `wrap` on, a point's `pos` may lie
/// outside `[0, 1)` — it reappears on the tile's other side, same
/// convention every other seamless-tiling node in this codebase already
/// follows, just for vector data instead of a raster sample.
#[derive(Debug, Clone, PartialEq, Hash, Serialize, Deserialize)]
pub struct Path {
    #[serde(default)]
    pub subpaths: Vec<SubPath>,
    #[serde(default = "default_wrap")]
    pub wrap: bool,
}

fn default_wrap() -> bool {
    true
}

impl Default for Path {
    fn default() -> Self {
        Self {
            subpaths: Vec::new(),
            wrap: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_kind_json_names_match_the_python_reference() {
        assert_eq!(
            serde_json::to_string(&PointKind::Corner).unwrap(),
            "\"corner\""
        );
        assert_eq!(
            serde_json::to_string(&PointKind::Smooth).unwrap(),
            "\"smooth\""
        );
        assert_eq!(
            serde_json::to_string(&PointKind::Symmetric).unwrap(),
            "\"symmetric\""
        );
    }

    #[test]
    fn point_field_names_match_the_python_reference() {
        let p = Point {
            pos: [0.1, 0.2],
            h_in: [-0.1, 0.0],
            h_out: [0.1, 0.0],
            kind: PointKind::Smooth,
            width: 0.7,
        };
        let v: serde_json::Value = serde_json::to_value(p).unwrap();
        for key in ["pos", "h_in", "h_out", "kind", "width"] {
            assert!(v.get(key).is_some(), "missing field {key:?}");
        }
    }

    #[test]
    fn equal_paths_hash_equal_and_a_tiny_nudge_changes_the_hash() {
        use std::hash::{Hash, Hasher};
        fn hash_of(p: &Path) -> u64 {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            p.hash(&mut hasher);
            hasher.finish()
        }
        let a = Path {
            subpaths: vec![SubPath::new(vec![Point::new([0.1, 0.2])], false)],
            wrap: true,
        };
        let b = a.clone();
        assert_eq!(hash_of(&a), hash_of(&b));

        let mut c = a.clone();
        c.subpaths[0].points[0].pos[0] += 1e-6;
        assert_ne!(hash_of(&a), hash_of(&c));
    }
}
