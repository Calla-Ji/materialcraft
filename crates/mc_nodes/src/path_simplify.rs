// SPDX-License-Identifier: MIT OR Apache-2.0
//! Path Simplify — reduces a Path's own point count via Ramer-Douglas-
//! Peucker vertex decimation, independent of where the path came from
//! (hand-drawn, SVG Import, or Image Trace). Points marked
//! `PointKind::Corner` always survive regardless of Tolerance (the
//! owner's own call: a deliberate sharp corner shouldn't get silently
//! rounded off just because it's geometrically "redundant" within
//! tolerance) — the standard distance-to-chord test is overridden to
//! force a split at the first corner found whenever the plain test alone
//! would otherwise have left a range unsplit.
//!
//! A closed subpath has no natural start/end for the classic
//! open-polyline algorithm, so it's split first: an anchor (the
//! subpath's own first corner if it has one, else its first point —
//! always kept) and the point farthest from it (also always kept)
//! divide the loop into two open arcs, each independently run through
//! the same algorithm.
//!
//! Kept points retain their *original* position, handles, and kind
//! unchanged — no curve re-fitting. That's the cheapest, most
//! predictable approximation, and it keeps a moderately-simplified curve
//! looking like itself rather than going polygonal.

use mc_core::{Path, Point, PointKind, SubPath};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimplifyParams {
    pub tolerance: f64,
}

impl Default for SimplifyParams {
    fn default() -> Self {
        Self { tolerance: 0.002 }
    }
}

impl SimplifyParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            tolerance: params
                .get("tolerance")
                .and_then(mc_core::ParamValue::as_float)
                .map(f64::from)
                .unwrap_or(defaults.tolerance),
        }
    }
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// Perpendicular distance from `p` to the line through `a`/`b` — falls
/// back to plain point-to-point distance if `a` and `b` coincide (a
/// zero-length chord has no well-defined perpendicular).
fn perpendicular_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = sub(b, a);
    let len = (ab[0] * ab[0] + ab[1] * ab[1]).sqrt();
    let ap = sub(p, a);
    if len < 1e-12 {
        return (ap[0] * ap[0] + ap[1] * ap[1]).sqrt();
    }
    (ab[0] * ap[1] - ab[1] * ap[0]).abs() / len
}

/// Ramer-Douglas-Peucker over `idx[lo..=hi]` (indices into `points`),
/// with `idx[lo]`/`idx[hi]` already marked kept by the caller. Splits at
/// the point farthest from the `idx[lo]`-`idx[hi]` chord when that
/// distance exceeds `tol`; otherwise, if any corner point lies in the
/// open range, splits there instead (forcing it to survive, with
/// preference for the corner farthest from the chord if there's more
/// than one) even though the plain distance test alone wouldn't have
/// kept anything.
fn dp_recurse(points: &[Point], idx: &[usize], lo: usize, hi: usize, tol: f64, keep: &mut [bool]) {
    if hi <= lo + 1 {
        return;
    }
    let a = points[idx[lo]].pos;
    let b = points[idx[hi]].pos;
    let mut max_dist = -1.0_f64;
    let mut max_at = lo + 1;
    let mut best_corner: Option<usize> = None;
    let mut best_corner_dist = -1.0_f64;
    for k in (lo + 1)..hi {
        let d = perpendicular_distance(points[idx[k]].pos, a, b);
        if d > max_dist {
            max_dist = d;
            max_at = k;
        }
        if points[idx[k]].kind == PointKind::Corner && d > best_corner_dist {
            best_corner_dist = d;
            best_corner = Some(k);
        }
    }
    let split = if max_dist > tol {
        max_at
    } else if let Some(k) = best_corner {
        k
    } else {
        return;
    };
    keep[idx[split]] = true;
    dp_recurse(points, idx, lo, split, tol, keep);
    dp_recurse(points, idx, split, hi, tol, keep);
}

/// Indices walking the ring from `from` to `to` inclusive, forward
/// (wrapping mod `n`) — one of the two arcs a closed subpath's anchor/far
/// split divides it into.
fn ring_range(from: usize, to: usize, n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut i = from;
    loop {
        out.push(i);
        if i == to {
            break;
        }
        i = (i + 1) % n;
    }
    out
}

fn keep_mask(sp: &SubPath, tolerance: f64) -> Vec<bool> {
    let n = sp.points.len();
    // Nothing meaningful to simplify below 4 points either way.
    if n <= 3 {
        return vec![true; n];
    }

    let mut keep = vec![false; n];
    if !sp.closed {
        keep[0] = true;
        keep[n - 1] = true;
        let idx: Vec<usize> = (0..n).collect();
        dp_recurse(&sp.points, &idx, 0, n - 1, tolerance, &mut keep);
        return keep;
    }

    let anchor = sp
        .points
        .iter()
        .position(|p| p.kind == PointKind::Corner)
        .unwrap_or(0);
    let mut far = (anchor + 1) % n;
    let mut far_dist = -1.0_f64;
    for i in 0..n {
        if i == anchor {
            continue;
        }
        let diff = sub(sp.points[i].pos, sp.points[anchor].pos);
        let d = (diff[0] * diff[0] + diff[1] * diff[1]).sqrt();
        if d > far_dist {
            far_dist = d;
            far = i;
        }
    }
    keep[anchor] = true;
    keep[far] = true;

    let arc1 = ring_range(anchor, far, n);
    dp_recurse(&sp.points, &arc1, 0, arc1.len() - 1, tolerance, &mut keep);
    let arc2 = ring_range(far, anchor, n);
    dp_recurse(&sp.points, &arc2, 0, arc2.len() - 1, tolerance, &mut keep);

    keep
}

/// Simplifies every subpath of `path` independently — see this module's
/// own doc comment for the algorithm.
pub fn simplify(path: &Path, tolerance: f64) -> Path {
    let subpaths = path
        .subpaths
        .iter()
        .map(|sp| {
            let keep = keep_mask(sp, tolerance);
            let points: Vec<Point> = sp
                .points
                .iter()
                .zip(keep)
                .filter_map(|(p, k)| k.then_some(*p))
                .collect();
            SubPath {
                points,
                closed: sp.closed,
            }
        })
        .collect();
    Path {
        subpaths,
        wrap: path.wrap,
    }
}

#[cfg(test)]
mod tests {
    use super::{simplify, SimplifyParams};
    use mc_core::{Path, Point, PointKind, SubPath};

    fn pt(x: f64, y: f64, kind: PointKind) -> Point {
        Point {
            pos: [x, y],
            h_in: [0.0, 0.0],
            h_out: [0.0, 0.0],
            kind,
            width: 1.0,
        }
    }

    #[test]
    fn default_tolerance_matches_the_struct_default() {
        assert_eq!(SimplifyParams::default().tolerance, 0.002);
        assert_eq!(
            SimplifyParams::from_graph(&Default::default()).tolerance,
            0.002
        );
    }

    #[test]
    fn a_negligible_bump_is_dropped_above_its_own_tolerance() {
        // Two near-colinear interior points, well within tolerance, on
        // an otherwise straight line — both should drop, keeping only
        // the two endpoints. (4 points, not 3: this module's own
        // "nothing to simplify" floor is <= 3.)
        let sp = SubPath {
            points: vec![
                pt(0.0, 0.0, PointKind::Smooth),
                pt(0.33, 0.001, PointKind::Smooth),
                pt(0.66, -0.001, PointKind::Smooth),
                pt(1.0, 0.0, PointKind::Smooth),
            ],
            closed: false,
        };
        let path = Path {
            subpaths: vec![sp],
            wrap: false,
        };
        let out = simplify(&path, 0.01);
        assert_eq!(
            out.subpaths[0].points.len(),
            2,
            "both near-colinear interior points should be dropped at a generous tolerance"
        );
        assert_eq!(out.subpaths[0].points[0].pos, [0.0, 0.0]);
        assert_eq!(out.subpaths[0].points[1].pos, [1.0, 0.0]);
    }

    #[test]
    fn a_real_bump_survives_a_tight_tolerance() {
        let sp = SubPath {
            points: vec![
                pt(0.0, 0.0, PointKind::Smooth),
                pt(0.5, 0.2, PointKind::Smooth),
                pt(1.0, 0.0, PointKind::Smooth),
                pt(1.0, 1.0, PointKind::Smooth),
            ],
            closed: false,
        };
        let path = Path {
            subpaths: vec![sp],
            wrap: false,
        };
        let out = simplify(&path, 0.001);
        assert_eq!(
            out.subpaths[0].points.len(),
            4,
            "a genuine 0.2-tall bump should survive a 0.001 tolerance"
        );
    }

    #[test]
    fn corner_points_always_survive_even_at_an_enormous_tolerance() {
        let sp = SubPath {
            points: vec![
                pt(0.0, 0.0, PointKind::Smooth),
                pt(0.25, 0.001, PointKind::Smooth),
                // A corner sitting almost exactly on the chord — the
                // plain distance test alone would drop it.
                pt(0.5, 0.0001, PointKind::Corner),
                pt(0.75, 0.001, PointKind::Smooth),
                pt(1.0, 0.0, PointKind::Smooth),
            ],
            closed: false,
        };
        let path = Path {
            subpaths: vec![sp],
            wrap: false,
        };
        let out = simplify(&path, 10.0);
        assert!(
            out.subpaths[0]
                .points
                .iter()
                .any(|p| p.kind == PointKind::Corner && p.pos == [0.5, 0.0001]),
            "the corner point must survive regardless of how loose the tolerance is"
        );
    }

    #[test]
    fn a_closed_near_circle_simplifies_but_stays_closed_and_non_trivial() {
        let n = 64;
        let points: Vec<Point> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64 * std::f64::consts::TAU;
                pt(t.cos(), t.sin(), PointKind::Smooth)
            })
            .collect();
        let path = Path {
            subpaths: vec![SubPath {
                points,
                closed: true,
            }],
            wrap: false,
        };
        let out = simplify(&path, 0.02);
        assert!(out.subpaths[0].closed);
        assert!(
            out.subpaths[0].points.len() < n,
            "a dense near-circle should lose some points at a non-trivial tolerance"
        );
        assert!(
            out.subpaths[0].points.len() >= 4,
            "a circle still needs a handful of points to read as round, not just 2"
        );
    }

    #[test]
    fn subpaths_with_3_or_fewer_points_pass_through_unchanged() {
        let sp = SubPath {
            points: vec![
                pt(0.0, 0.0, PointKind::Corner),
                pt(1.0, 0.0, PointKind::Corner),
                pt(0.5, 1.0, PointKind::Corner),
            ],
            closed: true,
        };
        let path = Path {
            subpaths: vec![sp.clone()],
            wrap: true,
        };
        let out = simplify(&path, 100.0);
        assert_eq!(out.subpaths[0].points.len(), sp.points.len());
    }

    #[test]
    fn is_deterministic() {
        let n = 40;
        let points: Vec<Point> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64 * std::f64::consts::TAU;
                pt(
                    t.cos() + 0.05 * (t * 7.0).sin(),
                    t.sin() + 0.05 * (t * 5.0).cos(),
                    PointKind::Smooth,
                )
            })
            .collect();
        let path = Path {
            subpaths: vec![SubPath {
                points,
                closed: true,
            }],
            wrap: false,
        };
        let a = simplify(&path, 0.03);
        let b = simplify(&path, 0.03);
        assert_eq!(a.subpaths[0].points.len(), b.subpaths[0].points.len());
        for (pa, pb) in a.subpaths[0].points.iter().zip(&b.subpaths[0].points) {
            assert_eq!(pa.pos, pb.pos);
        }
    }

    #[test]
    fn wrap_passes_through_unchanged() {
        let sp = SubPath {
            points: vec![
                pt(0.0, 0.0, PointKind::Smooth),
                pt(0.5, 0.5, PointKind::Smooth),
                pt(1.0, 0.0, PointKind::Smooth),
                pt(0.5, -0.5, PointKind::Smooth),
            ],
            closed: true,
        };
        let path = Path {
            subpaths: vec![sp],
            wrap: true,
        };
        assert!(simplify(&path, 0.001).wrap);
        let path = Path {
            wrap: false,
            ..path
        };
        assert!(!simplify(&path, 0.001).wrap);
    }

    /// An 11-point closed curve copied from a real Image Trace result (a
    /// Shape node's own disc, traced at 128x128) — a regression anchor
    /// for actual traced-curve geometry, not just synthetic points.
    #[test]
    fn simplifies_a_real_traced_circle_at_a_loose_enough_tolerance() {
        let raw: [[f64; 2]; 11] = [
            [0.6287417446964148, 0.22996531534058842],
            [0.76953125, 0.37054136650753006],
            [0.79296875, 0.49999999999999983],
            [0.76953125, 0.6294586334924698],
            [0.7107729946964143, 0.7114409346594125],
            [0.2892270053035857, 0.7114409346594119],
            [0.23046875, 0.6294586334924699],
            [0.20703125, 0.5005523834924699],
            [0.22996531534058817, 0.37125825530358547],
            [0.3705413665075301, 0.23046875],
            [0.4994476165075301, 0.20703125],
        ];
        let points: Vec<Point> = raw
            .iter()
            .enumerate()
            .map(|(i, &[x, y])| {
                pt(
                    x,
                    y,
                    if i == 2 || i == 7 || i == 10 {
                        PointKind::Symmetric
                    } else {
                        PointKind::Smooth
                    },
                )
            })
            .collect();
        let path = Path {
            subpaths: vec![SubPath {
                points,
                closed: true,
            }],
            wrap: false,
        };
        let out = simplify(&path, 0.05);
        assert!(
            out.subpaths[0].points.len() < 11,
            "0.05 is well above this curve's own point-to-point sagitta (~0.023-0.34), so it \
             should simplify"
        );
    }
}
