// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Path — Step 1 of PATH_NODE_SPEC.md's staged
//! build: the data geometry, CPU-only, no editor UI and no GPU yet.
//!
//! Ported 1:1 from `path_reference.py` (the owner's ground-truth script,
//! itself the authority over PATH_NODE_SPEC.md's own prose per that
//! document's rule 1), using `kurbo` for every piece of Bezier math the
//! spec names explicitly rather than hand-rolling it: [`segments`] builds
//! `kurbo::CubicBez`s directly from [`mc_core::SubPath`]'s control-point
//! formula; [`insert_point`] uses `CubicBez::subsegment` (an exact
//! de Casteljau-equivalent split, not Python's own hand-written
//! `split_segment`); [`nearest_on_path`] uses `ParamCurveNearest::nearest`
//! (Python's own `nearest_on_path` calls itself "a coarse stand-in" for
//! exactly this); [`sections`] uses `ParamCurveArclen::arclen` for true
//! arc length, not Python's dense-sample chord-length approximation.
//! Tangents are the one place this module deliberately matches Python's
//! own *method*, not just its result: `Section.tan` is a finite-difference
//! estimate ([`gradient_1d`]), not the analytic derivative — see that
//! function's own doc comment for why swapping in the (more "correct")
//! analytic derivative actually fails this step's own tolerance right at
//! a section's first/last sample.
//!
//! Internally f64 throughout (`mc_core::Point`'s own `pos`/`h_in`/`h_out`
//! fields already are), matching Python's own float64 and this step's own
//! acceptance bar: sections within `1e-5` tile units, JSON round-trip
//! within `1e-9` (`golden/path_geometry.json`, `golden/demo_path.json`).

use mc_core::{Path, Point, PointKind, SubPath};

fn to_arr(v: kurbo::Vec2) -> [f64; 2] {
    [v.x, v.y]
}

fn vec_len(v: [f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// Segment `i`'s control polygon, exactly PATH_NODE_SPEC.md §2's formula:
/// `P0 = p[i].pos`, `P1 = p[i].pos + p[i].h_out`, `P2 = p[i+1].pos +
/// p[i+1].h_in`, `P3 = p[i+1].pos` — and, for a closed subpath, one more
/// segment running from the last point back to the first.
pub fn segments(sp: &SubPath) -> Vec<kurbo::CubicBez> {
    let pts = &sp.points;
    let n = pts.len();
    if n == 0 {
        return Vec::new();
    }
    let count = if sp.closed { n } else { n - 1 };
    (0..count)
        .map(|i| {
            let a = &pts[i];
            let b = &pts[(i + 1) % n];
            let p0 = kurbo::Point::new(a.pos[0], a.pos[1]);
            let p1 = kurbo::Point::new(a.pos[0] + a.h_out[0], a.pos[1] + a.h_out[1]);
            let p2 = kurbo::Point::new(b.pos[0] + b.h_in[0], b.pos[1] + b.h_in[1]);
            let p3 = kurbo::Point::new(b.pos[0], b.pos[1]);
            kurbo::CubicBez::new(p0, p1, p2, p3)
        })
        .collect()
}

// ---------------------------------------------------------------------
// Handle constraints (PATH_NODE_SPEC.md §3 / path_reference.py's
// `drag_handle`/`set_kind`) — what the editor enforces while dragging.
// ---------------------------------------------------------------------

/// Which of a point's two handles is being dragged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandleSide {
    In,
    Out,
}

/// Move one handle of a point and apply its kind's constraint to the
/// other: `Corner` leaves the other handle untouched, `Smooth` rotates it
/// to stay exactly opposite (keeping its own length), `Symmetric` mirrors
/// it exactly (same length). `break_link` (Alt/Option-drag) converts the
/// point to a corner first, so only the dragged handle moves.
pub fn drag_handle(pt: &Point, which: HandleSide, new: [f64; 2], break_link: bool) -> Point {
    let kind = if break_link {
        PointKind::Corner
    } else {
        pt.kind
    };
    let (mut h_in, mut h_out) = (pt.h_in, pt.h_out);
    let other = match which {
        HandleSide::Out => {
            h_out = new;
            h_in
        }
        HandleSide::In => {
            h_in = new;
            h_out
        }
    };
    let l_new = vec_len(new);
    let o = if kind == PointKind::Smooth && l_new > 1e-12 {
        let l_other = vec_len(other);
        [-new[0] / l_new * l_other, -new[1] / l_new * l_other]
    } else if kind == PointKind::Symmetric {
        [-new[0], -new[1]]
    } else {
        other
    };
    match which {
        HandleSide::Out => h_in = o,
        HandleSide::In => h_out = o,
    }
    Point {
        pos: pt.pos,
        h_in,
        h_out,
        kind,
        width: pt.width,
    }
}

/// Convert a point's kind. To `Corner`: handles unchanged. To `Smooth`:
/// both handles align to the direction `h_out - h_in`, each keeping its
/// own length. To `Symmetric`: same as `Smooth`, then both lengths are set
/// to their average. If both handles are zero, they stay zero (the user
/// then drags them out) — matches `path_reference.py`'s `set_kind` exactly,
/// including that zero-handle short-circuit.
pub fn set_kind(pt: &Point, kind: PointKind) -> Point {
    if kind == PointKind::Corner {
        return Point { kind, ..*pt };
    }
    let (hi, ho) = (pt.h_in, pt.h_out);
    let (mut li, mut lo) = (vec_len(hi), vec_len(ho));
    let d = [ho[0] - hi[0], ho[1] - hi[1]];
    let ld = vec_len(d);
    if ld < 1e-12 {
        return Point { kind, ..*pt };
    }
    let u = [d[0] / ld, d[1] / ld];
    if kind == PointKind::Symmetric {
        let avg = 0.5 * (li + lo);
        li = avg;
        lo = avg;
    }
    Point {
        pos: pt.pos,
        h_in: [-u[0] * li, -u[1] * li],
        h_out: [u[0] * lo, u[1] * lo],
        kind,
        width: pt.width,
    }
}

/// Insert a point on segment `seg_index` at parameter `t` without changing
/// the curve's shape (`CubicBez::subsegment` is an exact split, the same
/// result de Casteljau subdivision would give). The neighbours' adjacent
/// handles shorten to match; the new point is `Smooth` if both its new
/// handles are nonzero (built from a genuinely curved split), else
/// `Corner` (a straight-line segment with zero handles splits into two
/// straight pieces, still zero). Width is interpolated linearly between
/// the two original neighbours.
pub fn insert_point(sp: &SubPath, seg_index: usize, t: f64) -> SubPath {
    use kurbo::ParamCurve;

    let n = sp.points.len();
    let segs = segments(sp);
    let c = segs[seg_index];
    let left = c.subsegment(0.0..t);
    let right = c.subsegment(t..1.0);

    let a_idx = seg_index;
    let b_idx = (seg_index + 1) % n;

    let mut pts = sp.points.clone();
    pts[a_idx].h_out = to_arr(left.p1 - left.p0);
    pts[b_idx].h_in = to_arr(right.p2 - right.p3);

    let h_in = to_arr(left.p2 - left.p3);
    let h_out = to_arr(right.p1 - right.p0);
    let kind = if vec_len(h_in) > 1e-12 && vec_len(h_out) > 1e-12 {
        PointKind::Smooth
    } else {
        PointKind::Corner
    };
    let w = pts[a_idx].width * (1.0 - t as f32) + pts[b_idx].width * t as f32;
    let new_point = Point {
        pos: [left.p3.x, left.p3.y],
        h_in,
        h_out,
        kind,
        width: w,
    };
    pts.insert(seg_index + 1, new_point);
    SubPath::new(pts, sp.closed)
}

/// Remove a point. Neighbours keep their handles — nothing is refitted. A
/// closed subpath with fewer than 3 points left becomes open (a subpath
/// with fewer than 2 points at all is the caller's concern, same as
/// `path_reference.py`'s own doc comment notes).
pub fn delete_point(sp: &SubPath, index: usize) -> SubPath {
    let mut pts = sp.points.clone();
    pts.remove(index);
    let closed = sp.closed && pts.len() >= 3;
    SubPath::new(pts, closed)
}

/// Reverses point order — used by the 2D View's "Join" action to orient
/// two subpaths so the endpoints being joined land adjacent to each other
/// before concatenating. `h_in`/`h_out` swap along with the order, since
/// they're directional (toward the previous/next point) and that
/// direction itself reverses; a plain point-list reversal without this
/// swap would leave every handle pointing the wrong way along the curve.
pub fn reverse_subpath(sp: &SubPath) -> SubPath {
    let points = sp
        .points
        .iter()
        .rev()
        .map(|p| Point {
            pos: p.pos,
            h_in: p.h_out,
            h_out: p.h_in,
            kind: p.kind,
            width: p.width,
        })
        .collect();
    SubPath::new(points, sp.closed)
}

/// `(segment_index, t, distance)` of the closest point on the subpath to
/// `q`, via `kurbo::ParamCurveNearest` — PATH_NODE_SPEC.md §6 explicitly
/// calls out `path_reference.py`'s own `nearest_on_path` as "a coarse
/// stand-in" for this. Used for "click on a segment inserts a point."
pub fn nearest_on_path(sp: &SubPath, q: [f64; 2]) -> (usize, f64, f64) {
    use kurbo::ParamCurveNearest;
    let q_pt = kurbo::Point::new(q[0], q[1]);
    let segs = segments(sp);
    let mut best = (0usize, 0.0f64, f64::INFINITY);
    for (i, seg) in segs.iter().enumerate() {
        let nearest = seg.nearest(q_pt, 1e-9);
        let dist = nearest.distance_sq.max(0.0).sqrt();
        if dist < best.2 {
            best = (i, nearest.t, dist);
        }
    }
    best
}

// ---------------------------------------------------------------------
// What the Stitch node consumes: arc-length sections
// (path_reference.py's `Section`/`sections`)
// ---------------------------------------------------------------------

/// A run of the path between two breaks (sharp corners or open-path ends),
/// tabulated densely against arc length `s` so a stitch rule can ask
/// "where is the path at distance `s` along this section?" via [`at`].
pub struct Section {
    /// Cumulative arc length, tile units, `s[0] = 0`.
    pub s: Vec<f64>,
    pub p: Vec<[f64; 2]>,
    pub tan: Vec<[f64; 2]>,
    pub width: Vec<f32>,
    /// Unit bisector normal at a corner start, for satin mitres — `None`
    /// if the section's own start isn't a break (an open path's first
    /// point, or a point whose tangent didn't actually turn sharply).
    pub start_corner: Option<[f64; 2]>,
    pub end_corner: Option<[f64; 2]>,
    /// A whole closed subpath with no breaks at all (one closed section,
    /// e.g. the demo's circle) rather than a run between two breaks.
    pub closed: bool,
}

impl Section {
    pub fn length(&self) -> f64 {
        self.s.last().copied().unwrap_or(0.0)
    }

    /// Position, unit tangent and width at arc length `s_query` along this
    /// section (clamped to `[0, length]`), linearly interpolated between
    /// the nearest table samples — `path_reference.py`'s own `Section.at`.
    pub fn at(&self, s_query: f64) -> ([f64; 2], [f64; 2], f32) {
        let n = self.s.len();
        if n == 0 {
            return ([0.0, 0.0], [1.0, 0.0], 1.0);
        }
        if n == 1 {
            return (self.p[0], self.tan[0], self.width[0]);
        }
        let s_query = s_query.clamp(0.0, self.length());
        let i = match self
            .s
            .binary_search_by(|v| v.partial_cmp(&s_query).unwrap())
        {
            Ok(i) => i.min(n - 2),
            Err(i) => i.saturating_sub(1).min(n - 2),
        };
        let (s0, s1) = (self.s[i], self.s[i + 1]);
        let t = if (s1 - s0).abs() > 1e-15 {
            (s_query - s0) / (s1 - s0)
        } else {
            0.0
        };
        let lerp2 = |a: [f64; 2], b: [f64; 2]| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        let p = lerp2(self.p[i], self.p[i + 1]);
        let raw_tan = lerp2(self.tan[i], self.tan[i + 1]);
        let norm = vec_len(raw_tan).max(1e-15);
        let tan = [raw_tan[0] / norm, raw_tan[1] / norm];
        let w = self.width[i] as f64 + (self.width[i + 1] as f64 - self.width[i] as f64) * t;
        (p, tan, w as f32)
    }
}

fn turn_angle_deg(din: kurbo::Vec2, dout: kurbo::Vec2) -> f64 {
    let a = din.y.atan2(din.x);
    let b = dout.y.atan2(dout.x);
    let d = (b - a + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
        - std::f64::consts::PI;
    d.abs().to_degrees()
}

/// Unit tangent direction at a segment's start/end, falling back to the
/// chord between the relevant control points for a zero handle (and to the
/// segment's own chord if that's also zero) — `path_reference.py`'s own
/// `seg_dir` closure inside `sections`.
fn seg_dir(c: &kurbo::CubicBez, end: bool) -> kurbo::Vec2 {
    use kurbo::{ParamCurve, ParamCurveDeriv};
    let t = if end { 1.0 } else { 0.0 };
    let d = c.deriv().eval(t).to_vec2();
    let d = if d.hypot() < 1e-12 {
        let d2 = if end { c.p3 - c.p2 } else { c.p1 - c.p0 };
        if d2.hypot() < 1e-12 {
            c.p3 - c.p0
        } else {
            d2
        }
    } else {
        d
    };
    d / (d.hypot() + 1e-15)
}

/// `n + 1` uniform-`t` samples of one segment: arc length *from this
/// segment's own start* (via `kurbo::ParamCurveArclen`, not a chord-length
/// approximation) and position. Tangents are deliberately *not* computed
/// here — `path_reference.py`'s own `Section.tan` isn't the analytic
/// derivative, it's a finite-difference estimate ([`gradient_1d`])
/// computed once over each *whole section's* concatenated `(s, p)` table,
/// which measurably differs from the analytic derivative right at a
/// section's own first/last sample (a one-sided estimate there, not
/// centered) — close enough everywhere else, but outside this step's
/// `1e-5` tolerance at the edges if skipped. Matched here, not "improved
/// on," per rule 1: the code is ground truth.
fn sample_segment(c: &kurbo::CubicBez, n: usize) -> (Vec<f64>, Vec<kurbo::Point>) {
    use kurbo::{ParamCurve, ParamCurveArclen};
    let mut s = vec![0.0f64; n + 1];
    let mut p = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t = i as f64 / n as f64;
        p.push(c.eval(t));
        if i > 0 {
            let t0 = (i - 1) as f64 / n as f64;
            s[i] = s[i - 1] + c.subsegment(t0..t).arclen(1e-7);
        }
    }
    (s, p)
}

/// `numpy.gradient(p, s, axis=0, edge_order=1)`, ported exactly: a
/// one-sided difference at each end, and the standard second-order-accurate
/// non-uniform-spacing central difference at every interior sample. Not
/// normalized — `sections()` normalizes afterward, same order of
/// operations as the Python reference.
fn gradient_1d(p: &[[f64; 2]], s: &[f64]) -> Vec<[f64; 2]> {
    let n = p.len();
    if n < 2 {
        return vec![[0.0, 0.0]; n];
    }
    let mut out = vec![[0.0, 0.0]; n];
    let ds0 = s[1] - s[0];
    out[0] = [(p[1][0] - p[0][0]) / ds0, (p[1][1] - p[0][1]) / ds0];
    let dsn = s[n - 1] - s[n - 2];
    out[n - 1] = [
        (p[n - 1][0] - p[n - 2][0]) / dsn,
        (p[n - 1][1] - p[n - 2][1]) / dsn,
    ];
    for i in 1..n - 1 {
        let dx1 = s[i] - s[i - 1];
        let dx2 = s[i + 1] - s[i];
        let a = -dx2 / (dx1 * (dx1 + dx2));
        let b = (dx2 - dx1) / (dx1 * dx2);
        let c = dx1 / (dx2 * (dx1 + dx2));
        out[i] = [
            a * p[i - 1][0] + b * p[i][0] + c * p[i + 1][0],
            a * p[i - 1][1] + b * p[i][1] + c * p[i + 1][1],
        ];
    }
    out
}

const SAMPLES_PER_SEGMENT: usize = 400;

/// Split every subpath into [`Section`]s at breaks (PATH_NODE_SPEC.md §7):
/// a break is a `Corner`-kind point where the tangent turns by more than
/// `corner_threshold_deg` (a `Corner` point whose handles happen to line
/// up isn't a break; `Smooth`/`Symmetric` points never break). Open path
/// ends are always section ends. A closed subpath with no breaks at all
/// becomes a single closed section.
pub fn sections(path: &Path, corner_threshold_deg: f64) -> Vec<Section> {
    let mut out = Vec::new();
    for sp in &path.subpaths {
        let segs = segments(sp);
        if segs.is_empty() {
            continue;
        }
        let n_pts = sp.points.len();

        let mut breaks: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        let mut bisect: std::collections::HashMap<usize, kurbo::Vec2> =
            std::collections::HashMap::new();
        for k in 0..n_pts {
            if !sp.closed && (k == 0 || k == n_pts - 1) {
                continue;
            }
            let prev_seg = &segs[(k + segs.len() - 1) % segs.len()];
            let next_seg = &segs[k % segs.len()];
            let din = seg_dir(prev_seg, true);
            let dout = seg_dir(next_seg, false);
            if sp.points[k].kind == PointKind::Corner
                && turn_angle_deg(din, dout) > corner_threshold_deg
            {
                breaks.insert(k);
                let mut b = din - dout;
                if b.hypot() < 1e-9 {
                    b = kurbo::Vec2::new(-din.y, din.x);
                }
                bisect.insert(k, b / b.hypot());
            }
        }

        let order: Vec<usize> = if sp.closed && !breaks.is_empty() {
            let first = *breaks.iter().next().unwrap();
            (0..segs.len()).map(|i| (first + i) % segs.len()).collect()
        } else {
            (0..segs.len()).collect()
        };

        fn finish_section(
            s: Vec<f64>,
            p: Vec<[f64; 2]>,
            width: Vec<f32>,
            start_corner: Option<[f64; 2]>,
            end_corner: Option<[f64; 2]>,
            closed: bool,
        ) -> Section {
            let raw_tan = gradient_1d(&p, &s);
            let tan: Vec<[f64; 2]> = raw_tan
                .iter()
                .map(|t| {
                    let n = vec_len(*t) + 1e-15;
                    [t[0] / n, t[1] / n]
                })
                .collect();
            Section {
                s,
                p,
                tan,
                width,
                start_corner,
                end_corner,
                closed,
            }
        }

        let mut cur_s: Vec<f64> = Vec::new();
        let mut cur_p: Vec<[f64; 2]> = Vec::new();
        let mut cur_width: Vec<f32> = Vec::new();
        let mut run_offset = 0.0f64;
        let mut start_pt = order[0];

        for &idx in &order {
            let c = segs[idx];
            let a = &sp.points[idx];
            let b = &sp.points[(idx + 1) % n_pts];
            let (local_s, pos) = sample_segment(&c, SAMPLES_PER_SEGMENT);
            let skip_first = !cur_s.is_empty();
            let start_i = usize::from(skip_first);
            for i in start_i..=SAMPLES_PER_SEGMENT {
                let t = i as f64 / SAMPLES_PER_SEGMENT as f64;
                let w = a.width + (b.width - a.width) * t as f32;
                cur_s.push(run_offset + local_s[i]);
                cur_p.push([pos[i].x, pos[i].y]);
                cur_width.push(w);
            }
            run_offset += local_s[SAMPLES_PER_SEGMENT];

            let end_pt = (idx + 1) % n_pts;
            let is_break_end = breaks.contains(&end_pt);
            let is_open_path_end = !sp.closed && end_pt == n_pts - 1;
            if is_break_end || is_open_path_end {
                out.push(finish_section(
                    std::mem::take(&mut cur_s),
                    std::mem::take(&mut cur_p),
                    std::mem::take(&mut cur_width),
                    bisect.get(&start_pt).map(|v| to_arr(*v)),
                    if is_break_end {
                        bisect.get(&end_pt).map(|v| to_arr(*v))
                    } else {
                        None
                    },
                    false,
                ));
                run_offset = 0.0;
                start_pt = end_pt;
            }
        }
        if !cur_s.is_empty() {
            let end_pt_for_bisect = if sp.closed { start_pt } else { n_pts - 1 };
            out.push(finish_section(
                cur_s,
                cur_p,
                cur_width,
                bisect.get(&start_pt).map(|v| to_arr(*v)),
                bisect.get(&end_pt_for_bisect).map(|v| to_arr(*v)),
                sp.closed && breaks.is_empty(),
            ));
        }
    }
    out
}

// ---------------------------------------------------------------------
// Step 2: Preview Mask — a thin 2px-wide raster of the whole path
// (PATH_NODE_SPEC.md §1), reusing Knit's tube_raster rather than a
// second rasterizer.
// ---------------------------------------------------------------------

/// Half the preview line's own width, in pixels (PATH_NODE_SPEC.md §1:
/// "the curve drawn 2 px wide").
const PREVIEW_HALF_WIDTH_PX: f32 = 1.0;

/// Dense position-only samples of one subpath, directly from its own
/// `CubicBez` segments — deliberately *not* [`sections`]'s arc-length
/// table: Preview Mask needs no corner breaks, mitre bisectors, or exact
/// arc length, just enough points for a smooth-looking line, and it
/// re-renders on every frame of a drag gesture (an editor concern Step 1
/// didn't have), so it stays far cheaper than `sections`'s exact-arclen
/// tabulation at 400 samples/segment.
///
/// `pub` (not just used by [`dispatch_preview_mask`] below) because
/// `mc_app`'s own 2D View editor overlay draws this same outline on top
/// of the rasterized mask — reusing it keeps the curve-sampling math in
/// one place rather than mc_app hand-rolling a second copy via its own
/// `kurbo` dependency.
pub fn preview_polyline(sp: &SubPath, samples_per_segment: usize) -> Vec<[f64; 2]> {
    use kurbo::ParamCurve;
    let segs = segments(sp);
    let mut out = Vec::new();
    for (i, c) in segs.iter().enumerate() {
        let start = usize::from(i != 0); // skip the duplicate joint sample
        for j in start..=samples_per_segment {
            let t = j as f64 / samples_per_segment as f64;
            let p = c.eval(t);
            out.push([p.x, p.y]);
        }
    }
    out
}

/// What a Path node should actually be treated as containing: its own
/// `"path"` param if it has any points, otherwise whatever's connected to
/// its Reference input — same "optional input with a built-in fallback"
/// pattern every other node's own optional input already follows (Tile
/// Generator's Shape, Fabric Generator's Fiber), just for Path-type data
/// read directly off the upstream node's own params instead of a texture
/// (see [`mc_core::PortType::Path`]'s own doc comment on why). Read-only:
/// the moment the artist edits anything, the editor writes a full copy
/// into this node's own `"path"` param (see `mc_app::path_editor`), so
/// editing never mutates the upstream node and a later edit upstream
/// stops affecting this node once that happens.
pub fn resolve_effective_path(graph: &mc_core::Graph, node_id: mc_core::NodeId) -> Path {
    let own = graph
        .node(node_id)
        .and_then(|n| n.params.get("path"))
        .and_then(mc_core::ParamValue::as_path)
        .cloned()
        .unwrap_or_default();
    if !own.subpaths.is_empty() {
        return own;
    }
    graph
        .upstream_from(node_id, "reference")
        .and_then(|(upstream_id, _)| graph.node(upstream_id))
        .and_then(|n| n.params.get("path"))
        .and_then(mc_core::ParamValue::as_path)
        .cloned()
        .unwrap_or(own)
}

// ---------------------------------------------------------------------
// Stamp distribution: repeat whatever's connected to the Stamp input
// along this node's own curve, oriented to its tangent — the owner's own
// "second input... to translate as a splatter on Path" design, with Path
// itself (not a separate Stitch/Shape node) owning the actual placement
// math. First version: a CPU instance table (position + rotation per
// copy) and a gather-style compute shader that tests every instance per
// output pixel — correct and simple, not yet optimized with the kind of
// spatial binning `tube_raster` uses, so a very small Stamp Spacing on a
// long curve at high resolution will be slow (see `MAX_INSTANCES`).
// ---------------------------------------------------------------------

/// One placed copy of the Stamp input: where, and rotated by how much
/// (radians, already including both the curve's own tangent angle *and*
/// the Stamp Rotation param — the shader itself doesn't know about
/// "align to curve" at all, it just rotates by whatever angle it's given).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StampInstance {
    pub pos: [f32; 2],
    pub angle: f32,
}

/// A spacing/path combination that would place more than this many
/// instances is silently capped rather than left to produce an
/// unusably slow (or, at the true pathological extreme, VRAM-exhausting)
/// dispatch — generous enough for any reasonable Stamp Spacing on a
/// hand-drawn curve.
const MAX_STAMP_INSTANCES: usize = 4000;

/// Places instances every `spacing` tile units along `path`'s own arc
/// length, reusing [`sections`]'s exact tabulation (the same one a future
/// Stitch node will walk) rather than a second, separately-maintained
/// curve sampler. `extra_rotation_rad` is added to every instance's own
/// angle; passing `align_to_curve: false` skips the tangent term entirely
/// (every instance gets just `extra_rotation_rad`, upright-with-a-twist
/// rather than curve-following).
pub fn stamp_instances(
    path: &Path,
    spacing: f64,
    align_to_curve: bool,
    extra_rotation_rad: f32,
) -> Vec<StampInstance> {
    const MIN_SPACING: f64 = 1e-4;
    let spacing = spacing.max(MIN_SPACING);
    let mut out = Vec::new();
    'sections: for section in sections(path, 30.0) {
        let length = section.length();
        let mut s = 0.0;
        loop {
            let (p, tan, _width) = section.at(s);
            let angle = if align_to_curve {
                tan[1].atan2(tan[0]) as f32 + extra_rotation_rad
            } else {
                extra_rotation_rad
            };
            out.push(StampInstance {
                pos: [p[0] as f32, p[1] as f32],
                angle,
            });
            if out.len() >= MAX_STAMP_INSTANCES {
                break 'sections;
            }
            if s >= length {
                break;
            }
            s = (s + spacing).min(length);
        }
    }
    out
}

/// Live-read params for the Stamp distribution controls and Show Curve —
/// same `XxxParams::from_graph` pattern every other node's own params
/// struct already follows.
pub struct PathParams {
    pub show_curve: bool,
    pub stamp_spacing: f32,
    pub stamp_size: f32,
    pub stamp_rotation_deg: f32,
    pub stamp_align_to_curve: bool,
}

impl Default for PathParams {
    fn default() -> Self {
        Self {
            show_curve: true,
            stamp_spacing: 0.1,
            stamp_size: 0.08,
            stamp_rotation_deg: 0.0,
            stamp_align_to_curve: true,
        }
    }
}

impl PathParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            show_curve: params
                .get("show_curve")
                .and_then(mc_core::ParamValue::as_bool)
                .unwrap_or(defaults.show_curve),
            stamp_spacing: params
                .get("stamp_spacing")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.stamp_spacing),
            stamp_size: params
                .get("stamp_size")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.stamp_size),
            stamp_rotation_deg: params
                .get("stamp_rotation")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.stamp_rotation_deg),
            stamp_align_to_curve: params
                .get("stamp_align_to_curve")
                .and_then(mc_core::ParamValue::as_bool)
                .unwrap_or(defaults.stamp_align_to_curve),
        }
    }
}

fn pack_stamp_instances(instances: &[StampInstance]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(instances.len() * 16);
    for inst in instances {
        bytes.extend_from_slice(&inst.pos[0].to_le_bytes());
        bytes.extend_from_slice(&inst.pos[1].to_le_bytes());
        bytes.extend_from_slice(&inst.angle.to_le_bytes());
        bytes.extend_from_slice(&0f32.to_le_bytes()); // pad to 16 bytes/instance
    }
    bytes
}

fn pack_stamp_params(
    resolution: u32,
    instance_count: u32,
    stamp_size: f32,
    wrap: bool,
) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&instance_count.to_le_bytes());
    bytes[8..12].copy_from_slice(&stamp_size.max(1e-4).to_le_bytes());
    bytes[12..16].copy_from_slice(&(wrap as u32).to_le_bytes());
    bytes
}

/// Composites `instances` worth of `stamp_key`'s own texture on top of
/// `base_key` (already-rendered content — the outline, or blank if Show
/// Curve is off) into `output_key`, via `max` per pixel (consistent with
/// every other "strands/shapes occupy space, the most prominent one
/// wins" composite in this codebase — `tube_raster`'s own height test,
/// Knit's ply occlusion). `base_key` and `output_key` must be different
/// textures — the shader reads one while writing the other.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_stamp(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    stamp_key: &str,
    base_key: &str,
    output_key: &str,
    resolution: u32,
    instances: &[StampInstance],
    stamp_size: f32,
    wrap: bool,
) {
    let shader_source = crate::shaders::full_source("path_stamp.wgsl");
    let pipeline = gpu.compute_pipeline("path_stamp", &shader_source, "main");

    let param_bytes = pack_stamp_params(resolution, instances.len() as u32, stamp_size, wrap);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("path_stamp_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let instance_bytes = pack_stamp_instances(instances);
    let instance_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("path_stamp_instances"),
        // wgpu rejects a zero-sized buffer outright — pad to at least 16
        // bytes (one unread instance) for the no-instances case; the
        // shader's own loop bound (`instance_count` in the uniform) is 0
        // then, so it's never actually read.
        size: (instance_bytes.len() as u64).max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !instance_bytes.is_empty() {
        gpu.queue.write_buffer(&instance_buffer, 0, &instance_bytes);
    }

    let stamp_view = pool
        .view(stamp_key)
        .unwrap_or_else(|| panic!("path::dispatch_stamp: no texture at stamp key {stamp_key:?}"))
        .clone();
    let base_view = pool
        .view(base_key)
        .unwrap_or_else(|| panic!("path::dispatch_stamp: no texture at base key {base_key:?}"))
        .clone();

    let output_view = pool.get_or_create(
        &gpu.device,
        output_key,
        resolution,
        wgpu::TextureFormat::Rgba8Unorm,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("path_stamp_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: instance_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&stamp_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&base_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("path_stamp_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("path_stamp_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

/// Renders `path` as a thin outline into `keys`' Preview Mask texture —
/// every subpath's own polyline, as constant-radius capsules
/// ([`crate::tube_raster`]), with wrap-around handled the same way Knit's
/// own closing segments are (raw, possibly out-of-`[0,1)` coordinates;
/// `tube_raster`'s minimum-image test does the rest). `path.wrap == false`
/// isn't specially handled yet — Preview Mask always renders as if
/// wrapping, matching `tube_raster`'s own native behavior; a true
/// wrap-off clip is Stitch's own concern once its GPU raster step adds it
/// (STITCH_NODE_SPEC.md §7: "Wrap off: clip at the tile edge").
pub fn dispatch_preview_mask(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    path: &Path,
) {
    const SAMPLES_PER_SEGMENT: usize = 32;
    let radius = PREVIEW_HALF_WIDTH_PX / resolution as f32;

    let mut segs = Vec::new();
    for (tag, sp) in path.subpaths.iter().enumerate() {
        let poly = preview_polyline(sp, SAMPLES_PER_SEGMENT);
        let n = poly.len();
        if n < 2 {
            continue;
        }
        for i in 0..n - 1 {
            segs.push(crate::tube_raster::Segment {
                p0: [poly[i][0] as f32, poly[i][1] as f32, 0.0],
                r0: radius,
                p1: [poly[i + 1][0] as f32, poly[i + 1][1] as f32, 0.0],
                r1: radius,
                tag: tag as u32,
            });
        }
        if sp.closed {
            segs.push(crate::tube_raster::Segment {
                p0: [poly[n - 1][0] as f32, poly[n - 1][1] as f32, 0.0],
                r0: radius,
                p1: [poly[0][0] as f32, poly[0][1] as f32, 0.0],
                r1: radius,
                tag: tag as u32,
            });
        }
    }

    if segs.is_empty() {
        // A fresh Path node (nothing drawn yet) has nothing to raster —
        // tube_raster requires at least one segment, so fill solid black
        // directly rather than dispatching a degenerate pass.
        gpu.fill_rgba8(pool, key, resolution, [0, 0, 0, 255]);
        return;
    }

    // tube_raster's own height/ply_id outputs aren't meaningful for a flat
    // preview outline — route them to scratch keys this function doesn't
    // expose, and reuse its `mask` output directly as Preview Mask.
    let height_key = format!("{key}__preview_height_scratch");
    let ply_id_key = format!("{key}__preview_ply_id_scratch");
    let cell_key = format!("{key}__preview_cell_scratch");
    crate::tube_raster::dispatch(
        gpu,
        pool,
        crate::tube_raster::RasterOutputKeys {
            height: &height_key,
            mask: key,
            ply_id: &ply_id_key,
            cell: &cell_key,
        },
        resolution,
        &segs,
        1,
        radius,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEMO_PATH_JSON: &str = include_str!("../../../tests/golden/path/demo_path.json");
    const PATH_GEOMETRY_JSON: &str = include_str!("../../../tests/golden/path/path_geometry.json");

    fn demo_path() -> Path {
        serde_json::from_str(DEMO_PATH_JSON).expect("golden demo_path.json should parse")
    }

    fn arr(v: &serde_json::Value) -> [f64; 2] {
        [v[0].as_f64().unwrap(), v[1].as_f64().unwrap()]
    }

    fn assert_close2(actual: [f64; 2], expected: [f64; 2], tol: f64, context: &str) {
        for i in 0..2 {
            assert!(
                (actual[i] - expected[i]).abs() <= tol,
                "{context}[{i}]: actual={} expected={}",
                actual[i],
                expected[i]
            );
        }
    }

    // --- JSON round-trip ---

    #[test]
    fn demo_path_round_trips_through_json_within_1e9() {
        let path = demo_path();
        let reserialized = serde_json::to_value(&path).unwrap();
        let original: serde_json::Value = serde_json::from_str(DEMO_PATH_JSON).unwrap();

        fn assert_value_close(a: &serde_json::Value, b: &serde_json::Value, path: &str) {
            match (a, b) {
                (serde_json::Value::Number(x), serde_json::Value::Number(y)) => {
                    let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
                    // `width` is f32 in mc_core::Point (PATH_NODE_SPEC.md §2's
                    // own stated type), unlike pos/h_in/h_out which are f64 —
                    // f32 round-trips at ~1e-7 relative precision, not 1e-9.
                    let tol = if path.ends_with(".width") { 1e-6 } else { 1e-9 };
                    assert!((x - y).abs() < tol, "{path}: {x} vs {y}");
                }
                (serde_json::Value::Array(xs), serde_json::Value::Array(ys)) => {
                    assert_eq!(xs.len(), ys.len(), "{path}: array length");
                    for (i, (x, y)) in xs.iter().zip(ys).enumerate() {
                        assert_value_close(x, y, &format!("{path}[{i}]"));
                    }
                }
                (serde_json::Value::Object(xs), serde_json::Value::Object(ys)) => {
                    for key in xs.keys() {
                        assert_value_close(
                            &xs[key],
                            ys.get(key)
                                .unwrap_or_else(|| panic!("{path}.{key} missing")),
                            &format!("{path}.{key}"),
                        );
                    }
                }
                _ => assert_eq!(a, b, "{path}"),
            }
        }
        assert_value_close(&reserialized, &original, "$");
    }

    // --- sections() ---

    #[test]
    fn sections_match_golden_geometry() {
        let path = demo_path();
        let secs = sections(&path, 30.0);
        let golden: serde_json::Value = serde_json::from_str(PATH_GEOMETRY_JSON).unwrap();
        let golden_secs = golden["sections"].as_array().unwrap();

        assert_eq!(secs.len(), golden_secs.len(), "section count");
        for (i, (got, want)) in secs.iter().zip(golden_secs).enumerate() {
            let ctx = format!("section {i}");
            assert!(
                (got.length() - want["length"].as_f64().unwrap()).abs() < 1e-5,
                "{ctx}: length {} vs {}",
                got.length(),
                want["length"]
            );
            assert_eq!(
                got.start_corner.is_some(),
                want["start_corner"].as_bool().unwrap(),
                "{ctx}: start_corner"
            );
            assert_eq!(
                got.end_corner.is_some(),
                want["end_corner"].as_bool().unwrap(),
                "{ctx}: end_corner"
            );
            assert_eq!(
                got.closed,
                want["closed"].as_bool().unwrap(),
                "{ctx}: closed"
            );

            let samples = want["samples"].as_array().unwrap();
            assert_eq!(samples.len(), 11, "{ctx}: golden sample count");
            let length = got.length();
            for (j, sample) in samples.iter().enumerate() {
                let s_query = j as f64 / 10.0 * length;
                let (p, tan, _w) = got.at(s_query);
                assert_close2(p, arr(&sample["p"]), 1e-5, &format!("{ctx} sample {j} p"));
                assert_close2(
                    tan,
                    arr(&sample["tangent"]),
                    1e-5,
                    &format!("{ctx} sample {j} tangent"),
                );
            }
        }
    }

    // --- insert_point ---

    #[test]
    fn insert_point_matches_golden() {
        let path = demo_path();
        let golden: serde_json::Value = serde_json::from_str(PATH_GEOMETRY_JSON).unwrap();
        let case = &golden["insert_point"];
        let subpath_idx = case["subpath"].as_u64().unwrap() as usize;
        let seg_index = case["segment"].as_u64().unwrap() as usize;
        let t = case["t"].as_f64().unwrap();

        let result = insert_point(&path.subpaths[subpath_idx], seg_index, t);
        let want = case["result"].as_array().unwrap();
        assert_eq!(result.points.len(), want.len());
        for (i, (got, want)) in result.points.iter().zip(want).enumerate() {
            let ctx = format!("point {i}");
            assert_close2(got.pos, arr(&want["pos"]), 1e-6, &format!("{ctx} pos"));
            assert_close2(got.h_in, arr(&want["h_in"]), 1e-6, &format!("{ctx} h_in"));
            assert_close2(
                got.h_out,
                arr(&want["h_out"]),
                1e-6,
                &format!("{ctx} h_out"),
            );
            assert_eq!(
                serde_json::to_value(got.kind).unwrap(),
                want["kind"],
                "{ctx} kind"
            );
            assert!(
                (got.width - want["width"].as_f64().unwrap() as f32).abs() < 1e-5,
                "{ctx} width"
            );
        }
    }

    // --- drag_handle ---

    fn point_from_json(v: &serde_json::Value) -> Point {
        Point {
            pos: arr(&v["pos"]),
            h_in: arr(&v["h_in"]),
            h_out: arr(&v["h_out"]),
            kind: match v["kind"].as_str().unwrap() {
                "corner" => PointKind::Corner,
                "smooth" => PointKind::Smooth,
                "symmetric" => PointKind::Symmetric,
                other => panic!("unknown kind {other:?}"),
            },
            width: v["width"].as_f64().unwrap() as f32,
        }
    }

    fn assert_point_matches(got: &Point, want: &serde_json::Value, ctx: &str) {
        assert_close2(got.pos, arr(&want["pos"]), 1e-9, &format!("{ctx} pos"));
        assert_close2(got.h_in, arr(&want["h_in"]), 1e-9, &format!("{ctx} h_in"));
        assert_close2(
            got.h_out,
            arr(&want["h_out"]),
            1e-9,
            &format!("{ctx} h_out"),
        );
        assert_eq!(
            serde_json::to_value(got.kind).unwrap(),
            want["kind"],
            "{ctx} kind"
        );
    }

    #[test]
    fn drag_handle_matches_golden() {
        let golden: serde_json::Value = serde_json::from_str(PATH_GEOMETRY_JSON).unwrap();
        let case = &golden["drag_handle"];
        let pt = point_from_json(&case["point"]);
        let which = match case["which"].as_str().unwrap() {
            "out" => HandleSide::Out,
            "in" => HandleSide::In,
            other => panic!("unknown which {other:?}"),
        };
        let new = arr(&case["new"]);

        let smooth_pt = Point {
            kind: PointKind::Smooth,
            ..pt
        };
        assert_point_matches(
            &drag_handle(&smooth_pt, which, new, false),
            &case["smooth"],
            "smooth",
        );

        let symmetric_pt = Point {
            kind: PointKind::Symmetric,
            ..pt
        };
        assert_point_matches(
            &drag_handle(&symmetric_pt, which, new, false),
            &case["symmetric"],
            "symmetric",
        );

        assert_point_matches(
            &drag_handle(&pt, which, new, true),
            &case["corner_via_alt"],
            "corner_via_alt",
        );
    }

    // --- set_kind ---

    #[test]
    fn set_kind_matches_golden() {
        let golden: serde_json::Value = serde_json::from_str(PATH_GEOMETRY_JSON).unwrap();
        let case = &golden["set_kind"];
        let pt = point_from_json(&case["point"]);

        assert_point_matches(&set_kind(&pt, PointKind::Smooth), &case["smooth"], "smooth");
        assert_point_matches(
            &set_kind(&pt, PointKind::Symmetric),
            &case["symmetric"],
            "symmetric",
        );
    }

    #[test]
    fn set_kind_to_corner_leaves_handles_unchanged() {
        let pt = Point {
            pos: [0.0, 0.0],
            h_in: [-0.1, 0.05],
            h_out: [0.2, -0.03],
            kind: PointKind::Smooth,
            width: 1.0,
        };
        let corner = set_kind(&pt, PointKind::Corner);
        assert_eq!(corner.h_in, pt.h_in);
        assert_eq!(corner.h_out, pt.h_out);
        assert_eq!(corner.kind, PointKind::Corner);
    }

    #[test]
    fn set_kind_leaves_zero_handles_zero() {
        let pt = Point::new([0.3, 0.3]);
        let smooth = set_kind(&pt, PointKind::Smooth);
        assert_eq!(smooth.h_in, [0.0, 0.0]);
        assert_eq!(smooth.h_out, [0.0, 0.0]);
    }

    // --- delete_point / nearest_on_path: not in the golden file, sanity only ---

    #[test]
    fn delete_point_removes_exactly_one_point_and_keeps_others_handles() {
        let path = demo_path();
        let sp = &path.subpaths[0];
        let before = sp.points.len();
        let deleted = delete_point(sp, 1);
        assert_eq!(deleted.points.len(), before - 1);
        assert_eq!(deleted.points[0], sp.points[0]);
        assert_eq!(deleted.points[1], sp.points[2]);
    }

    #[test]
    fn resolve_effective_path_prefers_its_own_data_when_non_empty() {
        let mut graph = mc_core::Graph::new();
        let own = mc_core::Node::new("path");
        let id = graph.add_node(own);
        let own_path = Path {
            subpaths: vec![SubPath::new(vec![Point::new([0.2, 0.2])], false)],
            wrap: true,
        };
        graph
            .set_param(id, "path", mc_core::ParamValue::Path(own_path.clone()))
            .unwrap();

        assert_eq!(resolve_effective_path(&graph, id), own_path);
    }

    #[test]
    fn resolve_effective_path_falls_back_to_a_connected_reference_when_empty() {
        let mut graph = mc_core::Graph::new();
        let reference = graph.add_node(mc_core::Node::new("path"));
        let reference_path = Path {
            subpaths: vec![SubPath::new(vec![Point::new([0.4, 0.4])], false)],
            wrap: true,
        };
        graph
            .set_param(
                reference,
                "path",
                mc_core::ParamValue::Path(reference_path.clone()),
            )
            .unwrap();

        let id = graph.add_node(mc_core::Node::new("path"));
        graph
            .set_param(id, "path", mc_core::ParamValue::Path(Path::default()))
            .unwrap();
        graph.connect(reference, "path", id, "reference").unwrap();

        assert_eq!(resolve_effective_path(&graph, id), reference_path);
    }

    #[test]
    fn resolve_effective_path_ignores_an_unconnected_reference_input() {
        let mut graph = mc_core::Graph::new();
        let id = graph.add_node(mc_core::Node::new("path"));
        assert_eq!(resolve_effective_path(&graph, id), Path::default());
    }

    #[test]
    fn reverse_subpath_swaps_order_and_handle_direction() {
        let sp = SubPath::new(
            vec![
                Point {
                    pos: [0.0, 0.0],
                    h_in: [0.0, 0.0],
                    h_out: [0.1, 0.0],
                    kind: PointKind::Smooth,
                    width: 1.0,
                },
                Point {
                    pos: [1.0, 0.0],
                    h_in: [-0.1, 0.0],
                    h_out: [0.1, 0.1],
                    kind: PointKind::Smooth,
                    width: 1.0,
                },
            ],
            false,
        );
        let rev = reverse_subpath(&sp);
        assert_eq!(rev.points.len(), 2);
        // Order reversed...
        assert_eq!(rev.points[0].pos, sp.points[1].pos);
        assert_eq!(rev.points[1].pos, sp.points[0].pos);
        // ...and each point's own h_in/h_out swapped.
        assert_eq!(rev.points[0].h_in, sp.points[1].h_out);
        assert_eq!(rev.points[0].h_out, sp.points[1].h_in);
        assert_eq!(rev.points[1].h_in, sp.points[0].h_out);
        assert_eq!(rev.points[1].h_out, sp.points[0].h_in);
    }

    #[test]
    fn delete_point_opens_a_closed_subpath_once_it_drops_below_3_points() {
        let sp = SubPath::new(
            vec![
                Point::new([0.0, 0.0]),
                Point::new([1.0, 0.0]),
                Point::new([0.5, 1.0]),
            ],
            true,
        );
        let after = delete_point(&sp, 0);
        assert_eq!(after.points.len(), 2);
        assert!(!after.closed, "a 2-point subpath can't stay closed");
    }

    #[test]
    fn nearest_on_path_finds_a_point_actually_on_the_curve() {
        let path = demo_path();
        let sp = &path.subpaths[0];
        let segs = segments(sp);
        // A point we know is exactly ON segment 0 at t=0.5.
        let on_curve = {
            use kurbo::ParamCurve;
            segs[0].eval(0.5)
        };
        let (seg_index, t, dist) = nearest_on_path(sp, [on_curve.x, on_curve.y]);
        assert_eq!(seg_index, 0);
        assert!((t - 0.5).abs() < 1e-6, "t={t}");
        assert!(dist < 1e-9, "dist={dist}");
    }

    #[test]
    fn segments_counts_one_more_for_a_closed_subpath() {
        let open = SubPath::new(vec![Point::new([0.0, 0.0]), Point::new([1.0, 0.0])], false);
        let closed = SubPath::new(open.points.clone(), true);
        assert_eq!(segments(&open).len(), 1);
        assert_eq!(segments(&closed).len(), 2);
    }

    // --- Step 2: Preview Mask ---

    #[test]
    fn preview_mask_is_blank_for_an_empty_path() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch_preview_mask(&mut gpu, &mut pool, "preview", 64, &Path::default());
        let pixels = gpu.read_rgba8(pool.texture("preview").unwrap(), 64);
        assert!(pixels.iter().step_by(4).all(|&v| v == 0));
    }

    #[test]
    fn preview_mask_draws_something_for_the_demo_path() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch_preview_mask(&mut gpu, &mut pool, "preview", 256, &demo_path());
        let pixels = gpu.read_rgba8(pool.texture("preview").unwrap(), 256);
        let lit = pixels.iter().step_by(4).filter(|&&v| v > 0).count();
        assert!(lit > 0, "demo path should draw a visible outline");
        // A thin 2px line shouldn't cover a large fraction of the tile.
        let total = pixels.len() / 4;
        assert!(
            (lit as f32) < 0.25 * total as f32,
            "outline coverage looks too thick: {lit}/{total}"
        );
    }

    #[test]
    fn preview_mask_is_deterministic() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch_preview_mask(&mut gpu, &mut pool, "a", 128, &demo_path());
        dispatch_preview_mask(&mut gpu, &mut pool, "b", 128, &demo_path());
        let a = gpu.read_rgba8(pool.texture("a").unwrap(), 128);
        let b = gpu.read_rgba8(pool.texture("b").unwrap(), 128);
        assert_eq!(a, b);
    }

    fn straight_line() -> Path {
        Path {
            subpaths: vec![SubPath::new(
                vec![Point::new([0.1, 0.5]), Point::new([0.9, 0.5])],
                false,
            )],
            wrap: true,
        }
    }

    #[test]
    fn stamp_instances_spaces_evenly_along_a_straight_line() {
        let path = straight_line();
        let instances = stamp_instances(&path, 0.2, true, 0.0);
        // Length 0.8 at spacing 0.2 → samples at s = 0.0, 0.2, 0.4, 0.6, 0.8.
        assert_eq!(instances.len(), 5);
        assert!((instances[0].pos[0] - 0.1).abs() < 1e-5);
        assert!((instances[4].pos[0] - 0.9).abs() < 1e-5);
    }

    #[test]
    fn stamp_instances_align_to_curve_matches_the_tangent_angle() {
        let path = straight_line(); // tangent points along +x
        let instances = stamp_instances(&path, 0.5, true, 0.0);
        for inst in &instances {
            assert!(
                inst.angle.abs() < 1e-4,
                "expected ~0 rad, got {}",
                inst.angle
            );
        }
    }

    #[test]
    fn stamp_instances_without_align_ignores_the_tangent() {
        let path = straight_line();
        let extra = std::f32::consts::FRAC_PI_2;
        let instances = stamp_instances(&path, 0.5, false, extra);
        for inst in &instances {
            assert!((inst.angle - extra).abs() < 1e-5);
        }
    }

    #[test]
    fn stamp_instances_is_empty_for_an_empty_path() {
        assert!(stamp_instances(&Path::default(), 0.05, true, 0.0).is_empty());
    }

    #[test]
    fn stamp_instances_respects_the_hard_cap_on_a_pathological_spacing() {
        let path = straight_line();
        let instances = stamp_instances(&path, 1e-6, true, 0.0);
        assert_eq!(instances.len(), MAX_STAMP_INSTANCES);
    }

    #[test]
    fn dispatch_stamp_composites_the_stamp_texture_over_the_base() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        const RES: u32 = 128;

        // A disc stamp, bright in the middle — something unmistakably
        // different from a blank base.
        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "disc_stamp",
            RES,
            &crate::shape::ShapeParams::default(),
        );
        gpu.fill_rgba8(&mut pool, "blank_base", RES, [0, 0, 0, 255]);

        let path = straight_line();
        let instances = stamp_instances(&path, 0.1, true, 0.0);
        assert!(!instances.is_empty());

        dispatch_stamp(
            &mut gpu,
            &mut pool,
            "disc_stamp",
            "blank_base",
            "stamped",
            RES,
            &instances,
            0.15,
            path.wrap,
        );
        let stamped = gpu.read_rgba8(pool.texture("stamped").unwrap(), RES);
        let base = gpu.read_rgba8(pool.texture("blank_base").unwrap(), RES);
        assert_ne!(stamped, base, "stamping should change the output");
        let lit = stamped.iter().step_by(4).filter(|&&v| v > 0).count();
        assert!(lit > 0, "stamped output should have visible pixels");
    }

    /// Regression test for a real bug caught while building this feature:
    /// at the *default* Stamp Size, stamps are centered exactly on the
    /// curve the outline also draws, so a too-small size can land
    /// entirely inside the outline's own already-lit stroke — same
    /// `max()` value both ways, zero observable change, even though
    /// stamping genuinely ran. `PathParams::default`'s own Stamp Size is
    /// deliberately kept comfortably bigger than the outline's stroke
    /// width specifically to avoid this; this test guards that margin
    /// with a non-blank (outline) base, not the blank one every other
    /// `dispatch_stamp` test above uses.
    #[test]
    fn dispatch_stamp_is_visible_over_a_non_blank_base_at_default_size() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        const RES: u32 = 64;

        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "disc_stamp",
            RES,
            &crate::shape::ShapeParams::default(),
        );
        let path = straight_line();
        dispatch_preview_mask(&mut gpu, &mut pool, "outline_base", RES, &path);
        let base_before = gpu.read_rgba8(pool.texture("outline_base").unwrap(), RES);

        let defaults = PathParams::default();
        let instances = stamp_instances(
            &path,
            defaults.stamp_spacing as f64,
            defaults.stamp_align_to_curve,
            defaults.stamp_rotation_deg.to_radians(),
        );
        assert!(!instances.is_empty());

        dispatch_stamp(
            &mut gpu,
            &mut pool,
            "disc_stamp",
            "outline_base",
            "result",
            RES,
            &instances,
            defaults.stamp_size,
            path.wrap,
        );
        let result = gpu.read_rgba8(pool.texture("result").unwrap(), RES);
        assert_ne!(
            base_before, result,
            "stamping at the default size onto a non-blank base should still be visible"
        );
    }

    #[test]
    fn dispatch_stamp_with_no_instances_leaves_the_base_untouched() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        const RES: u32 = 64;

        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "disc_stamp",
            RES,
            &crate::shape::ShapeParams::default(),
        );
        dispatch_preview_mask(&mut gpu, &mut pool, "outline_base", RES, &demo_path());

        dispatch_stamp(
            &mut gpu,
            &mut pool,
            "disc_stamp",
            "outline_base",
            "unchanged",
            RES,
            &[],
            0.1,
            true,
        );
        let base = gpu.read_rgba8(pool.texture("outline_base").unwrap(), RES);
        let result = gpu.read_rgba8(pool.texture("unchanged").unwrap(), RES);
        assert_eq!(base, result);
    }
}
