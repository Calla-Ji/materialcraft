// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Stitch — Step 1 of STITCH_NODE_SPEC.md's staged
//! build: stitch data types and the CPU stitch-rule generation (§4), ported
//! 1:1 from `stitch_reference.py` ("if this document and the code
//! disagree, the code wins" per that file's own docstring). No drape, no
//! GPU raster, no node registration yet.
//!
//! Reuses [`crate::path::sections`]/[`crate::path::Section`] directly —
//! `sec.at(s)` is exactly `path_reference.py`'s own `Section.at`, already
//! ported and golden-tested in Step 1 of the Path node's own build.
//!
//! `hash01` is bit-for-bit the same algorithm as Knit's own
//! `knit_stitch_info.wgsl` (that shader's own comment: "same as the Knit
//! node's style") — ported to Rust `u32` wrapping arithmetic here since
//! stitch-rule generation runs on the CPU, not the GPU.

use crate::path::Section;

pub const STITCH_KINDS: [StitchKind; 7] = [
    StitchKind::Running,
    StitchKind::Back,
    StitchKind::Bean,
    StitchKind::Stem,
    StitchKind::Split,
    StitchKind::Zigzag,
    StitchKind::Satin,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StitchKind {
    Running,
    Back,
    Bean,
    Stem,
    Split,
    Zigzag,
    Satin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Look {
    Hand,
    Machine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StemSide {
    Stem,
    Outline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlyDirection {
    S,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SatinMode {
    Parallel,
    Zigzag,
}

/// Mirrors `stitch_reference.py`'s own `StitchParams` dataclass field for
/// field — `None` fields there (meaning "take this from the Look preset")
/// are `Option` here for the same reason.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StitchParams {
    pub resolution: u32,
    pub stitch: StitchKind,
    pub look: Look,
    pub tile_mm: f64,
    pub stitch_length_mm: f64,
    pub gap_mm: Option<f64>,
    pub width_mm: f64,
    pub stem_width_mm: Option<f64>,
    pub density_mm: Option<f64>,
    pub bean_repeats: u32,
    pub stem_side: StemSide,
    pub zigzag_angle_deg: f64,
    pub pull_comp_mm: f64,
    pub satin_mitre: bool,
    pub corner_threshold_deg: f64,
    pub thread_mm: Option<f64>,
    pub plies: Option<u32>,
    pub twist_angle_deg: Option<f64>,
    pub ply_direction: PlyDirection,
    pub regularity: Option<f64>,
    pub height_range: f64,
    pub seed: u32,
}

impl Default for StitchParams {
    fn default() -> Self {
        Self {
            resolution: 768,
            stitch: StitchKind::Back,
            look: Look::Hand,
            tile_mm: 50.0,
            stitch_length_mm: 3.0,
            gap_mm: None,
            width_mm: 2.5,
            stem_width_mm: None,
            density_mm: None,
            bean_repeats: 1,
            stem_side: StemSide::Stem,
            zigzag_angle_deg: 0.0,
            pull_comp_mm: 0.0,
            satin_mitre: true,
            corner_threshold_deg: 30.0,
            thread_mm: None,
            plies: None,
            twist_angle_deg: None,
            ply_direction: PlyDirection::S,
            regularity: None,
            height_range: 4.0,
            seed: 1,
        }
    }
}

struct LookDefaults {
    thread_mm: f64,
    plies: u32,
    twist_angle_deg: f64,
    regularity: f64,
    satin_mode: SatinMode,
    density_factor: f64,
}

fn look_defaults(look: Look) -> LookDefaults {
    match look {
        // Two strands of stranded cotton lying nearly flat, small human
        // irregularity.
        Look::Hand => LookDefaults {
            thread_mm: 0.45,
            plies: 2,
            twist_angle_deg: 10.0,
            regularity: 0.7,
            satin_mode: SatinMode::Parallel,
            density_factor: 0.85,
        },
        // 40wt twisted thread, perfect spacing, satin zig-zags between rails.
        Look::Machine => LookDefaults {
            thread_mm: 0.30,
            plies: 2,
            twist_angle_deg: 35.0,
            regularity: 1.0,
            satin_mode: SatinMode::Zigzag,
            density_factor: 1.33,
        },
    }
}

/// `StitchParams::resolved()` — every length converted from mm into tile
/// units via `tile_mm`, and every `None` field filled from the Look preset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resolved {
    pub thread: f64,
    pub r: f64,
    pub plies: u32,
    pub twist_deg: f64,
    pub reg: f64,
    pub satin_mode: SatinMode,
    pub length: f64,
    pub gap: f64,
    pub width: f64,
    pub stem_w: f64,
    pub density: f64,
    pub pull: f64,
}

impl StitchParams {
    pub fn resolved(&self) -> Resolved {
        let look = look_defaults(self.look);
        let thread_mm = self.thread_mm.unwrap_or(look.thread_mm);
        Resolved {
            thread: thread_mm / self.tile_mm,
            r: 0.5 * thread_mm / self.tile_mm,
            plies: self.plies.unwrap_or(look.plies),
            twist_deg: self.twist_angle_deg.unwrap_or(look.twist_angle_deg),
            reg: self.regularity.unwrap_or(look.regularity),
            satin_mode: look.satin_mode,
            length: self.stitch_length_mm / self.tile_mm,
            gap: self.gap_mm.unwrap_or(self.stitch_length_mm) / self.tile_mm,
            width: self.width_mm / self.tile_mm,
            stem_w: self.stem_width_mm.unwrap_or(thread_mm) / self.tile_mm,
            density: self.density_mm.unwrap_or(thread_mm * look.density_factor) / self.tile_mm,
            pull: self.pull_comp_mm / self.tile_mm,
        }
    }
}

/// One thread lying between two needle holes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stitch {
    pub a: [f64; 2],
    pub b: [f64; 2],
    /// `false` = this end emerges through an earlier thread instead of a
    /// fresh hole (split stitch) — no dip in the drape there.
    pub dip_a: bool,
    pub dip_b: bool,
    /// Sideways bulge at the middle (bean stitch's threads spreading).
    pub bow: f64,
    /// May rest on nearly-parallel earlier threads instead of skipping
    /// them (stem/split's own rope-like overlap).
    pub stack: bool,
    /// Sewing order, global across every section — set by [`generate`],
    /// `0` from every `rule_*` function on its own.
    pub order: u32,
}

/// `stitch_reference.py`'s own `hash01`, ported bit for bit (32-bit
/// wrapping arithmetic — XOR commutes with the final `& 0xFFFFFF` mask
/// per-bit, so masking each product individually via `wrapping_mul` here
/// is equivalent to Python's arbitrary-precision multiply-then-mask).
pub fn hash01(a: u32, b: u32, c: u32) -> f64 {
    let mut h = a.wrapping_mul(73856093) ^ b.wrapping_mul(19349663) ^ c.wrapping_mul(83492791);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5BD1E995);
    h ^= h >> 15;
    f64::from(h & 0xFFFFFF) / f64::from(0x1000000u32)
}

fn jitter(k: u32, comp: u32, seed: u32) -> f64 {
    2.0 * hash01(k, comp, seed) - 1.0
}

/// Normal to the left of travel direction in y-down image coordinates:
/// travelling up `(0,-1)` → left is `(-1,0)`.
fn left_normal(t: [f64; 2]) -> [f64; 2] {
    [t[1], -t[0]]
}

/// Ink/Stitch-style even distribution: `n = ceil(L / target)`, so
/// `step = L / n` divides the section into equal pieces that both start
/// and end exactly on its own endpoints.
fn even_count(length: f64, target: f64) -> usize {
    (((length / target) - 1e-9).ceil() as i64).max(1) as usize
}

/// Python's `round()` uses round-half-to-even (banker's rounding);
/// Rust's `f64::round()` rounds half away from zero. `rule_running`'s own
/// stitch count depends on this at exact `.5` boundaries, so a plain
/// `.round()` here could silently produce a different stitch count than
/// the golden fixture at those boundaries — this matches Python exactly.
fn round_half_even(x: f64) -> f64 {
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

/// Places holes with hand irregularity (STITCH_NODE_SPEC.md §5) — a
/// running counter `k` (one `hash01` input) across every hole in the
/// whole generation pass, matching Python's own per-`StitchParams`-call
/// `HoleMaker` instance.
struct HoleMaker<'a> {
    params: &'a StitchParams,
    resolved: &'a Resolved,
    k: u32,
}

impl<'a> HoleMaker<'a> {
    fn new(params: &'a StitchParams, resolved: &'a Resolved) -> Self {
        Self {
            params,
            resolved,
            k: 0,
        }
    }

    /// `pin`: section-end holes (corners, path ends) never move along the
    /// path — a sharp corner always gets a hole exactly on it.
    fn hole(&mut self, sec: &Section, s: f64, lateral: f64, pin: bool, step: f64) -> [f64; 2] {
        let irr = 1.0 - self.resolved.reg;
        self.k += 1;
        let ds = if pin {
            0.0
        } else {
            irr * 0.15 * step * jitter(self.k, 0, self.params.seed)
        };
        let dl = irr * 0.25 * self.resolved.thread * jitter(self.k, 1, self.params.seed);
        let (p, t, _width) = sec.at(s + ds);
        let n = left_normal(t);
        let total = lateral + dl;
        [p[0] + n[0] * total, p[1] + n[1] * total]
    }
}

fn stitch(a: [f64; 2], b: [f64; 2]) -> Stitch {
    Stitch {
        a,
        b,
        dip_a: true,
        dip_b: true,
        bow: 0.0,
        stack: false,
        order: 0,
    }
}

// ---------------------------------------------------------------------
// Stitch rules (STITCH_NODE_SPEC.md §4 / `stitch_reference.py`'s own
// `rule_*` functions) — one per stitch type, all returning stitches in
// sewing order along this one section.
// ---------------------------------------------------------------------

fn rule_running(sec: &Section, _p: &StitchParams, r: &Resolved, h: &mut HoleMaker) -> Vec<Stitch> {
    let (ls_in, lg_in, length) = (r.length, r.gap, sec.length());
    let n = (round_half_even((length + lg_in) / (ls_in + lg_in)) as i64).max(1) as usize;
    let k = length / (n as f64 * ls_in + (n as f64 - 1.0) * lg_in);
    let (ls, lg) = (ls_in * k, lg_in * k);
    (0..n)
        .map(|i| {
            let s0 = i as f64 * (ls + lg);
            let a = h.hole(sec, s0, 0.0, i == 0, ls);
            let b = h.hole(sec, s0 + ls, 0.0, i == n - 1, ls);
            stitch(a, b)
        })
        .collect()
}

fn rule_back(sec: &Section, _p: &StitchParams, r: &Resolved, h: &mut HoleMaker) -> Vec<Stitch> {
    let n = even_count(sec.length(), r.length);
    let st = sec.length() / n as f64;
    let holes: Vec<[f64; 2]> = (0..=n)
        .map(|i| h.hole(sec, i as f64 * st, 0.0, i == 0 || i == n, st))
        .collect();
    (0..n).map(|i| stitch(holes[i], holes[i + 1])).collect()
}

/// Back-stitch holes, each stitch worked `2 * bean_repeats + 1` times
/// (forward, back, forward, ...) — the repeated threads share the holes
/// and bow apart by `0.55 * thread` in the middle, lying side by side.
fn rule_bean(sec: &Section, p: &StitchParams, r: &Resolved, h: &mut HoleMaker) -> Vec<Stitch> {
    let base = rule_back(sec, p, r, h);
    let m = 2 * p.bean_repeats + 1;
    let mut out = Vec::with_capacity(base.len() * m as usize);
    for st in &base {
        for j in 0..m {
            let off = (f64::from(j) - (f64::from(m) - 1.0) / 2.0) * 0.55 * r.thread;
            let (a, b) = if j % 2 == 0 {
                (st.a, st.b)
            } else {
                (st.b, st.a)
            };
            let bow = if j % 2 == 0 { off } else { -off };
            out.push(Stitch {
                bow,
                ..stitch(a, b)
            });
        }
    }
    out
}

/// RSN: each stitch begins halfway along the previous one; the thread is
/// kept to one side so the stitches slant and overlap into a rope. Stem
/// twists "/" going up the path; Outline twists "\".
fn rule_stem(sec: &Section, p: &StitchParams, r: &Resolved, h: &mut HoleMaker) -> Vec<Stitch> {
    let n = even_count(sec.length(), r.length);
    let st = sec.length() / n as f64;
    let half = st / 2.0;
    let side = if p.stem_side == StemSide::Stem {
        1.0
    } else {
        -1.0
    };
    let d = 0.5 * r.stem_w * side;
    let count = 2 * n - 1;
    (0..count)
        .map(|i| {
            let s0 = i as f64 * half;
            let a = h.hole(sec, s0, d, i == 0, half);
            let b = h.hole(sec, s0 + st, -d, i == count - 1, half);
            Stitch {
                stack: true,
                ..stitch(a, b)
            }
        })
        .collect()
}

/// RSN: each new stitch comes up through the centre of the previous one
/// (splitting it, `dip_a = false`) and goes down one stitch length on.
fn rule_split(sec: &Section, _p: &StitchParams, r: &Resolved, h: &mut HoleMaker) -> Vec<Stitch> {
    let n = even_count(sec.length(), r.length);
    let st = sec.length() / n as f64;
    let half = st / 2.0;
    let count = 2 * n - 1;
    (0..count)
        .map(|i| {
            let s0 = i as f64 * half;
            let a = h.hole(sec, s0, 0.0, i == 0, half);
            let b = h.hole(sec, s0 + st, 0.0, i == count - 1, half);
            Stitch {
                dip_a: i == 0,
                stack: true,
                ..stitch(a, b)
            }
        })
        .collect()
}

fn width_at(sec: &Section, s: f64, r: &Resolved) -> f64 {
    let (_, _, w) = sec.at(s);
    0.5 * r.width * f64::from(w) + r.pull
}

/// Ink/Stitch zigzag: holes alternate left/right of the path, peak-to-peak
/// spacing equal to the stitch length, rotated by `zigzag_angle`.
fn rule_zigzag(sec: &Section, p: &StitchParams, r: &Resolved, h: &mut HoleMaker) -> Vec<Stitch> {
    let spacing = r.length / 2.0;
    let n = even_count(sec.length(), spacing);
    let st = sec.length() / n as f64;
    let ang = p.zigzag_angle_deg.to_radians();
    let (cos_a, sin_a) = (ang.cos(), ang.sin());
    let mut holes = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let s = i as f64 * st;
        let (_, t, _) = sec.at(s);
        let nrm = left_normal(t);
        let rot = [
            nrm[0] * cos_a - nrm[1] * sin_a,
            nrm[0] * sin_a + nrm[1] * cos_a,
        ];
        let side = if i % 2 == 0 { 1.0 } else { -1.0 };
        let h0 = h.hole(sec, s, 0.0, i == 0 || i == n, st);
        let w = width_at(sec, s, r);
        holes.push([h0[0] + rot[0] * side * w, h0[1] + rot[1] * side * w]);
    }
    (0..n).map(|i| stitch(holes[i], holes[i + 1])).collect()
}

/// Clips segment `a`-`b` to the side of the line through `origin` in
/// `direction` that contains `keep_point` — the satin mitre's own corner
/// bisector cut.
fn clip_to_halfplane(
    a: [f64; 2],
    b: [f64; 2],
    origin: [f64; 2],
    direction: [f64; 2],
    keep_point: [f64; 2],
) -> Option<([f64; 2], [f64; 2])> {
    let nrm = [-direction[1], direction[0]];
    let dot_keep = (keep_point[0] - origin[0]) * nrm[0] + (keep_point[1] - origin[1]) * nrm[1];
    // `np.sign(x) or 1.0`: falls back to +1 exactly at zero, not just for
    // positive x.
    let sk = if dot_keep < 0.0 { -1.0 } else { 1.0 };
    let da = ((a[0] - origin[0]) * nrm[0] + (a[1] - origin[1]) * nrm[1]) * sk;
    let db = ((b[0] - origin[0]) * nrm[0] + (b[1] - origin[1]) * nrm[1]) * sk;
    if da >= 0.0 && db >= 0.0 {
        return Some((a, b));
    }
    if da < 0.0 && db < 0.0 {
        return None;
    }
    let t = da / (da - db);
    let m = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    if da < 0.0 {
        Some((m, b))
    } else {
        Some((a, m))
    }
}

/// Hand (parallel): flat stitches straight across the column, `density`
/// apart. Machine (zigzag): alternates between the two rails, every
/// stitch visible, `density` peak-to-peak. Width follows each point's own
/// width. Near a break, mitred along the corner bisector
/// (`Section::start_corner`/`end_corner`) if `satin_mitre` — the clip only
/// applies within 2 column widths of the corner, and a clipped stitch
/// shorter than half a thread is dropped.
fn rule_satin(sec: &Section, p: &StitchParams, r: &Resolved, h: &mut HoleMaker) -> Vec<Stitch> {
    let dens = r.density;
    let mut pairs: Vec<([f64; 2], [f64; 2], f64)> = Vec::new();

    if r.satin_mode == SatinMode::Parallel {
        let n = even_count(sec.length(), dens);
        let st = sec.length() / n as f64;
        for i in 0..n {
            let s = (i as f64 + 0.5) * st;
            let w = width_at(sec, s, r);
            let a = h.hole(sec, s, w, false, st);
            let b = h.hole(sec, s, -w, false, st);
            pairs.push((a, b, s));
        }
    } else {
        let n = even_count(sec.length(), dens / 2.0);
        let st = sec.length() / n as f64;
        let mut holes = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let s = i as f64 * st;
            let w = width_at(sec, s, r);
            let lateral = if i % 2 == 0 { w } else { -w };
            holes.push(h.hole(sec, s, lateral, i == 0 || i == n, st));
        }
        for i in 0..n {
            pairs.push((holes[i], holes[i + 1], (i as f64 + 0.5) * st));
        }
    }

    let width_max = sec.width.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let reach = 2.0 * (r.width * f64::from(width_max) + 2.0 * r.pull);

    let mut out = Vec::with_capacity(pairs.len());
    for (a, b, s) in pairs {
        let mut seg: Option<([f64; 2], [f64; 2])> = Some((a, b));
        if p.satin_mitre {
            for (corner, at_s) in [(sec.start_corner, 0.0), (sec.end_corner, sec.length())] {
                let (Some(corner), Some((sa, sb))) = (corner, seg) else {
                    continue;
                };
                if (s - at_s).abs() >= reach {
                    continue;
                }
                let keep_s = if sec.length() > 2.0 * reach {
                    at_s.max(reach).min(sec.length() - reach)
                } else {
                    sec.length() / 2.0
                };
                let (keep, _, _) = sec.at(keep_s);
                let (o, _, _) = sec.at(at_s);
                seg = clip_to_halfplane(sa, sb, o, corner, keep);
            }
        }
        if let Some((sa, sb)) = seg {
            let (dx, dy) = (sb[0] - sa[0], sb[1] - sa[1]);
            if (dx * dx + dy * dy).sqrt() > 0.5 * r.thread {
                out.push(stitch(sa, sb));
            }
        }
    }
    out
}

/// Every stitch for `path` under `params`, in sewing order (section by
/// section, along increasing arc length) — `stitch_reference.py`'s own
/// `generate`.
pub fn generate(path: &mc_core::Path, params: &StitchParams) -> Vec<Stitch> {
    let resolved = params.resolved();
    let mut holes = HoleMaker::new(params, &resolved);
    let rule: fn(&Section, &StitchParams, &Resolved, &mut HoleMaker) -> Vec<Stitch> =
        match params.stitch {
            StitchKind::Running => rule_running,
            StitchKind::Back => rule_back,
            StitchKind::Bean => rule_bean,
            StitchKind::Stem => rule_stem,
            StitchKind::Split => rule_split,
            StitchKind::Zigzag => rule_zigzag,
            StitchKind::Satin => rule_satin,
        };
    let mut out = Vec::new();
    for sec in crate::path::sections(path, params.corner_threshold_deg) {
        out.extend(rule(&sec, params, &resolved, &mut holes));
    }
    for (i, s) in out.iter_mut().enumerate() {
        s.order = i as u32;
    }
    out
}

// ---------------------------------------------------------------------
// Drape (STITCH_NODE_SPEC.md §6 / `stitch_reference.py`'s own `drape`) —
// the sequential 3D centre line of every thread. Runs on the CPU because
// it's inherently sequential: every thread depends on every thread before
// it (does it rest on the fabric, or on top of an earlier one it
// crosses?).
// ---------------------------------------------------------------------

/// Hole depth: a dipping end's centre goes to `-DIP * r` inside the hole.
const DIP: f64 = 0.5;
/// A thread dives into its hole over this many thread diameters.
const END_LEN: f64 = 1.5;
/// Earlier threads within this many degrees of the new thread's own
/// direction are ignored for occlusion (unless the new stitch has
/// `stack: true`) — real threads lying side by side spread out and
/// flatten instead of piling up.
const PARALLEL_DEG: f64 = 25.0;
/// A thread's centre never rises above `r + 2r * (MAX_LAYERS - 1)`.
const MAX_LAYERS: f64 = 3.0;

/// One thread's own 3D centre line, `x`/`y`/`z` all the same length —
/// `stitch_reference.py`'s own `lines` tuple, as a named struct.
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadLine {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
}

/// Minimum-image delta: when `wrap`, maps `d` into `[-0.5, 0.5]` (the
/// shortest way around a periodic `[0,1)` tile) — the same convention
/// `tube_raster.wgsl`'s own `min_image` uses on the GPU side.
fn wrap_delta(d: f64, wrap: bool) -> f64 {
    if wrap {
        d - d.round()
    } else {
        d
    }
}

fn smoothstep01(x: f64) -> f64 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Every stitch's own 3D centre line, in the same sewing order as
/// `stitches` — later entries can rest on top of earlier ones, never the
/// reverse, since real embroidery is sewn one thread at a time.
/// `resolution` is the node's own output resolution: sample density along
/// each thread scales with it (`n = max(8, ceil(L * res * 1.5))`), same as
/// the reference — a thread needs enough samples to splat cleanly at
/// whatever pixel density it'll actually be rasterized at.
pub fn drape(
    stitches: &[Stitch],
    params: &StitchParams,
    wrap: bool,
    resolution: u32,
) -> Vec<ThreadLine> {
    let resolved = params.resolved();
    let r = resolved.r;
    let cos_par = PARALLEL_DEG.to_radians().cos();

    let mut lines: Vec<ThreadLine> = Vec::with_capacity(stitches.len());
    // (center_x, center_y, half_extent_x, half_extent_y) per thread, for
    // the bounding-box prefilter — required for speed, must not change
    // the result (STITCH_NODE_SPEC.md §6.6).
    let mut boxes: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(stitches.len());
    let mut dirs: Vec<[f64; 2]> = Vec::with_capacity(stitches.len());

    for st in stitches {
        let mut ab = [st.b[0] - st.a[0], st.b[1] - st.a[1]];
        if wrap {
            ab = [wrap_delta(ab[0], true), wrap_delta(ab[1], true)];
        }
        let length = (ab[0] * ab[0] + ab[1] * ab[1]).sqrt() + 1e-12;
        let dvec = [ab[0] / length, ab[1] / length];
        let n = (((length * f64::from(resolution) * 1.5).ceil()) as i64).max(8) as usize;
        let perp = [ab[1] / length, -ab[0] / length];

        let mut us = vec![0.0; n];
        let mut xs = vec![0.0; n];
        let mut ys = vec![0.0; n];
        for (i, (u_out, (x_out, y_out))) in us
            .iter_mut()
            .zip(xs.iter_mut().zip(ys.iter_mut()))
            .enumerate()
        {
            let u = (i as f64 + 0.5) / n as f64;
            *u_out = u;
            let bow_term = st.bow * (std::f64::consts::PI * u).sin();
            *x_out = st.a[0] + ab[0] * u + perp[0] * bow_term;
            *y_out = st.a[1] + ab[1] * u + perp[1] * bow_term;
        }

        let mut z_rest = vec![r; n];
        let cx = xs.iter().sum::<f64>() / n as f64;
        let cy = ys.iter().sum::<f64>() / n as f64;
        let hx = 0.5 * ab[0].abs() + st.bow.abs() + 2.0 * r;
        let hy = 0.5 * ab[1].abs() + st.bow.abs() + 2.0 * r;

        for (j, &(bcx, bcy, bhx, bhy)) in boxes.iter().enumerate() {
            let dx = wrap_delta(cx - bcx, wrap);
            let dy = wrap_delta(cy - bcy, wrap);
            if dx.abs() > hx + bhx || dy.abs() > hy + bhy {
                continue;
            }
            if !st.stack && (dvec[0] * dirs[j][0] + dvec[1] * dirs[j][1]).abs() > cos_par {
                continue;
            }
            let other = &lines[j];
            for i in 0..n {
                let mut best = f64::NEG_INFINITY;
                for k in 0..other.x.len() {
                    let ddx = wrap_delta(xs[i] - other.x[k], wrap);
                    let ddy = wrap_delta(ys[i] - other.y[k], wrap);
                    let d2 = ddx * ddx + ddy * ddy;
                    let rr = (2.0 * r) * (2.0 * r) - d2;
                    if rr > 0.0 {
                        let cand = other.z[k] + rr.sqrt();
                        if cand > best {
                            best = cand;
                        }
                    }
                }
                if best > z_rest[i] {
                    z_rest[i] = best;
                }
            }
        }

        let layer_cap = r + 2.0 * r * (MAX_LAYERS - 1.0);
        for z in &mut z_rest {
            *z = z.min(layer_cap);
        }

        let e = (END_LEN * resolved.thread / length).min(0.4);
        let mut z = vec![0.0; n];
        for i in 0..n {
            let env_a = if st.dip_a {
                smoothstep01(us[i] / e)
            } else {
                1.0
            };
            let env_b = if st.dip_b {
                smoothstep01((1.0 - us[i]) / e)
            } else {
                1.0
            };
            let env = env_a * env_b;
            z[i] = -DIP * r + (z_rest[i] + DIP * r) * env;
        }

        lines.push(ThreadLine { x: xs, y: ys, z });
        boxes.push((cx, cy, hx, hy));
        dirs.push(dvec);
    }
    lines
}

// ---------------------------------------------------------------------
// Plies and GPU raster (STITCH_NODE_SPEC.md §7 / `stitch_reference.py`'s
// own `packing`/`render`) — reuses `crate::tube_raster` for Height/Mask/
// Ply ID (per that spec section's own instruction: "do not write a second
// rasterizer"), plus a small second pass (`stitch_info.wgsl`, mirroring
// `crate::knit`'s own `knit_stitch_info.wgsl`) that decodes the raw tag
// into Thread ID/Order/Direction and re-normalizes Height into Stitch's
// own one-sided `height_range * thread` scale (tube_raster's own `height`
// output is a *symmetric* clamp around z=0, the wrong shape for this).
// ---------------------------------------------------------------------

/// Ply packing radii: `a_p` (how far each ply's helix centre sits from the
/// stitch's own centre line) and `r_p` (each ply's own tube radius) — so
/// that `n` touching plies of radius `r_p` pack into an overall footprint
/// of radius `R`. `n <= 1` has nothing to pack: the single "ply" just *is*
/// the stitch's own full radius.
fn packing(radius: f64, n: u32) -> (f64, f64) {
    if n <= 1 {
        return (0.0, radius);
    }
    let s = (std::f64::consts::PI / f64::from(n)).sin();
    let r = radius / (1.0 + 1.0 / s);
    (r / s, r)
}

fn normalize3_py(v: [f64; 3]) -> [f64; 3] {
    // Matches `stitch_reference.py`'s own `v / (norm(v) + 1e-15)` — an
    // epsilon added to the denominator, not a branch on near-zero input.
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() + 1e-15;
    [v[0] / len, v[1] / len, v[2] / len]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `numpy.gradient(points, axis=0)` with its own default *uniform unit*
/// spacing (no explicit `x` given) — one-sided differences at the ends,
/// central difference (`/2`, not arc-length-weighted) in between. A
/// different, simpler formula than [`crate::path`]'s own `gradient_1d`,
/// which is for *non-uniform* arc-length spacing — Python's own `drape`
/// output samples are at uniform `u` steps, not uniform arc length, but
/// `render`'s own tangent-frame code still asks for the plain unit-spacing
/// gradient, so this matches that exactly rather than reusing the 2D one.
fn gradient_3d_unit_spacing(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let n = points.len();
    let mut out = vec![[0.0; 3]; n];
    if n < 2 {
        return out;
    }
    for i in 0..n {
        out[i] = if i == 0 {
            sub3(points[1], points[0])
        } else if i == n - 1 {
            sub3(points[n - 1], points[n - 2])
        } else {
            let d = sub3(points[i + 1], points[i - 1]);
            [d[0] / 2.0, d[1] / 2.0, d[2] / 2.0]
        };
    }
    out
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn push_capsule_chain(
    out: &mut Vec<crate::tube_raster::Segment>,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    radius: f32,
    tag: u32,
) {
    for i in 0..x.len().saturating_sub(1) {
        out.push(crate::tube_raster::Segment {
            p0: [x[i] as f32, y[i] as f32, z[i] as f32],
            r0: radius,
            p1: [x[i + 1] as f32, y[i + 1] as f32, z[i + 1] as f32],
            r1: radius,
            tag,
        });
    }
}

/// One capsule chain per ply per stitch (`stitch_reference.py`'s own
/// `render`'s inner `tubes` loop) — tags packed `order * 8 + ply`, the
/// same convention `crate::knit`'s own segments use and `tube_raster`'s
/// own shader assumes universally for its `ply_id` output (`best_tag % 8`).
fn build_tube_segments(
    stitches: &[Stitch],
    lines: &[ThreadLine],
    resolved: &Resolved,
    ply_direction_sgn: f64,
    seed: u32,
) -> Vec<crate::tube_raster::Segment> {
    let (ap, rp) = packing(resolved.r, resolved.plies);
    let mut segments = Vec::new();

    for (st, line) in stitches.iter().zip(lines) {
        let n = line.x.len();
        let tag_base = st.order * 8;
        if resolved.plies <= 1 {
            push_capsule_chain(
                &mut segments,
                &line.x,
                &line.y,
                &line.z,
                resolved.r as f32,
                tag_base,
            );
            continue;
        }

        let p3: Vec<[f64; 3]> = (0..n).map(|i| [line.x[i], line.y[i], line.z[i]]).collect();
        let tangents: Vec<[f64; 3]> = gradient_3d_unit_spacing(&p3)
            .into_iter()
            .map(normalize3_py)
            .collect();
        const UP: [f64; 3] = [0.0, 0.0, 1.0];
        let normals: Vec<[f64; 3]> = tangents
            .iter()
            .map(|t| normalize3_py(cross3(UP, *t)))
            .collect();
        let binormals: Vec<[f64; 3]> = tangents
            .iter()
            .zip(&normals)
            .map(|(t, nrm)| cross3(*t, *nrm))
            .collect();

        let mut s_arc = vec![0.0; n];
        for i in 1..n {
            let d = sub3(p3[i], p3[i - 1]);
            s_arc[i] = s_arc[i - 1] + (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        }
        let total_len = s_arc[n - 1];
        let turns =
            resolved.twist_deg.to_radians().tan() * total_len / (2.0 * std::f64::consts::PI * ap);
        let phase = 2.0 * std::f64::consts::PI * hash01(st.order, 7, seed);
        let total_len_guarded = total_len.max(1e-12);

        for k in 0..resolved.plies {
            let mut qx = vec![0.0; n];
            let mut qy = vec![0.0; n];
            let mut qz = vec![0.0; n];
            for i in 0..n {
                let th = ply_direction_sgn * 2.0 * std::f64::consts::PI * turns * s_arc[i]
                    / total_len_guarded
                    + 2.0 * std::f64::consts::PI * f64::from(k) / f64::from(resolved.plies)
                    + phase;
                let (cos_th, sin_th) = (th.cos(), th.sin());
                qx[i] = p3[i][0] + ap * (cos_th * normals[i][0] + sin_th * binormals[i][0]);
                qy[i] = p3[i][1] + ap * (cos_th * normals[i][1] + sin_th * binormals[i][1]);
                qz[i] = p3[i][2] + ap * (cos_th * normals[i][2] + sin_th * binormals[i][2]);
            }
            push_capsule_chain(&mut segments, &qx, &qy, &qz, rp as f32, tag_base + k);
        }
    }
    segments
}

/// This stitch's own thread direction angle (`stitch_reference.py`'s own
/// `math.atan2(*wrap_delta(s.b - s.a, path.wrap)[::-1])` — the `[::-1]`
/// reverses `[dx, dy]` to `[dy, dx]` before unpacking as `atan2`'s
/// positional `(y, x)` args, i.e. this is just the ordinary
/// `atan2(dy, dx)` convention).
fn thread_angle(st: &Stitch, wrap: bool) -> f32 {
    let dx = wrap_delta(st.b[0] - st.a[0], wrap);
    let dy = wrap_delta(st.b[1] - st.a[1], wrap);
    dy.atan2(dx) as f32
}

fn pack_f32_slice(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn pack_stitch_info_params(
    resolution: u32,
    n_stitches: u32,
    seed: u32,
    zmax: f32,
    height_norm: f32,
) -> [u8; 20] {
    let mut bytes = [0u8; 20];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&n_stitches.to_le_bytes());
    bytes[8..12].copy_from_slice(&seed.to_le_bytes());
    bytes[12..16].copy_from_slice(&zmax.to_le_bytes());
    bytes[16..20].copy_from_slice(&height_norm.to_le_bytes());
    bytes
}

/// Where [`dispatch_render`] stores its six real outputs.
pub struct RenderOutputKeys<'a> {
    pub height: &'a str,
    pub mask: &'a str,
    pub ply_id: &'a str,
    pub thread_id: &'a str,
    pub order: &'a str,
    pub direction: &'a str,
}

/// Generates, drapes and rasterizes `path` under `params` into `keys`' six
/// output textures. A `path` whose own stitch rule produces no stitches at
/// all (a degenerate, e.g. zero-length, curve) fills every output blank
/// rather than dispatching a degenerate GPU pass, same fallback
/// `dispatch_preview_mask` already uses.
pub fn dispatch_render(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    keys: RenderOutputKeys,
    path: &mc_core::Path,
    params: &StitchParams,
) {
    let resolution = params.resolution;
    let resolved = params.resolved();
    let stitches = generate(path, params);

    if stitches.is_empty() {
        gpu.fill_rgba8(pool, keys.height, resolution, [0, 0, 0, 255]);
        gpu.fill_rgba8(pool, keys.mask, resolution, [0, 0, 0, 255]);
        gpu.fill_rgba8(pool, keys.ply_id, resolution, [0, 0, 0, 255]);
        gpu.fill_rgba8(pool, keys.thread_id, resolution, [0, 0, 0, 255]);
        gpu.fill_rgba8(pool, keys.order, resolution, [0, 0, 0, 255]);
        gpu.fill_rgba8(pool, keys.direction, resolution, [128, 128, 128, 255]);
        return;
    }

    let lines = drape(&stitches, params, path.wrap, resolution);
    let ply_sgn = if params.ply_direction == PlyDirection::S {
        1.0
    } else {
        -1.0
    };
    let segments = build_tube_segments(&stitches, &lines, &resolved, ply_sgn, params.seed);

    // Generous enough that no real z value (bounded by MAX_LAYERS plus one
    // sphere radius) ever actually clips against tube_raster's own
    // symmetric clamp — it only needs to be invertible, not tight.
    let zmax = (20.0 * resolved.r + 1.0) as f32;
    let scratch_height_key = format!("{}__stitch_raw_height_scratch", keys.height);
    let scratch_cell_key = format!("{}__stitch_cell_scratch", keys.height);
    crate::tube_raster::dispatch(
        gpu,
        pool,
        crate::tube_raster::RasterOutputKeys {
            height: &scratch_height_key,
            mask: keys.mask,
            ply_id: keys.ply_id,
            cell: &scratch_cell_key,
        },
        resolution,
        &segments,
        resolved.plies.max(1),
        zmax,
    );

    let thread_angles: Vec<f32> = stitches
        .iter()
        .map(|st| thread_angle(st, path.wrap))
        .collect();
    let height_norm = (params.height_range * resolved.thread) as f32;

    dispatch_stitch_info(
        gpu,
        pool,
        &scratch_cell_key,
        &scratch_height_key,
        keys.height,
        keys.thread_id,
        keys.order,
        keys.direction,
        resolution,
        &thread_angles,
        params.seed,
        zmax,
        height_norm,
    );
}

#[allow(clippy::too_many_arguments)]
fn dispatch_stitch_info(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    cell_key: &str,
    scratch_height_key: &str,
    height_key: &str,
    thread_id_key: &str,
    order_key: &str,
    direction_key: &str,
    resolution: u32,
    thread_angles: &[f32],
    seed: u32,
    zmax: f32,
    height_norm: f32,
) {
    let shader_source = crate::shaders::full_source("stitch_info.wgsl");
    let pipeline = gpu.compute_pipeline("stitch_info", &shader_source, "main");

    let param_bytes = pack_stitch_info_params(
        resolution,
        thread_angles.len() as u32,
        seed,
        zmax,
        height_norm,
    );
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stitch_info_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let angles_bytes = pack_f32_slice(thread_angles);
    let angles_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stitch_info_thread_angles"),
        size: (angles_bytes.len() as u64).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !angles_bytes.is_empty() {
        gpu.queue.write_buffer(&angles_buffer, 0, &angles_bytes);
    }

    let cell_view = pool
        .get_or_create(
            &gpu.device,
            cell_key,
            resolution,
            wgpu::TextureFormat::R32Uint,
        )
        .clone();
    let scratch_height_view = pool
        .get_or_create(
            &gpu.device,
            scratch_height_key,
            resolution,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .clone();
    let height_view = pool
        .get_or_create(
            &gpu.device,
            height_key,
            resolution,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .clone();
    let thread_id_view = pool
        .get_or_create(
            &gpu.device,
            thread_id_key,
            resolution,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .clone();
    let order_view = pool
        .get_or_create(
            &gpu.device,
            order_key,
            resolution,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .clone();
    let direction_view = pool
        .get_or_create(
            &gpu.device,
            direction_key,
            resolution,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .clone();

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stitch_info_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: angles_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&cell_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&scratch_height_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&height_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&thread_id_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&order_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&direction_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stitch_info_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("stitch_info_pass"),
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
    use super::*;

    const DEMO_PATH_JSON: &str = include_str!("../../../tests/golden/path/demo_path.json");
    const PARAMS_JSON: &str = include_str!("../../../tests/golden/stitch/params.json");
    const STITCHES_JSON: &str = include_str!("../../../tests/golden/stitch/stitches.json");

    fn demo_path() -> mc_core::Path {
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

    fn params_from_json(v: &serde_json::Value) -> StitchParams {
        let stitch = match v["stitch"].as_str().unwrap() {
            "running" => StitchKind::Running,
            "back" => StitchKind::Back,
            "bean" => StitchKind::Bean,
            "stem" => StitchKind::Stem,
            "split" => StitchKind::Split,
            "zigzag" => StitchKind::Zigzag,
            "satin" => StitchKind::Satin,
            other => panic!("unknown stitch kind {other:?}"),
        };
        let look = match v["look"].as_str().unwrap() {
            "hand" => Look::Hand,
            "machine" => Look::Machine,
            other => panic!("unknown look {other:?}"),
        };
        let stem_side = match v["stem_side"].as_str().unwrap() {
            "stem" => StemSide::Stem,
            "outline" => StemSide::Outline,
            other => panic!("unknown stem_side {other:?}"),
        };
        let ply_direction = match v["ply_direction"].as_str().unwrap() {
            "S" => PlyDirection::S,
            "Z" => PlyDirection::Z,
            other => panic!("unknown ply_direction {other:?}"),
        };
        StitchParams {
            resolution: v["resolution"].as_u64().unwrap() as u32,
            stitch,
            look,
            tile_mm: v["tile_mm"].as_f64().unwrap(),
            stitch_length_mm: v["stitch_length_mm"].as_f64().unwrap(),
            gap_mm: v["gap_mm"].as_f64(),
            width_mm: v["width_mm"].as_f64().unwrap(),
            stem_width_mm: v["stem_width_mm"].as_f64(),
            density_mm: v["density_mm"].as_f64(),
            bean_repeats: v["bean_repeats"].as_u64().unwrap() as u32,
            stem_side,
            zigzag_angle_deg: v["zigzag_angle"].as_f64().unwrap(),
            pull_comp_mm: v["pull_comp_mm"].as_f64().unwrap(),
            satin_mitre: v["satin_mitre"].as_bool().unwrap(),
            corner_threshold_deg: v["corner_threshold"].as_f64().unwrap(),
            thread_mm: v["thread_mm"].as_f64(),
            plies: v["plies"].as_u64().map(|x| x as u32),
            twist_angle_deg: v["twist_angle"].as_f64(),
            ply_direction,
            regularity: v["regularity"].as_f64(),
            height_range: v["height_range"].as_f64().unwrap(),
            seed: v["seed"].as_u64().unwrap() as u32,
        }
    }

    #[test]
    fn hash01_matches_knit_stitch_infos_own_shader_port() {
        // Spot-check against knit_stitch_info.wgsl's identical algorithm —
        // both are ports of the same `stitch_reference.py`/`knit_reference.py`
        // `hash01`, so they must agree bit for bit on any input.
        let a = hash01(1, 0, 1);
        let b = hash01(1, 0, 1);
        assert_eq!(a, b, "deterministic");
        assert!((0.0..1.0).contains(&a));
        assert_ne!(hash01(1, 0, 1), hash01(2, 0, 1));
        assert_ne!(hash01(1, 0, 1), hash01(1, 1, 1));
    }

    #[test]
    fn round_half_even_matches_pythons_own_banker_rounding() {
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(3.5), 4.0);
        assert_eq!(round_half_even(2.4), 2.0);
        assert_eq!(round_half_even(2.6), 3.0);
    }

    #[test]
    fn generate_matches_golden_stitches_for_every_case() {
        let params_json: serde_json::Value = serde_json::from_str(PARAMS_JSON).unwrap();
        let stitches_json: serde_json::Value = serde_json::from_str(STITCHES_JSON).unwrap();
        let path = demo_path();

        let cases = params_json.as_object().expect("params.json is an object");
        assert_eq!(cases.len(), 12, "expected all 12 acceptance-test cases");

        for (case_name, case_params) in cases {
            let params = params_from_json(case_params);
            let golden_list = stitches_json[case_name]
                .as_array()
                .unwrap_or_else(|| panic!("missing golden stitches for case {case_name:?}"));

            let generated = generate(&path, &params);

            assert_eq!(
                generated.len(),
                golden_list.len(),
                "case {case_name}: stitch count mismatch"
            );

            for (i, (st, g)) in generated.iter().zip(golden_list).enumerate() {
                assert_close2(st.a, arr(&g["a"]), 1e-6, &format!("{case_name}[{i}].a"));
                assert_close2(st.b, arr(&g["b"]), 1e-6, &format!("{case_name}[{i}].b"));
                assert_eq!(
                    st.dip_a,
                    g["dip_a"].as_bool().unwrap(),
                    "{case_name}[{i}].dip_a"
                );
                assert_eq!(
                    st.dip_b,
                    g["dip_b"].as_bool().unwrap(),
                    "{case_name}[{i}].dip_b"
                );
                assert_eq!(
                    st.stack,
                    g["stack"].as_bool().unwrap(),
                    "{case_name}[{i}].stack"
                );
                let gbow = g["bow"].as_f64().unwrap();
                assert!(
                    (st.bow - gbow).abs() < 1e-6,
                    "{case_name}[{i}].bow: actual={} expected={gbow}",
                    st.bow
                );
            }
        }
    }

    // `drape()` has no golden fixture in the reference package itself
    // (STITCH_NODE_SPEC.md §2's own table: Drape is validated "implicitly,
    // through the height PNGs" — i.e. only indirectly, via Step 3's GPU
    // raster). To still get a real bit-level check now rather than only a
    // property test, `tests/golden/stitch/drape.json` was generated by
    // running `stitch_reference.py`'s own `drape()` directly (numpy
    // installed locally) against a handful of real stitches from the demo
    // path, truncated short to keep the fixture small — see the Python
    // snippet in this module's own git history for exactly how.
    const DRAPE_JSON: &str = include_str!("../../../tests/golden/stitch/drape.json");

    fn stitch_from_json(v: &serde_json::Value) -> Stitch {
        Stitch {
            a: arr(&v["a"]),
            b: arr(&v["b"]),
            dip_a: v["dip_a"].as_bool().unwrap(),
            dip_b: v["dip_b"].as_bool().unwrap(),
            bow: v["bow"].as_f64().unwrap(),
            stack: v["stack"].as_bool().unwrap(),
            order: 0,
        }
    }

    #[test]
    fn drape_matches_a_self_generated_golden_for_every_case() {
        let drape_json: serde_json::Value = serde_json::from_str(DRAPE_JSON).unwrap();
        let cases = drape_json.as_object().unwrap();
        assert_eq!(cases.len(), 3);

        for (case_name, case) in cases {
            let stitch_kind = match case["stitch_params"]["stitch"].as_str().unwrap() {
                "running" => StitchKind::Running,
                "bean" => StitchKind::Bean,
                "stem" => StitchKind::Stem,
                other => panic!("unknown stitch kind {other:?} in case {case_name}"),
            };
            let resolution = case["resolution"].as_u64().unwrap() as u32;
            let wrap = case["wrap"].as_bool().unwrap();
            let params = StitchParams {
                stitch: stitch_kind,
                resolution,
                ..Default::default()
            };

            let stitches: Vec<Stitch> = case["stitches"]
                .as_array()
                .unwrap()
                .iter()
                .map(stitch_from_json)
                .collect();
            let golden_lines = case["lines"].as_array().unwrap();
            assert_eq!(stitches.len(), golden_lines.len());

            let lines = drape(&stitches, &params, wrap, resolution);
            assert_eq!(lines.len(), golden_lines.len(), "case {case_name}");

            for (i, (line, g)) in lines.iter().zip(golden_lines).enumerate() {
                let gx: Vec<f64> = g["x"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                let gy: Vec<f64> = g["y"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                let gz: Vec<f64> = g["z"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                assert_eq!(
                    line.x.len(),
                    gx.len(),
                    "{case_name}[{i}]: sample count mismatch"
                );
                for k in 0..line.x.len() {
                    assert!(
                        (line.x[k] - gx[k]).abs() < 1e-6,
                        "{case_name}[{i}].x[{k}]: actual={} expected={}",
                        line.x[k],
                        gx[k]
                    );
                    assert!(
                        (line.y[k] - gy[k]).abs() < 1e-6,
                        "{case_name}[{i}].y[{k}]: actual={} expected={}",
                        line.y[k],
                        gy[k]
                    );
                    assert!(
                        (line.z[k] - gz[k]).abs() < 1e-6,
                        "{case_name}[{i}].z[{k}]: actual={} expected={}",
                        line.z[k],
                        gz[k]
                    );
                }
            }
        }
    }

    #[test]
    fn drape_caps_height_at_the_max_layers_limit() {
        // A pile of coincident, stacking stitches should never rise past
        // r + 2r*(MAX_LAYERS - 1) no matter how many layer on top of each
        // other.
        let params = StitchParams::default();
        let r = params.resolved().r;
        let stitches: Vec<Stitch> = (0..10)
            .map(|_| Stitch {
                a: [0.4, 0.4],
                b: [0.5, 0.4],
                dip_a: false,
                dip_b: false,
                bow: 0.0,
                stack: true,
                order: 0,
            })
            .collect();
        let lines = drape(&stitches, &params, true, 64);
        let cap = r + 2.0 * r * (MAX_LAYERS - 1.0);
        for line in &lines {
            for &z in &line.z {
                assert!(z <= cap + 1e-9, "z={z} exceeds cap {cap}");
            }
        }
    }

    #[test]
    fn drape_dips_toward_the_hole_at_a_dipping_end() {
        let params = StitchParams::default();
        let r = params.resolved().r;
        let stitches = [Stitch {
            a: [0.2, 0.2],
            b: [0.4, 0.2],
            dip_a: true,
            dip_b: true,
            bow: 0.0,
            stack: false,
            order: 0,
        }];
        let lines = drape(&stitches, &params, true, 64);
        let line = &lines[0];
        // The very first sample (near u=0, a dipping end) should be well
        // below the resting height r; the middle sample should be at r
        // (nothing else to rest on).
        assert!(line.z[0] < r * 0.5, "z[0]={} should dip low", line.z[0]);
        let mid = line.z.len() / 2;
        assert!(
            (line.z[mid] - r).abs() < 1e-6,
            "z[mid]={} should rest at r={r}",
            line.z[mid]
        );
    }

    fn back_hand_params(resolution: u32) -> StitchParams {
        StitchParams {
            stitch: StitchKind::Back,
            look: Look::Hand,
            resolution,
            ..Default::default()
        }
    }

    #[test]
    fn dispatch_render_is_deterministic() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let path = demo_path();
        let params = back_hand_params(128);

        for suffix in ["a", "b"] {
            dispatch_render(
                &mut gpu,
                &mut pool,
                RenderOutputKeys {
                    height: &format!("height_{suffix}"),
                    mask: &format!("mask_{suffix}"),
                    ply_id: &format!("ply_id_{suffix}"),
                    thread_id: &format!("thread_id_{suffix}"),
                    order: &format!("order_{suffix}"),
                    direction: &format!("direction_{suffix}"),
                },
                &path,
                &params,
            );
        }
        let height_a = gpu.read_rgba8(pool.texture("height_a").unwrap(), 128);
        let height_b = gpu.read_rgba8(pool.texture("height_b").unwrap(), 128);
        assert_eq!(
            height_a, height_b,
            "two renders with the same inputs must be bit-identical"
        );
    }

    #[test]
    fn dispatch_render_tiles_seamlessly() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let path = demo_path();
        let params = back_hand_params(128);

        dispatch_render(
            &mut gpu,
            &mut pool,
            RenderOutputKeys {
                height: "seam_height",
                mask: "seam_mask",
                ply_id: "seam_ply_id",
                thread_id: "seam_thread_id",
                order: "seam_order",
                direction: "seam_direction",
            },
            &path,
            &params,
        );
        let res = 128usize;
        let pixels = gpu.read_rgba8(pool.texture("seam_height").unwrap(), res as u32);
        let at = |x: usize, y: usize| pixels[(y * res + x) * 4] as f32 / 255.0;

        // STITCH_NODE_SPEC.md §11's own seam_ok: the mean difference across
        // the tile edge must be no larger than 1.05x the largest mean step
        // between neighboring *interior* columns/rows.
        let col_diff = |x: usize| -> f32 {
            (0..res)
                .map(|y| (at(x, y) - at(x.wrapping_sub(1).min(res - 1), y)).abs())
                .sum::<f32>()
                / res as f32
        };
        let row_diff = |y: usize| -> f32 {
            (0..res)
                .map(|x| (at(x, y) - at(x, y.wrapping_sub(1).min(res - 1))).abs())
                .sum::<f32>()
                / res as f32
        };
        let max_interior_col = (1..res).map(col_diff).fold(0.0f32, f32::max);
        let max_interior_row = (1..res).map(row_diff).fold(0.0f32, f32::max);
        let seam_col = (0..res)
            .map(|y| (at(0, y) - at(res - 1, y)).abs())
            .sum::<f32>()
            / res as f32;
        let seam_row = (0..res)
            .map(|x| (at(x, 0) - at(x, res - 1)).abs())
            .sum::<f32>()
            / res as f32;

        assert!(
            seam_col <= 1.05 * max_interior_col + 1e-4,
            "column seam {seam_col} exceeds 1.05x the largest interior step {max_interior_col}"
        );
        assert!(
            seam_row <= 1.05 * max_interior_row + 1e-4,
            "row seam {seam_row} exceeds 1.05x the largest interior step {max_interior_row}"
        );
    }

    /// Full end-to-end check against the reference package's own
    /// `golden/back_hand_{height,mask}.png`, using STITCH_NODE_SPEC.md
    /// §11's own acceptance thresholds — this is the one case (of the
    /// package's 12) copied into the project; the others share the exact
    /// same code path (just different stitch/look params), so this is
    /// strong evidence for all of them, not proof of each individually.
    #[test]
    fn dispatch_render_matches_golden_png_for_back_hand() {
        const HEIGHT_PNG: &[u8] =
            include_bytes!("../../../tests/golden/stitch/back_hand_height.png");
        const MASK_PNG: &[u8] = include_bytes!("../../../tests/golden/stitch/back_hand_mask.png");

        fn load_luma16_normalized(bytes: &[u8]) -> Vec<f32> {
            image::load_from_memory(bytes)
                .expect("decode golden png")
                .into_luma16()
                .into_raw()
                .iter()
                .map(|&v| f32::from(v) / 65535.0)
                .collect()
        }

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let path = demo_path();
        let params = back_hand_params(768);

        dispatch_render(
            &mut gpu,
            &mut pool,
            RenderOutputKeys {
                height: "golden_height",
                mask: "golden_mask",
                ply_id: "golden_ply_id",
                thread_id: "golden_thread_id",
                order: "golden_order",
                direction: "golden_direction",
            },
            &path,
            &params,
        );

        let res = 768usize;
        let actual_height = gpu.read_rgba8(pool.texture("golden_height").unwrap(), res as u32);
        let actual_mask = gpu.read_rgba8(pool.texture("golden_mask").unwrap(), res as u32);
        let golden_height = load_luma16_normalized(HEIGHT_PNG);
        let golden_mask = load_luma16_normalized(MASK_PNG);
        assert_eq!(golden_height.len(), res * res);

        let mut abs_diffs = Vec::with_capacity(res * res);
        for i in 0..res * res {
            let actual = f32::from(actual_height[i * 4]) / 255.0;
            abs_diffs.push((actual - golden_height[i]).abs());
        }
        let mad: f32 = abs_diffs.iter().sum::<f32>() / abs_diffs.len() as f32;
        let off_by_a_lot = abs_diffs.iter().filter(|&&d| d > 0.1).count();
        assert!(
            mad < 0.02,
            "height mean absolute difference {mad} exceeds 0.02"
        );
        assert!(
            (off_by_a_lot as f32) < 0.01 * (res * res) as f32,
            "{off_by_a_lot} pixels off by more than 0.1 (limit 1% of {})",
            res * res
        );

        let (mut intersection, mut union) = (0usize, 0usize);
        for i in 0..res * res {
            let actual_on = actual_mask[i * 4] > 0;
            let expected_on = golden_mask[i] > 0.5;
            if actual_on && expected_on {
                intersection += 1;
            }
            if actual_on || expected_on {
                union += 1;
            }
        }
        let iou = intersection as f32 / union.max(1) as f32;
        assert!(iou > 0.97, "mask IoU {iou} is below 0.97");
    }
}
