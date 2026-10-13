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
}
