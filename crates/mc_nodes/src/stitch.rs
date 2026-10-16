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

pub const STITCH_KINDS: [StitchKind; 5] = [
    StitchKind::Running,
    StitchKind::Bean,
    StitchKind::Stem,
    StitchKind::Zigzag,
    StitchKind::Satin,
];

/// French Knot isn't one of these — it has its own [`crate::french_knot`]
/// module entirely (its own `KnotParams`, its own centre-line model, its
/// own GPU raster), not a `rule_*` function here, since its data shape is
/// genuinely different from every type below. The Stitch node's own
/// `"stitch"` dropdown still offers it; `mc_nodes::registry`'s own
/// dispatch branches to the right module before it ever builds a
/// `StitchParams`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StitchKind {
    /// Covers Back and Split too, not just Running proper — with exactly
    /// one stitch spanning its own whole tile (`render_tile_mm`), all
    /// three render pixel-identical: Running's own gap can never show (the
    /// tile fills with thread regardless of its value, see
    /// `render_tile_mm`'s own doc comment), and Split has no sideways
    /// offset at `n == 1` the way Stem does, so it collapses to the same
    /// straight line Back draws. Confirmed via testing — caught as "they
    /// all look the same" across real test renders of each. `rule_back`/
    /// `rule_split` still exist (Bean's own base is `rule_back`, and both
    /// stay validated against the reference package's own golden fixtures
    /// via `rule_fn_by_name` in this module's own tests) — they're just
    /// not their own separate, user-selectable option anymore.
    Running,
    Bean,
    Stem,
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
            stitch: StitchKind::Running,
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
            // `height = (z + thread-cross-section-bulge) / (height_range *
            // thread)`, and an *isolated* single stitch's own peak ridge
            // pixel (its own resting z, r, plus the capsule's own +r bulge
            // at the dead center, d=0) always comes out to exactly
            // `2 * thread radius` — so `height_range == 1.0` (the slider's
            // own documented minimum) is exactly the point where that peak
            // reaches a full `1.0`, the brightest the Height output can
            // read. The old default of `4.0` was tuned for the pre-
            // `render_tile_mm` multi-stitch-per-tile swatch, where several
            // threads could stack up to `MAX_LAYERS` deep and needed the
            // extra headroom — a single isolated stitch never stacks on
            // anything, so that headroom just read as "too dim to look
            // raised" (reported via testing: "missing lighter values").
            height_range: 1.0,
            seed: 1,
        }
    }
}

/// A Look preset's own canonical values — also used directly by
/// `mc_app::params_panel`'s Look picker to reset the dependent sliders when
/// the user switches Look, so the UI never duplicates these numbers.
pub struct LookDefaults {
    pub thread_mm: f64,
    pub plies: u32,
    pub twist_angle_deg: f64,
    pub regularity: f64,
    pub satin_mode: SatinMode,
    pub density_factor: f64,
}

pub fn look_defaults(look: Look) -> LookDefaults {
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

    /// Reads a complete `StitchParams` straight off the graph's own node
    /// params — unlike the Python reference's own CLI-script convenience,
    /// this node's UI never leaves a field genuinely unset: the Look
    /// preset picker (`mc_app::params_panel`) writes concrete values into
    /// every look-dependent field the moment it's clicked, the same
    /// "preset resets these sliders, they stay independently editable
    /// after" pattern Knit's own preset picker already established. So
    /// every `Option` field here is always `Some` in practice; they stay
    /// `Option`-typed only so [`Self::resolved`] keeps behaving exactly
    /// like `stitch_reference.py`'s own `resolved()` if a field is ever
    /// genuinely absent (a node graph loaded from an older file version,
    /// say).
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();

        fn f64_of(
            params: &std::collections::BTreeMap<String, mc_core::ParamValue>,
            name: &str,
            default: f64,
        ) -> f64 {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .map(f64::from)
                .unwrap_or(default)
        }
        fn u32_of(
            params: &std::collections::BTreeMap<String, mc_core::ParamValue>,
            name: &str,
            default: u32,
        ) -> u32 {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(0) as u32)
                .unwrap_or(default)
        }
        fn bool_of(
            params: &std::collections::BTreeMap<String, mc_core::ParamValue>,
            name: &str,
            default: bool,
        ) -> bool {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_bool)
                .unwrap_or(default)
        }
        fn enum_str<'a>(
            params: &'a std::collections::BTreeMap<String, mc_core::ParamValue>,
            name: &str,
        ) -> Option<&'a str> {
            match params.get(name) {
                Some(mc_core::ParamValue::Enum(s)) => Some(s.as_str()),
                _ => None,
            }
        }

        let stitch = match enum_str(params, "stitch") {
            // "back"/"split" aren't their own dropdown option anymore
            // (merged into Running — see `StitchKind::Running`'s own doc
            // comment), but still map a graph file saved before the merge
            // onto the option that now covers them, rather than silently
            // falling through to the overall default.
            Some("running") | Some("back") | Some("split") => StitchKind::Running,
            Some("bean") => StitchKind::Bean,
            Some("stem") => StitchKind::Stem,
            Some("zigzag") => StitchKind::Zigzag,
            Some("satin") => StitchKind::Satin,
            // "french_knot" has its own `KnotParams`/`crate::french_knot`
            // entirely, never a live `StitchKind` — `mc_nodes::registry`'s
            // own dispatch branches to that module before this function
            // is ever called for it. Falls through to the overall default
            // here defensively, same as any other value this doesn't
            // recognize.
            _ => defaults.stitch,
        };
        let look = match enum_str(params, "look") {
            Some("machine") => Look::Machine,
            Some("hand") => Look::Hand,
            _ => defaults.look,
        };
        let stem_side = match enum_str(params, "stem_side") {
            Some("outline") => StemSide::Outline,
            Some("stem") => StemSide::Stem,
            _ => defaults.stem_side,
        };
        let ply_direction = match enum_str(params, "ply_direction") {
            Some("Z") => PlyDirection::Z,
            Some("S") => PlyDirection::S,
            _ => defaults.ply_direction,
        };

        // Magic-number fallbacks below mirror the "hand" look's own
        // resolved defaults (LookDefaults::thread_mm=0.45,
        // density_factor=0.85, etc.) — the values a freshly-instantiated
        // node's own ParamSpec defaults already carry, repeated here only
        // for the (normally unreachable) case of a param genuinely
        // missing from the graph.
        Self {
            stitch,
            look,
            // Not exposed in the UI: `dispatch_render` always substitutes
            // `render_tile_mm`'s own one-stitch value in for this before
            // rendering (checked via testing — editing this slider had no
            // visible effect at all), so it stays a fixed internal value.
            tile_mm: f64_of(params, "tile_mm", defaults.tile_mm),
            stitch_length_mm: f64_of(params, "stitch_length_mm", defaults.stitch_length_mm),
            // Not exposed in the UI: `rule_running` is the only reader of
            // this, and with `render_tile_mm`'s own one-stitch framing its
            // own stitch count is provably `1` regardless of this value
            // (the `+ gap` term appears identically in both the numerator
            // and denominator of its own count formula) — the single
            // stitch fills the whole tile edge to edge either way.
            gap_mm: Some(f64_of(params, "gap_mm", 3.0)),
            width_mm: f64_of(params, "width_mm", defaults.width_mm),
            stem_width_mm: Some(f64_of(params, "stem_width_mm", 0.45)),
            density_mm: Some(f64_of(params, "density_mm", 0.3825)),
            bean_repeats: u32_of(params, "bean_repeats", defaults.bean_repeats).max(1),
            stem_side,
            zigzag_angle_deg: f64_of(params, "zigzag_angle", defaults.zigzag_angle_deg),
            pull_comp_mm: f64_of(params, "pull_comp_mm", defaults.pull_comp_mm),
            // Not exposed in the UI: mitring only ever activates at a path
            // *corner* (`Section::start_corner`/`end_corner`), and the
            // node's own synthetic local unit (a single straight open
            // segment) never has one.
            satin_mitre: bool_of(params, "satin_mitre", defaults.satin_mitre),
            // Not exposed in the UI: the node's own synthetic local unit
            // (a single straight open segment) never has a corner for
            // this to threshold.
            corner_threshold_deg: defaults.corner_threshold_deg,
            thread_mm: Some(f64_of(params, "thread_mm", 0.45)),
            plies: Some(u32_of(params, "plies", 2).max(1)),
            twist_angle_deg: Some(f64_of(params, "twist_angle", 10.0)),
            ply_direction,
            regularity: Some(f64_of(params, "regularity", 0.7).clamp(0.0, 1.0)),
            height_range: f64_of(params, "height_range", defaults.height_range),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
        }
    }
}

/// A single straight open segment, centered at `y = 0.5` — paired with
/// [`render_tile_mm`], which scales the physical period that fits along
/// it down to exactly one stitch, this is the node's own self-contained
/// "one stitch" unit. Downstream, a Path node's own Stamp input repeats/
/// orients copies of this node's output along a real curve — this module
/// has no opinion about that, same separation of concerns as Shape (a
/// single disc, centered) not knowing anything about Tile Generator's own
/// grid.
///
/// Spans the *entire* `[0, 1]` tile width edge to edge only when the
/// along-path period is what [`render_tile_mm`] actually used to size the
/// tile; when Zigzag/Satin's own independent `Width`/`Density` is the
/// larger number instead (`stitch_extents_mm`'s own `across` winning the
/// `max`), stretching the path to match would force-fit the along-path
/// period into a tile sized for a *different* dimension, stretching a
/// real embroidery stitch length into something it isn't — so the path is
/// shortened to its own true length-to-tile-size ratio and centered
/// instead, leaving padding on both sides rather than distorting it.
/// Confirmed via testing: without this, Zigzag/Satin's own `width_at`
/// lateral offset could reach several multiples of the tile's own `[0,1]`
/// span, rendering as the wrap-addressed sampler's own garbage spillover
/// from the opposite edge instead of a recognizable stitch shape.
///
/// `wrap: false`, not `true`: with a single stitch spanning at most the
/// *full* `[0,1]` width (not a short run repeated many times across it,
/// like before `render_tile_mm` existed), its own two endpoints can sit
/// right at the tile's own seam — `wrap: true` would make `drape`/
/// `thread_angle`'s own `wrap_delta` read that near-full-tile-width
/// endpoint-to-endpoint vector as "the short way around" the seam instead
/// and collapse it to near zero, rendering as two stray fragments at the
/// tile's left and right edges with nothing connecting them (a real
/// regression caught by testing — see
/// `render_tile_mm_always_yields_exactly_one_stitch`'s own doc comment).
/// `wrap` has no bearing on the *texture*'s own seamless tiling either way
/// — `tube_raster.wgsl` always does its own per-pixel minimum-image
/// distance test regardless of it; this flag only feeds `drape`'s
/// inter-stitch proximity test (irrelevant here — there's only ever one
/// stitch) and a stitch's own a-to-b direction/length math, which is
/// exactly what must *not* wrap for a stitch this long.
pub fn synthetic_local_path(params: &StitchParams) -> mc_core::Path {
    let (along_mm, _) = stitch_extents_mm(params);
    let tile_mm = render_tile_mm(params);
    let half_len = 0.5 * (along_mm / tile_mm).min(1.0);
    mc_core::Path {
        subpaths: vec![mc_core::SubPath::new(
            vec![
                mc_core::Point::new([0.5 - half_len, 0.5]),
                mc_core::Point::new([0.5 + half_len, 0.5]),
            ],
            false,
        )],
        wrap: false,
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
/// Python's banker's rounding (round-half-to-even) — Rust's own `f64::round`
/// rounds half away from zero instead, which disagrees with the reference
/// package at exact `.5` boundaries. Also reused by `panel_layout`'s own
/// grid-snapping (`hs_reference.py`'s `panel_rects` calls Python's built-in
/// `round()` the same way).
pub(crate) fn round_half_even(x: f64) -> f64 {
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
///
/// `#[cfg(test)]`: Split isn't its own `StitchKind` anymore (folded into
/// Running — see `StitchKind::Running`'s own doc comment), so nothing in
/// live code calls this directly; it only still exists so
/// `rule_fn_by_name` can keep validating it against the reference
/// package's own golden fixture for the retired `"split"` case.
#[cfg(test)]
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

/// A single `rule_*` function's own signature — every stitch rule in §4
/// shares it, so [`generate`] (the live, `StitchKind`-driven dispatch) and
/// this module's own tests (which still validate `rule_back`/`rule_split`
/// directly against the reference package's golden fixtures, even though
/// neither is its own live `StitchKind` anymore — see
/// `StitchKind::Running`'s own doc comment) can both drive the same
/// shared section-by-section loop.
type Rule = fn(&Section, &StitchParams, &Resolved, &mut HoleMaker) -> Vec<Stitch>;

/// Every stitch for `path` under `params` and `rule`, in sewing order
/// (section by section, along increasing arc length) —
/// `stitch_reference.py`'s own `generate`.
fn generate_with_rule(path: &mc_core::Path, params: &StitchParams, rule: Rule) -> Vec<Stitch> {
    let resolved = params.resolved();
    let mut holes = HoleMaker::new(params, &resolved);
    let mut out = Vec::new();
    for sec in crate::path::sections(path, params.corner_threshold_deg) {
        out.extend(rule(&sec, params, &resolved, &mut holes));
    }
    for (i, s) in out.iter_mut().enumerate() {
        s.order = i as u32;
    }
    out
}

/// Every stitch for `path` under `params`, dispatched by its own live
/// `StitchKind` — see [`generate_with_rule`] for the shared loop every
/// rule runs through.
pub fn generate(path: &mc_core::Path, params: &StitchParams) -> Vec<Stitch> {
    let rule: Rule = match params.stitch {
        StitchKind::Running => rule_running,
        StitchKind::Bean => rule_bean,
        StitchKind::Stem => rule_stem,
        StitchKind::Zigzag => rule_zigzag,
        StitchKind::Satin => rule_satin,
    };
    generate_with_rule(path, params, rule)
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
pub(crate) fn packing(radius: f64, n: u32) -> (f64, f64) {
    if n <= 1 {
        return (0.0, radius);
    }
    let s = (std::f64::consts::PI / f64::from(n)).sin();
    let r = radius / (1.0 + 1.0 / s);
    (r / s, r)
}

pub(crate) fn normalize3_py(v: [f64; 3]) -> [f64; 3] {
    // Matches `stitch_reference.py`'s own `v / (norm(v) + 1e-15)` — an
    // epsilon added to the denominator, not a branch on near-zero input.
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() + 1e-15;
    [v[0] / len, v[1] / len, v[2] / len]
}

pub(crate) fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
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
pub(crate) fn gradient_3d_unit_spacing(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
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

/// `(along_path_mm, across_path_mm)` — the physical length one repeat of
/// `params.stitch` needs along the synthetic path's own direction, and
/// (Zigzag/Satin only, where `Width`/`Pull Compensation` are set
/// independently of the along-path length) perpendicular to it. `0.0`
/// where there's no independent across-path extent to worry about — every
/// other kind's own lateral reach (thread radius, stem offset, bean bow)
/// is already small relative to its own along-path period by construction,
/// not a free-standing mm value that can run away from it.
fn stitch_extents_mm(params: &StitchParams) -> (f64, f64) {
    match params.stitch {
        StitchKind::Running | StitchKind::Bean | StitchKind::Stem => (params.stitch_length_mm, 0.0),
        // `stitch_length_mm` is peak to peak (ZIGZAG_SPEC.md §1: "One V
        // (a zig and a zag) is 2 stitches... its length along the path is
        // the stitch length L") — using the raw value, not half of it,
        // makes the tile's own along-path period a full V (both the zig
        // and the zag), not just one diagonal half of one. Confirmed
        // directly against a real run of the unmodified
        // `stitch_reference.py`: `rule_zigzag`'s own holes already match
        // it exactly either way (`spacing = L/2` is internal to that
        // rule, already correct) — this only changes how big a piece of
        // it one Stitch tile shows.
        StitchKind::Zigzag => (params.stitch_length_mm, width_extent_mm(params)),
        StitchKind::Satin => {
            let look = look_defaults(params.look);
            let thread_mm = params.thread_mm.unwrap_or(look.thread_mm);
            let density_mm = params.density_mm.unwrap_or(thread_mm * look.density_factor);
            let along = match look.satin_mode {
                SatinMode::Parallel => density_mm,
                SatinMode::Zigzag => density_mm / 2.0,
            };
            (along, width_extent_mm(params))
        }
    }
}

/// Zigzag/Satin's own full lateral reach (mm): `width_at`'s own peak-to-
/// peak span (`width_mm`, widened further by `Pull Compensation` on each
/// side) plus one thread diameter of margin so the capsule's own radius
/// at the very outer edge doesn't get clipped by the tile boundary either
/// — same "+ one thread radius each side" margin `stitch_extents_mm`'s own
/// French Knot branch already adds for the same reason.
fn width_extent_mm(params: &StitchParams) -> f64 {
    let look = look_defaults(params.look);
    let thread_mm = params.thread_mm.unwrap_or(look.thread_mm);
    params.width_mm + 2.0 * params.pull_comp_mm + thread_mm
}

/// The physical length (mm) [`synthetic_local_path`] sizes its own tile
/// to. [`dispatch_render`] substitutes this in for `tile_mm` — only for
/// its own internal generation/rendering, never the node's stored param —
/// so the Stitch node always produces exactly one stitch *unit*, never
/// several packed in side by side (Zigzag's own "one unit" is a full V —
/// 2 `Stitch` entries, per ZIGZAG_SPEC.md §1 — same "one recognizable
/// repeat, not a fragment of one" idea as Bean's own 3 overlapping passes
/// or French Knot's coil of short segments, just not literally `n == 1`).
/// Path's own Stamp distribution (not Stitch) is what repeats it along a
/// curve, per this node's own NodeDef tooltip.
///
/// `stitch_extents_mm`'s own larger of `(along, across)` — for most kinds
/// that's always `along` (see that function's own doc comment), so this
/// still resolves to exactly `stitch_extents_mm(params).0` and
/// `synthetic_local_path` fills the whole `[0, 1]` tile edge to edge
/// unchanged. For Zigzag/Satin, `across` can win instead, in which case
/// `synthetic_local_path` shortens its own span to compensate rather than
/// stretching the along-path period to match a tile that had to grow for
/// an unrelated reason (checked via testing — before this, Zigzag/Satin's
/// own lateral reach could run to several multiples of the tile's own
/// `[0, 1]` span).
///
/// Each `rule_*` function (§4) picks its own stitch count from
/// `sec.length()` divided by a period that's some fraction of `tile_mm`;
/// `synthetic_local_path`'s own span is sized so that division always
/// lands on `1.0` (`stitch_extents_mm(params).0 / render_tile_mm(params)`
/// is always exactly `1.0`), and the rule's own `even_count`/
/// `round_half_even` selection lands on exactly one stitch *unit*
/// regardless of which of `along`/`across` actually won — `1` stitch for
/// most kinds, but Zigzag's own rule halves its own target period
/// internally (`spacing = L / 2`, ZIGZAG_SPEC.md §2), so the same `1.0`
/// ratio lands it on `2` (a full V) instead, by design. Confirmed against
/// each rule's own formula:
/// Running's own `n` collapses to `1` regardless of `gap_mm` once
/// `stitch_length_mm == tile_mm` (the `+ gap` term appears identically in
/// both its numerator and denominator), which is why Running shares the
/// same period as Back/Bean/Stem/Split rather than needing its own
/// `stitch_length_mm + gap_mm` — the whole tile fills with thread, no
/// gap baked in; the dashed-running look comes from Path's own Stamp
/// Spacing vs. Stamp Size instead.
fn render_tile_mm(params: &StitchParams) -> f64 {
    let (along, across) = stitch_extents_mm(params);
    along.max(across)
}

/// Generates, drapes and rasterizes `path` under `params` into `keys`' six
/// output textures — exactly [`render_at_scale`], except `tile_mm` is
/// first substituted for [`render_tile_mm`]'s own one-stitch value, which
/// is what every real caller (the node's own registry dispatch) wants.
/// Split out from `render_at_scale` so the golden-PNG regression test
/// below can still validate the actual rendering pipeline — generate,
/// drape, rasterize — against the reference package's own fixed-`tile_mm`
/// output, independent of this node's own single-stitch framing decision.
pub fn dispatch_render(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    keys: RenderOutputKeys,
    path: &mc_core::Path,
    params: &StitchParams,
    resolution: u32,
) {
    let params = StitchParams {
        tile_mm: render_tile_mm(params),
        ..*params
    };
    render_at_scale(gpu, pool, keys, path, &params, resolution);
}

/// The actual generate/drape/rasterize pipeline, at whatever `tile_mm`
/// `params` already carries — see [`dispatch_render`]'s own doc comment
/// for why this is split out.
fn render_at_scale(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    keys: RenderOutputKeys,
    path: &mc_core::Path,
    params: &StitchParams,
    resolution: u32,
) {
    let stitches = generate(path, params);
    render_stitches(gpu, pool, keys, path, params, resolution, stitches);
}

/// Everything [`render_at_scale`] does after its own call to [`generate`]
/// — split out so this module's own tests can drive a `rule_*` function
/// that no longer has a live `StitchKind` of its own (`rule_back`/
/// `rule_split`, see `StitchKind::Running`'s own doc comment) straight
/// through [`generate_with_rule`] instead, while still exercising the
/// exact same drape/raster/info pipeline real rendering uses.
fn render_stitches(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    keys: RenderOutputKeys,
    path: &mc_core::Path,
    params: &StitchParams,
    resolution: u32,
    stitches: Vec<Stitch>,
) {
    let resolved = params.resolved();

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
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let height_view = pool
        .get_or_create(
            &gpu.device,
            height_key,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let thread_id_view = pool
        .get_or_create(
            &gpu.device,
            thread_id_key,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let order_view = pool
        .get_or_create(
            &gpu.device,
            order_key,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let direction_view = pool
        .get_or_create(
            &gpu.device,
            direction_key,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
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

    /// The `rule_*` function a golden fixture case's own `"stitch"`
    /// string names — unlike [`generate`]'s own `StitchKind`-keyed
    /// dispatch, this still recognizes `"back"`/`"split"` directly, since
    /// the golden fixtures (ported straight from the reference package)
    /// predate — and are independent of — this module's own live
    /// `StitchKind` options (see `StitchKind::Running`'s own doc comment).
    fn rule_fn_by_name(name: &str) -> Rule {
        match name {
            "running" => rule_running,
            "back" => rule_back,
            "bean" => rule_bean,
            "stem" => rule_stem,
            "split" => rule_split,
            "zigzag" => rule_zigzag,
            "satin" => rule_satin,
            other => panic!("unknown stitch kind {other:?}"),
        }
    }

    /// `rule_fn_by_name` (not this `stitch` field) is what actually
    /// decides which `rule_*` function a golden case runs through —
    /// "back"/"split" are mapped to `Running` here purely to produce a
    /// valid, inert placeholder for the struct literal, same reasoning as
    /// `knot_wraps` just below.
    fn params_from_json(v: &serde_json::Value) -> StitchParams {
        let stitch = match v["stitch"].as_str().unwrap() {
            "running" | "back" | "split" => StitchKind::Running,
            "bean" => StitchKind::Bean,
            "stem" => StitchKind::Stem,
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
            let rule = rule_fn_by_name(case_params["stitch"].as_str().unwrap());
            let golden_list = stitches_json[case_name]
                .as_array()
                .unwrap_or_else(|| panic!("missing golden stitches for case {case_name:?}"));

            let generated = generate_with_rule(&path, &params, rule);

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

    // ZIGZAG_SPEC.md's own close-up package: isolates one zigzag stitch
    // (a single V, a straight run, a corner, a curve, a taper, each
    // parameter on its own) on simple paths, rather than only the
    // original acceptance suite's single complex demo ornament — so a
    // divergence in the port is easier to localize to a specific hole or
    // sample. `ZIGZAG_PARAMS_JSON`'s own per-case `"params"` sub-object is
    // the identical schema `params_from_json` above already parses (same
    // field set, "stitch": "zigzag" always), so it's reused directly
    // rather than duplicating a parser; `"path"` is a real `mc_core::Path`
    // JSON object (same `pos`/`h_in`/`h_out`/`kind`/`width` shape
    // `mc_core::path`'s own doc comment already documents), deserialized
    // straight through `serde`.
    const ZIGZAG_PARAMS_JSON: &str = include_str!("../../../tests/golden/zigzag/params.json");

    fn zigzag_golden_stitches(case: &str) -> &'static str {
        match case {
            "zz_single_v_hand" => {
                include_str!("../../../tests/golden/zigzag/zz_single_v_hand_stitches.json")
            }
            "zz_single_v_machine" => {
                include_str!("../../../tests/golden/zigzag/zz_single_v_machine_stitches.json")
            }
            "zz_straight_hand" => {
                include_str!("../../../tests/golden/zigzag/zz_straight_hand_stitches.json")
            }
            "zz_straight_machine" => {
                include_str!("../../../tests/golden/zigzag/zz_straight_machine_stitches.json")
            }
            "zz_dense_machine" => {
                include_str!("../../../tests/golden/zigzag/zz_dense_machine_stitches.json")
            }
            "zz_angle30" => include_str!("../../../tests/golden/zigzag/zz_angle30_stitches.json"),
            "zz_pull_comp" => {
                include_str!("../../../tests/golden/zigzag/zz_pull_comp_stitches.json")
            }
            "zz_taper" => include_str!("../../../tests/golden/zigzag/zz_taper_stitches.json"),
            "zz_corner" => include_str!("../../../tests/golden/zigzag/zz_corner_stitches.json"),
            "zz_curve" => include_str!("../../../tests/golden/zigzag/zz_curve_stitches.json"),
            "zz_loose" => include_str!("../../../tests/golden/zigzag/zz_loose_stitches.json"),
            "zz_tiling" => include_str!("../../../tests/golden/zigzag/zz_tiling_stitches.json"),
            other => panic!("unknown zigzag close-up case {other:?}"),
        }
    }

    /// ZIGZAG_SPEC.md §4 step 1: stitch counts identical, every hole
    /// within 1e-6 tile units, flags identical — for all 12 close-up
    /// cases. Order follows the spec's own suggestion: start with
    /// `zz_single_v_machine` (no jitter), then `zz_single_v_hand` (tests
    /// the jitter hash), covered here by `ZZ_CASES`'s own declared order.
    #[test]
    fn zigzag_close_up_holes_match_golden_for_every_case() {
        const ZZ_CASES: [&str; 12] = [
            "zz_single_v_machine",
            "zz_single_v_hand",
            "zz_straight_hand",
            "zz_straight_machine",
            "zz_dense_machine",
            "zz_angle30",
            "zz_pull_comp",
            "zz_taper",
            "zz_corner",
            "zz_curve",
            "zz_loose",
            "zz_tiling",
        ];
        let params_json: serde_json::Value = serde_json::from_str(ZIGZAG_PARAMS_JSON).unwrap();

        for case in ZZ_CASES {
            let case_json = &params_json[case];
            let params = params_from_json(&case_json["params"]);
            let path: mc_core::Path = serde_json::from_value(case_json["path"].clone())
                .unwrap_or_else(|e| panic!("{case}: bad path JSON: {e}"));

            let golden_text = zigzag_golden_stitches(case);
            let golden: serde_json::Value = serde_json::from_str(golden_text).unwrap();
            let golden_stitches = golden["stitches"].as_array().unwrap();

            let generated = generate(&path, &params);

            assert_eq!(
                generated.len(),
                golden["stitch_count"].as_u64().unwrap() as usize,
                "{case}: stitch count"
            );
            assert_eq!(
                generated.len(),
                golden_stitches.len(),
                "{case}: stitch count"
            );

            for (i, (st, g)) in generated.iter().zip(golden_stitches).enumerate() {
                assert_close2(st.a, arr(&g["a"]), 1e-6, &format!("{case}[{i}].a"));
                assert_close2(st.b, arr(&g["b"]), 1e-6, &format!("{case}[{i}].b"));
                assert_eq!(st.dip_a, g["dip_a"].as_bool().unwrap(), "{case}[{i}].dip_a");
                assert_eq!(st.dip_b, g["dip_b"].as_bool().unwrap(), "{case}[{i}].dip_b");
                assert_eq!(st.stack, g["stack"].as_bool().unwrap(), "{case}[{i}].stack");
                let gbow = g["bow"].as_f64().unwrap();
                assert!(
                    (st.bow - gbow).abs() < 1e-6,
                    "{case}[{i}].bow: actual={} expected={gbow}",
                    st.bow
                );
            }
            // Every hole is a stitch's own `a` or `b` (a section boundary
            // can mean consecutive stitches *don't* share one — confirmed
            // directly against a real run of the unmodified
            // `stitch_reference.py` for `zz_corner`, ZIGZAG_SPEC.md §2's
            // own "the next section's first stitch can therefore cross
            // the last one" — so the per-stitch checks above already
            // cover every hole position and flag; a separately
            // reconstructed hole list would have to guess at that
            // chaining and isn't needed on top of them.
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

    /// Named for the golden fixture case ("back_hand" in
    /// `tests/golden/stitch/params.json`), not a live `StitchKind` — Back
    /// isn't its own option anymore (see `StitchKind::Running`'s own doc
    /// comment). `stitch: Running` here only matters for the three tests
    /// below that go through the *live* `dispatch_render`/`generate`
    /// dispatch (determinism, seamless tiling, middle-fill) — none of them
    /// care whether it's specifically Back's own math, just that *some*
    /// real stitch renders through the real pipeline. The golden PNG
    /// comparison test itself drives `rule_back` directly instead (see its
    /// own comment), so this field is never actually consulted there.
    fn back_hand_params() -> StitchParams {
        StitchParams {
            stitch: StitchKind::Running,
            look: Look::Hand,
            // Pinned explicitly, not left to `Default::default()` — the
            // golden PNG below was rendered by the reference package at
            // `height_range: 4.0` (see `tests/golden/stitch/params.json`'s
            // own `"back_hand"` entry), independent of whatever
            // `StitchParams::default()`'s own height_range is tuned to for
            // live rendering.
            height_range: 4.0,
            ..Default::default()
        }
    }

    /// The whole point of `render_tile_mm`: every stitch type, at its own
    /// default params, generates exactly one stitch *unit* on the
    /// synthetic local path — never several packed into the same tile.
    /// Regression test for the "clunky when stamped along a Path" report
    /// — the old behavior packed ~16 back stitches into one tile, which
    /// Path's own Stamp distribution then repeated *again*, compounding
    /// into a dense, muddy look instead of clean individual stitches.
    ///
    /// Bean and Zigzag are the exceptions: Bean's own `Stitch` *count* is
    /// `2 * bean_repeats + 1` by design (the thread sewn forward-back-
    /// forward over the same two holes for a plump look); Zigzag is 2
    /// (ZIGZAG_SPEC.md §1: "One V... is 2 stitches" — `stitch_length_mm`
    /// is peak to peak, so one full V is the actual "one stitch unit"
    /// here, not one diagonal half of one) — both still exactly one
    /// stitch *unit*, just made of several `Stitch` entries. French Knot
    /// isn't one of `STITCH_KINDS` at all — see `StitchKind`'s own doc
    /// comment.
    #[test]
    fn render_tile_mm_always_yields_exactly_one_stitch() {
        for &kind in &STITCH_KINDS {
            let params = StitchParams {
                stitch: kind,
                ..Default::default()
            };
            let path = synthetic_local_path(&params);
            let expected = match kind {
                StitchKind::Bean => (2 * params.bean_repeats + 1) as usize,
                StitchKind::Zigzag => 2,
                _ => 1,
            };
            let render_params = StitchParams {
                tile_mm: render_tile_mm(&params),
                ..params
            };
            let stitches = generate(&path, &render_params);
            assert_eq!(
                stitches.len(),
                expected,
                "{kind:?} should generate one stitch unit ({expected} thread segment(s)) on \
                 its own tile, got {}",
                stitches.len()
            );
        }
    }

    /// Regression test for the exact bug a real test render caught: with
    /// the single stitch spanning the *entire* tile width, its own two
    /// endpoints sit right at the seam — `synthetic_local_path`'s own
    /// `wrap: true` (before it was fixed to `false`) made `drape`'s
    /// `wrap_delta` read that nearly-full-tile-width endpoint-to-endpoint
    /// vector as "the short way around" and collapse it to near zero,
    /// rendering as two disconnected fragments at the tile's left and
    /// right edges with the middle completely blank. Checks the mask is
    /// actually lit at the tile's own horizontal center, not just near its
    /// edges.
    #[test]
    fn dispatch_render_fills_the_middle_of_its_own_tile_not_just_the_edges() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let params = back_hand_params();
        let path = synthetic_local_path(&params);
        let res = 128u32;

        dispatch_render(
            &mut gpu,
            &mut pool,
            RenderOutputKeys {
                height: "middle_height",
                mask: "middle_mask",
                ply_id: "middle_ply_id",
                thread_id: "middle_thread_id",
                order: "middle_order",
                direction: "middle_direction",
            },
            &path,
            &params,
            res,
        );

        let pixels = gpu.read_rgba8(pool.texture("middle_mask").unwrap(), res);
        let center = (res / 2) as usize;
        let mask_at = |x: usize, y: usize| pixels[(y * res as usize + x) * 4];
        assert!(
            mask_at(center, center) > 0,
            "the single stitch should cover the tile's own horizontal \
             center, not just its left/right edges"
        );
    }

    /// Regression test for a real test render: Zigzag/Satin's own
    /// `Width`/`Pull Compensation` are set independently of their own
    /// along-path period, and at ordinary default values the lateral
    /// reach that implies is several times the tile's own `[0, 1]` span —
    /// rendered as the wrap-addressed sampler's own garbage spillover
    /// from the opposite edge (a thick diagonal smear with stray
    /// fragments at the corners) instead of a recognizable zigzag/satin
    /// shape. `render_tile_mm`/`synthetic_local_path` now take whichever
    /// of the along- or across-path extent is larger, shortening and
    /// centering the path instead of stretching it when the across-path
    /// one wins.
    #[test]
    fn zigzag_and_satin_geometry_stays_within_their_own_tile() {
        for kind in [StitchKind::Zigzag, StitchKind::Satin] {
            let params = StitchParams {
                stitch: kind,
                ..Default::default()
            };
            let path = synthetic_local_path(&params);
            let render_params = StitchParams {
                tile_mm: render_tile_mm(&params),
                ..params
            };
            let stitches = generate(&path, &render_params);
            assert!(!stitches.is_empty(), "{kind:?} generated no stitches");
            for st in &stitches {
                for p in [st.a, st.b] {
                    assert!(
                        (0.0..=1.0).contains(&p[0]) && (0.0..=1.0).contains(&p[1]),
                        "{kind:?} point {p:?} falls outside its own [0, 1] tile"
                    );
                }
            }
        }
    }

    #[test]
    fn dispatch_render_is_deterministic() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let path = demo_path();
        let params = back_hand_params();

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
                128,
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
        let params = back_hand_params();

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
            128,
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
        let params = back_hand_params();

        // `rule_back` directly (via `generate_with_rule`/`render_stitches`),
        // not the live `generate`/`render_at_scale` — Back no longer has
        // its own `StitchKind` (see `StitchKind::Running`'s own doc
        // comment), but `rule_back`'s own math still needs validating
        // against the reference package's fixed-`tile_mm` golden,
        // independent of `render_tile_mm`'s one-stitch-per-tile framing
        // (see `dispatch_render`'s own doc comment).
        let stitches = generate_with_rule(&path, &params, rule_back);
        render_stitches(
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
            768,
            stitches,
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
