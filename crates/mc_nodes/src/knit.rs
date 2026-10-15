// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Knit.
//!
//! All 6 steps of KNIT_NODE_SPEC.md's staged build (private notes repo) are
//! done. Step 1: the course centerline curve builder, CPU-only, ported 1:1
//! from `knit_reference.py`'s `resolve()`/`course_centerline()` — the
//! owner's ground-truth renderer, derived from Keenan Crane's "A Simple
//! Parametric Model of Plain-Knit Yarns" (CMU, 2023) and her textile
//! reference book's knitting chapter (pp.138-161) — verified against its
//! own golden centerline samples (`tests/golden/knit/curves.json`, max
//! absolute error `1e-5`). Step 2: [`dispatch`] turns those curves into
//! capsule segments and hands them to [`crate::tube_raster`], matched
//! against the `jersey_front`/`jersey_back` goldens. Step 3: the chart/
//! rib/purl/interlock rules already live in `resolve()`/`course_centerline()`
//! from Step 1 — this step proved the *rasterizer* handles the more complex
//! geometry they produce, matched against `rib_1x1`, `rib_2x2`, `purl`,
//! `interlock` and `jersey_tight`. Step 4: [`course_tubes`] adds the
//! rotation-minimizing-frame ply math for `plies > 1`, matched against
//! `jersey_2ply`. Step 5: [`dispatch_stitch_info`] adds Stitch ID (same
//! `hash01` as the reference, bit for bit) and Stitch Type. Step 6: wired
//! into `mc_nodes::defs`/`registry` and `mc_app::params_panel` as a real,
//! selectable node.
//!
//! One deliberate deviation from KNIT_NODE_SPEC.md §7: it describes a
//! Colour input/output pair (sampled once per stitch, for a jacquard/
//! intarsia look), but the owner cut that from the shipped node as
//! redundant with Uniform Color + Blend + a generic Color output already
//! covering the same ground — not worth a dedicated pin on every Knit
//! node. Per rule 1 ("the code wins, do not invent deviations") this is
//! called out explicitly rather than silently diverging, since it's a
//! product decision, not a correctness fix.
//!
//! A weft-knit fabric is COURSES (horizontal rows) of continuous yarn,
//! each course forming one loop per WALE (vertical column/needle). A
//! course's centerline follows Crane's closed-form curve:
//!
//! ```text
//! gamma(t) = (t + a*sin(2t), h*cos(t), d*cos(2t))
//! ```
//!
//! One period of `t` (`2*pi`) is one stitch. Knit vs purl is only the
//! *sign* of the z term (`sign` in [`course_centerline`]) — the back of a
//! knit stitch is a purl stitch (reversing `side` swaps every K and P in
//! the chart, which is exactly that). Every phase-1 fabric (jersey, the
//! ribs, purl/links-links, interlock) is a chart of K/P plus a handful of
//! per-wale/per-course adjustments ([`resolve`]'s own doc comment covers
//! exactly which) — not faked with noise, stripes, or a tiled bitmap.
//!
//! Deliberately **not** reusing Thread's per-column helix math: loops
//! double back in x (the roundness term overshoots and backtracks), so
//! "one centerline per column" doesn't exist here the way it does for a
//! straight twisted strand. `packing()` (ply packing) *is* shared with
//! Thread — identical formula, reused directly from
//! [`crate::thread_shape::packing`] rather than duplicated — but the
//! curve and its rasterization are their own thing (later steps).
//!
//! Internally f64, not f32: this step's whole job is numerically matching
//! a float64 numpy reference to `1e-5`, and the CPU cost here is trivial
//! (wales x courses x ~96 samples, not a per-pixel GPU path) — downcasting
//! to f32 is deferred to whichever later step actually uploads this to the
//! GPU.

use std::f64::consts::PI;

/// Which K/P pattern to knit, before any custom chart override
/// (KNIT_NODE_SPEC.md §4). Each preset also carries the book-specific
/// adjustments (loop height for links-links fabric compressing
/// lengthwise, wale stagger for interlock's alternating-course offset,
/// yarn thickness for interlock's thinner face yarn) that a custom chart
/// override does *not* replace — a custom chart only swaps which K/P
/// pattern is knitted, not the preset's own physical modifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnitPreset {
    Jersey,
    ReverseJersey,
    Purl,
    Rib1x1,
    Rib2x2,
    Rib2x1,
    Rib6x3,
    Interlock,
    /// KNIT_NODE_SPEC.md §4's own "custom" row: editing the Chart text
    /// switches the UI's Preset dropdown to this — no book-specific extra
    /// modifiers of its own (every one of `loop_height_override`/
    /// `stagger`/`thickness_override` below falls through to the same
    /// neutral default Jersey uses), since a hand-typed chart's rib/
    /// links-links behavior already comes from `resolve()`'s own
    /// chart-content detection (`is_rib`/`is_links`), not from picking a
    /// specific named preset.
    Custom,
}

impl KnitPreset {
    pub const NAMES: [&'static str; 9] = [
        "Jersey",
        "Reverse Jersey",
        "Purl",
        "1x1 Rib",
        "2x2 Rib",
        "2x1 Rib",
        "6x3 Rib",
        "Interlock",
        "Custom",
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Jersey" => Some(KnitPreset::Jersey),
            "Reverse Jersey" => Some(KnitPreset::ReverseJersey),
            "Purl" => Some(KnitPreset::Purl),
            "1x1 Rib" => Some(KnitPreset::Rib1x1),
            "2x2 Rib" => Some(KnitPreset::Rib2x2),
            "2x1 Rib" => Some(KnitPreset::Rib2x1),
            "6x3 Rib" => Some(KnitPreset::Rib6x3),
            "Interlock" => Some(KnitPreset::Interlock),
            "Custom" => Some(KnitPreset::Custom),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            KnitPreset::Jersey => "Jersey",
            KnitPreset::ReverseJersey => "Reverse Jersey",
            KnitPreset::Purl => "Purl",
            KnitPreset::Rib1x1 => "1x1 Rib",
            KnitPreset::Rib2x2 => "2x2 Rib",
            KnitPreset::Rib2x1 => "2x1 Rib",
            KnitPreset::Rib6x3 => "6x3 Rib",
            KnitPreset::Interlock => "Interlock",
            KnitPreset::Custom => "Custom",
        }
    }

    /// Wales for this preset's own clean single repeat of its chart — the
    /// same count used by its golden test case where one exists
    /// (`jersey`/`purl`/`reverse_jersey`/`custom`: 8, `rib_1x1`: 10,
    /// `rib_2x2`: 12, `interlock`: 10), or, for the two presets
    /// `tests/golden/knit/` doesn't cover (`rib_2x1`, `rib_6x3`), the
    /// smallest multiple of the chart's own width. Used by the UI's Preset
    /// picker (`mc_app::params_panel::knit_preset_picker`) to reset Wales
    /// — along with every other construction slider, back to
    /// [`KnitParams::default`]'s own values — to this preset's own
    /// canonical look, regardless of whatever Wales previously held
    /// (including a prior "randomize everything" roll).
    pub fn canonical_wales(self) -> u32 {
        match self {
            KnitPreset::Jersey
            | KnitPreset::ReverseJersey
            | KnitPreset::Purl
            | KnitPreset::Custom => 8,
            KnitPreset::Rib1x1 => 10,
            KnitPreset::Rib2x2 => 12,
            KnitPreset::Rib2x1 => 9,
            KnitPreset::Rib6x3 => 9,
            KnitPreset::Interlock => 10,
        }
    }

    /// Never actually read for `Custom` in practice — a `KnitParams.chart`
    /// of `None` with `preset: Custom` shouldn't normally occur (the UI
    /// always sets Chart text together with switching to Custom), but
    /// `resolve()` still needs a well-formed fallback rather than an empty
    /// slice if it ever does.
    fn default_chart_rows(self) -> &'static [&'static str] {
        match self {
            KnitPreset::Jersey => &["K"],
            KnitPreset::ReverseJersey => &["P"],
            KnitPreset::Purl => &["K", "P"],
            KnitPreset::Rib1x1 => &["KP"],
            KnitPreset::Rib2x2 => &["KKPP"],
            KnitPreset::Rib2x1 => &["KKP"],
            KnitPreset::Rib6x3 => &["KKKKKKPPP"],
            KnitPreset::Interlock => &["K"],
            KnitPreset::Custom => &["K"],
        }
    }

    /// Purl/links-links fabric compresses lengthwise (book: alternating
    /// knit/purl courses), so its loops are taller than the course
    /// spacing compared to jersey's default.
    fn loop_height_override(self) -> Option<f64> {
        match self {
            KnitPreset::Purl => Some(1.35),
            _ => None,
        }
    }

    /// Interlock's odd wales are offset half a course down (two 1x1 ribs
    /// knitted in alternating courses, approximated here as the face only
    /// — see the module doc comment).
    fn stagger(self) -> f64 {
        match self {
            KnitPreset::Interlock => 0.5,
            _ => 0.0,
        }
    }

    fn thickness_override(self) -> Option<f64> {
        match self {
            KnitPreset::Interlock => Some(0.2),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Front,
    Back,
}

impl Side {
    pub const NAMES: [&'static str; 2] = ["Front", "Back"];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Front" => Some(Side::Front),
            "Back" => Some(Side::Back),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Side::Front => "Front",
            Side::Back => "Back",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tension {
    Loose,
    Normal,
    Tight,
}

impl Tension {
    pub const NAMES: [&'static str; 3] = ["Loose", "Normal", "Tight"];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Loose" => Some(Tension::Loose),
            "Normal" => Some(Tension::Normal),
            "Tight" => Some(Tension::Tight),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Tension::Loose => "Loose",
            Tension::Normal => "Normal",
            Tension::Tight => "Tight",
        }
    }

    fn thickness(self) -> f64 {
        match self {
            Tension::Loose => 0.15,
            Tension::Normal => 0.178,
            Tension::Tight => 0.215,
        }
    }

    fn loop_height(self) -> f64 {
        match self {
            Tension::Loose => 0.95,
            Tension::Normal => 0.889,
            Tension::Tight => 0.84,
        }
    }
}

/// Course spacing / wale width: Crane's own numbers (rows 4.5 apart, a
/// stitch 2*pi wide).
const STITCH_ASPECT: f64 = 4.5 / (2.0 * PI);

pub struct KnitParams {
    pub preset: KnitPreset,
    /// Custom chart rows (K/P only), overriding the preset's own chart —
    /// the preset's physical modifiers (loop height, stagger, thickness)
    /// still apply regardless. `None` uses the preset's own chart as-is.
    pub chart: Option<Vec<String>>,
    pub side: Side,
    pub wales: u32,
    /// `None` = Auto (keeps the natural stitch aspect ratio; see
    /// [`resolve`]).
    pub courses: Option<u32>,
    pub roundness: f64,
    pub loop_height: Option<f64>,
    pub depth: f64,
    pub tension: Tension,
    pub thickness: Option<f64>,
    pub plies: u32,
    pub ply_direction: crate::thread_shape::TwistDirection,
    pub twist_angle: f64,
    pub rib_depth: f64,
    pub rib_compaction: f64,
    pub seed: u32,
}

impl Default for KnitParams {
    fn default() -> Self {
        Self {
            preset: KnitPreset::Jersey,
            chart: None,
            side: Side::Front,
            wales: 8,
            courses: None,
            roundness: 1.5,
            loop_height: None,
            depth: 1.25,
            tension: Tension::Normal,
            thickness: None,
            plies: 1,
            ply_direction: crate::thread_shape::TwistDirection::S,
            twist_angle: 30.0,
            rib_depth: 1.5,
            rib_compaction: 0.5,
            seed: 1,
        }
    }
}

impl KnitParams {
    /// Builds a `KnitParams` from a graph node's param map
    /// (`mc_nodes::defs`'s Knit `NodeDef`). Two UI-only conventions, since
    /// the graph's param map has no `Option` of its own:
    ///
    /// - An empty Chart string means "use the preset's own chart" (`None`);
    ///   non-empty text is split into rows on `/` (KNIT_NODE_SPEC.md §7:
    ///   "text, K/P, rows split by /"). A malformed chart (no rows, or rows
    ///   of inconsistent length — `resolve()`'s own `assert!` would panic
    ///   on that, and this function runs on whatever text a user actually
    ///   typed) falls back to `None` rather than passing it through, per
    ///   CLAUDE.md §12's "no `unwrap`/`expect` in a path a user can
    ///   trigger."
    /// - Thickness and Loop Height use `0.0` (below either's real minimum)
    ///   as "not overridden, derive from Tension" — the alternative, a
    ///   separate Bool per field, didn't have a precedent elsewhere in this
    ///   codebase worth introducing just for these two.
    ///
    /// Auto Courses is its own Bool param (`auto_courses`) rather than a
    /// sentinel on Courses, since `0` is already a meaningless value for a
    /// fabric to have no rows at all, unlike Thickness/Loop Height where
    /// `0.0` is merely *physically* meaningless (not a valid input either
    /// way) and so was free to repurpose.
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let float = |name: &str, default: f64| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .map(|v| v as f64)
                .unwrap_or(default)
        };
        let int = |name: &str, default: i32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_int)
                .unwrap_or(default)
        };
        let enum_name = |name: &str| -> Option<&str> {
            match params.get(name) {
                Some(mc_core::ParamValue::Enum(s)) => Some(s.as_str()),
                _ => None,
            }
        };

        let preset = enum_name("preset")
            .and_then(KnitPreset::from_name)
            .unwrap_or(defaults.preset);

        let chart = params
            .get("chart")
            .and_then(mc_core::ParamValue::as_text)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .and_then(|text| {
                let rows: Vec<String> = text
                    .split('/')
                    .map(|row| row.trim().to_uppercase())
                    .collect();
                let width = rows.first().map(|r| r.len()).unwrap_or(0);
                let well_formed =
                    width > 0 && rows.iter().all(|row| !row.is_empty() && row.len() == width);
                well_formed.then_some(rows)
            });

        let side = enum_name("side")
            .and_then(Side::from_name)
            .unwrap_or(defaults.side);

        let wales = (int("wales", defaults.wales as i32).max(1)) as u32;

        let auto_courses = params
            .get("auto_courses")
            .and_then(mc_core::ParamValue::as_bool)
            .unwrap_or(true);
        let courses = if auto_courses {
            None
        } else {
            Some(int("courses", 11).clamp(2, 96) as u32)
        };

        let roundness = float("roundness", defaults.roundness);

        let loop_height_raw = float("loop_height", 0.0);
        let loop_height = (loop_height_raw > 0.0).then_some(loop_height_raw);

        let depth = float("depth", defaults.depth);

        let tension = enum_name("tension")
            .and_then(Tension::from_name)
            .unwrap_or(defaults.tension);

        let thickness_raw = float("thickness", 0.0);
        let thickness = (thickness_raw > 0.0).then_some(thickness_raw);

        let plies = (int("plies", defaults.plies as i32).clamp(1, 4)) as u32;

        let ply_direction = enum_name("ply_direction")
            .and_then(crate::thread_shape::TwistDirection::from_name)
            .unwrap_or(defaults.ply_direction);

        let twist_angle = float("twist_angle", defaults.twist_angle);
        let rib_depth = float("rib_depth", defaults.rib_depth);
        let rib_compaction = float("rib_compaction", defaults.rib_compaction);

        let seed = params
            .get("seed")
            .and_then(mc_core::ParamValue::as_seed)
            .unwrap_or(defaults.seed);

        Self {
            preset,
            chart,
            side,
            wales,
            courses,
            roundness,
            loop_height,
            depth,
            tension,
            thickness,
            plies,
            ply_direction,
            twist_angle,
            rib_depth,
            rib_compaction,
            seed,
        }
    }
}

/// Python's `round()` is round-half-to-even (banker's rounding), unlike
/// Rust's `f64::round()` (round-half-away-from-zero) — matters for the
/// wale/course snapping below, which the spec's own acceptance list
/// checks exact values for ("Wales 7 with chart KKPP gives 8"). None of
/// this step's own golden cases land on an exact `.5` boundary, but
/// getting this wrong would quietly diverge from the reference for wale/
/// course counts the golden tests don't happen to cover.
fn round_half_even(x: f64) -> i64 {
    let floor = x.floor();
    let diff = x - floor;
    let floor_i = floor as i64;
    if diff < 0.5 {
        floor_i
    } else if diff > 0.5 {
        floor_i + 1
    } else if floor_i % 2 == 0 {
        floor_i
    } else {
        floor_i + 1
    }
}

/// Everything [`course_centerline`] needs, turned from artist-facing
/// params into concrete tile-unit geometry (`knit_reference.py`'s own
/// `resolve()`). Rules applied here, each from the owner's textile
/// reference book (KNIT_NODE_SPEC.md §4):
///
/// - **Rib rule**: a wale that's entirely knit or entirely purl across
///   every course (mixed wales — purl/links-links — are left alone) gets
///   narrowed (`1 - rib_compaction`) and pushed back (`+-0.5 * rib_depth`
///   yarn radii) if purl, or pushed forward if knit — only when the chart
///   actually has *both* all-knit and all-purl wales (a rib), not for an
///   all-knit or all-purl fabric. Widths are renormalized afterward so
///   the wales always fill the tile exactly.
/// - **Links-links rule**: symmetric, but per *course* instead of per
///   wale — an all-purl course comes forward, an all-knit course between
///   purl courses goes back, only when the chart has both (purl/
///   links-links specifically, not plain jersey or reverse jersey).
/// - **Auto courses**: keeps a full-width wale's natural stitch aspect
///   ratio (`STITCH_ASPECT`), scaled by how much taller the loop is than
///   jersey's default — so changing Loop Height doesn't silently distort
///   the stitch shape when Courses is left on Auto.
pub struct Geometry {
    /// Resolved chart (side-swapped already if `side` is Back), true = K.
    pub chart: Vec<Vec<bool>>,
    pub chart_width: usize,
    pub chart_height: usize,
    pub wales: u32,
    pub courses: u32,
    /// `c` in the spec/script: course spacing, tile units (`1 / courses`).
    pub course_spacing: f64,
    /// `R`: yarn radius, tile units.
    pub yarn_radius: f64,
    /// `h`: loop half-height, tile units.
    pub loop_half_height: f64,
    /// `d`: in/out depth, tile units.
    pub depth: f64,
    /// `a`: roundness, dimensionless (passed straight through from the
    /// artist param, not derived).
    pub roundness: f64,
    /// Per-wale width as a fraction of the tile (sums to 1), length
    /// `wales`.
    pub widths: Vec<f64>,
    /// Per-wale x-center (cumulative width), length `wales`.
    pub centers: Vec<f64>,
    /// Per-wale z-offset from the rib rule, tile units, length `wales`.
    pub zoff: Vec<f64>,
    /// Per-wale y-offset from interlock-style staggering, tile units,
    /// length `wales`.
    pub stagger: Vec<f64>,
    /// Per-course z-offset from the links-links rule, tile units, length
    /// `courses`.
    pub czoff: Vec<f64>,
    /// Fixed height-normalization constant (never derived from the
    /// rendered image's own min/max — see the spec's "Do not" list).
    pub zmax: f64,
    /// Whether the rib rule actually fired (the chart has both all-knit
    /// and all-purl wales) — exposed mainly so the UI can show Rib
    /// Depth/Rib Compaction only when they'd do anything, for *any* chart
    /// (including a hand-typed custom one), not just by checking which
    /// named preset is selected.
    pub is_rib: bool,
    /// Whether the links-links rule fired (the chart has both all-knit and
    /// all-purl *courses*) — same reasoning as `is_rib`.
    pub is_links: bool,
}

pub fn resolve(p: &KnitParams) -> Geometry {
    let raw_rows: Vec<String> = match &p.chart {
        Some(rows) => rows.iter().map(|r| r.to_uppercase()).collect(),
        None => p
            .preset
            .default_chart_rows()
            .iter()
            .map(|r| r.to_uppercase())
            .collect(),
    };
    let mut chart: Vec<Vec<bool>> = raw_rows
        .iter()
        .map(|row| row.chars().map(|c| c == 'K').collect::<Vec<bool>>())
        .collect();
    if p.side == Side::Back {
        for row in chart.iter_mut() {
            for cell in row.iter_mut() {
                *cell = !*cell;
            }
        }
    }
    let chart_width = chart[0].len();
    let chart_height = chart.len();
    assert!(
        chart.iter().all(|row| row.len() == chart_width),
        "chart rows must have equal length"
    );

    let wales = chart_width
        .max((round_half_even(p.wales as f64 / chart_width as f64) as usize) * chart_width)
        as u32;
    let w = wales as usize;

    let purl_wale: Vec<bool> = (0..w)
        .map(|j| (0..chart_height).all(|i| !chart[i][j % chart_width]))
        .collect();
    let knit_wale: Vec<bool> = (0..w)
        .map(|j| (0..chart_height).all(|i| chart[i][j % chart_width]))
        .collect();
    let is_rib = purl_wale.iter().any(|&v| v) && knit_wale.iter().any(|&v| v);

    let mut widths = vec![1.0f64; w];
    let mut zoff = vec![0.0f64; w];
    if is_rib {
        for j in 0..w {
            if purl_wale[j] {
                widths[j] = 1.0 - p.rib_compaction;
                zoff[j] = -0.5 * p.rib_depth;
            } else {
                zoff[j] = 0.5 * p.rib_depth;
            }
        }
    }
    let width_sum: f64 = widths.iter().sum();
    for width in widths.iter_mut() {
        *width /= width_sum;
    }

    let loop_height = p
        .loop_height
        .or(p.preset.loop_height_override())
        .unwrap_or(p.tension.loop_height());

    let target = match p.courses {
        Some(c) => c as f64,
        None => {
            let sw = widths.iter().cloned().fold(f64::MIN, f64::max);
            1.0 / (STITCH_ASPECT * sw) * (loop_height / Tension::Normal.loop_height())
        }
    };
    let courses = chart_height
        .max((round_half_even(target / chart_height as f64) as usize) * chart_height)
        as u32;
    let c = courses as usize;
    let course_spacing = 1.0 / courses as f64;

    let yarn_radius = p
        .thickness
        .or(p.preset.thickness_override())
        .unwrap_or(p.tension.thickness())
        * course_spacing;
    let h = loop_height * course_spacing;
    let d = p.depth * yarn_radius;
    for z in zoff.iter_mut() {
        *z *= yarn_radius;
    }

    let purl_course: Vec<bool> = (0..chart_height)
        .map(|i| chart[i].iter().all(|&k| !k))
        .collect();
    let knit_course: Vec<bool> = (0..chart_height)
        .map(|i| chart[i].iter().all(|&k| k))
        .collect();
    let is_links = purl_course.iter().any(|&v| v) && knit_course.iter().any(|&v| v);

    let czoff: Vec<f64> = (0..c)
        .map(|i| {
            if !is_links {
                return 0.0;
            }
            let row = i % chart_height;
            let sign = if purl_course[row] { 0.5 } else { -0.5 };
            sign * p.rib_depth * yarn_radius
        })
        .collect();

    let mut centers = vec![0.0f64; w];
    let mut cumulative = 0.0f64;
    for j in 0..w {
        cumulative += widths[j];
        centers[j] = cumulative - widths[j] / 2.0;
    }

    let stagger_amount = p.preset.stagger();
    let stagger: Vec<f64> = (0..w)
        .map(|j| stagger_amount * course_spacing * ((j % 2) as f64))
        .collect();

    let zmax = d
        + 0.5 * p.rib_depth * yarn_radius * (if is_rib || is_links { 1.0 } else { 0.0 })
        + yarn_radius;

    Geometry {
        chart,
        chart_width,
        chart_height,
        wales,
        courses,
        course_spacing,
        yarn_radius,
        loop_half_height: h,
        depth: d,
        roundness: p.roundness,
        widths,
        centers,
        zoff,
        stagger,
        czoff,
        zmax,
        is_rib,
        is_links,
    }
}

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Radians of `t` over which a knit/purl sign change (and the stagger/
/// zoff that go with it) blends smoothly across a wale boundary, rather
/// than snapping — physically, the yarn passing through the fabric
/// between two needles (KNIT_NODE_SPEC.md §4).
const BLEND: f64 = 0.6;

/// One course's yarn centerline, sampled densely (`samples_per_stitch`
/// per stitch). `x` is wrapped to `[0, 1)`; `y`/`z` are not (tiling wraps
/// them later, once this feeds a rasterizer — not this step's concern).
pub struct CourseCenterline {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    /// Which wale each sample belongs to.
    pub wale: Vec<u32>,
}

pub fn course_centerline(g: &Geometry, course: u32, samples_per_stitch: u32) -> CourseCenterline {
    let w = g.wales as usize;
    let row = &g.chart[(course as usize) % g.chart_height];
    let sign: Vec<f64> = (0..w)
        .map(|j| if row[j % g.chart_width] { 1.0 } else { -1.0 })
        .collect();

    let n = w as u32 * samples_per_stitch;
    let mut xs = Vec::with_capacity(n as usize);
    let mut ys = Vec::with_capacity(n as usize);
    let mut zs = Vec::with_capacity(n as usize);
    let mut wales = Vec::with_capacity(n as usize);

    let czoff_course = g.czoff[course as usize];

    for i in 0..n {
        let t = (i as f64 + 0.5) / samples_per_stitch as f64 * 2.0 * PI - PI;
        let j = ((t + PI) / (2.0 * PI)).floor() as i64;
        let tau = t - 2.0 * PI * j as f64;
        let jw = j.rem_euclid(w as i64) as usize;

        let x_raw =
            g.centers[jw] + g.widths[jw] * (tau + g.roundness * (2.0 * tau).sin()) / (2.0 * PI);

        let b = ((t - PI) / (2.0 * PI) + 0.5).floor() as i64;
        let tb = 2.0 * PI * b as f64 + PI;
        let s = smoothstep(-BLEND, BLEND, t - tb);
        let b_idx = b.rem_euclid(w as i64) as usize;
        let b1_idx = (b + 1).rem_euclid(w as i64) as usize;
        let blend = |v: &[f64]| v[b_idx] * (1.0 - s) + v[b1_idx] * s;

        let sgn = blend(&sign);
        let zoff = blend(&g.zoff);
        let stag = blend(&g.stagger);

        let y = (course as f64 + 0.5) * g.course_spacing + stag - g.loop_half_height * tau.cos();
        let z = zoff + czoff_course - sgn * g.depth * (2.0 * tau).cos();

        xs.push(x_raw.rem_euclid(1.0));
        ys.push(y);
        zs.push(z);
        wales.push(jw as u32);
    }

    CourseCenterline {
        x: xs,
        y: ys,
        z: zs,
        wale: wales,
    }
}

/// `knit_reference.py`'s `render()`: dense enough that neighboring samples
/// are under a pixel apart (`sps = max(96, ceil(6 * resolution / wales))`),
/// so the capsule chain [`dispatch`] builds from them is visually
/// indistinguishable from the reference's own raw point-sphere splatting.
fn samples_per_stitch(resolution: u32, wales: u32) -> u32 {
    (6 * resolution).div_ceil(wales).max(96)
}

/// Where [`dispatch`] stores every output. `height`, `mask`, `ply_id` come
/// from [`crate::tube_raster::dispatch`] directly; `stitch_id`,
/// `stitch_type` come from the second (`knit_stitch_info.wgsl`) pass — see
/// [`dispatch`]'s own doc comment. No Colour output: KNIT_NODE_SPEC.md §7
/// has one (per-stitch pixelated sampling of an input texture), but the
/// owner cut it from the node as redundant with Uniform Color + Blend + a
/// generic Color output already covering the same ground.
pub struct KnitOutputKeys<'a> {
    pub height: &'a str,
    pub mask: &'a str,
    pub ply_id: &'a str,
    pub stitch_id: &'a str,
    pub stitch_type: &'a str,
}

// --- Step 4: plies (rotation-minimizing frame) -----------------------

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    if n > 1e-15 {
        scale(a, 1.0 / n)
    } else {
        a
    }
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    dot(sub(a, b), sub(a, b)).sqrt()
}

/// `numpy.unwrap`, generalized to an arbitrary period: corrects each
/// consecutive jump bigger than half a period by folding it back by whole
/// periods, so a sequence that's individually wrapped to `[0, period)` (as
/// [`course_centerline`]'s own `x` is) becomes continuous again — needed
/// before the curve goes into [`rotation_minimizing_frame`] or an arc-length
/// sum, neither of which can tolerate a false ~1-period jump where the
/// curve happens to cross the tile edge.
fn unwrap_period(values: &[f64], period: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(values.len());
    out.push(values[0]);
    let mut accum = values[0];
    for i in 1..values.len() {
        let raw_diff = values[i] - values[i - 1];
        let diff = raw_diff - period * (raw_diff / period).round();
        accum += diff;
        out.push(accum);
    }
    out
}

/// Periodic rotation-minimizing frame along a closed polyline `p` (double
/// reflection, Wang et al. 2008) — `knit_reference.py`'s own
/// `rotation_minimizing_frame()`, ported as literally as possible rather
/// than "improved": `T[i]` is a central difference wrapped via a plain
/// cyclic roll, exactly like `numpy.roll`, which means `T[0]` and
/// `T[n-1]` are computed from `p`'s raw first/last points with **no**
/// re-added `+1` tile-width correction — unlike the arc-length `seg` array
/// in [`course_tubes`], which explicitly does correct its own last entry.
/// That asymmetry looks like an oversight in the reference script, but
/// rule 1 of KNIT_NODE_SPEC.md is "the code wins, do not invent
/// deviations" — the checked-in golden images were rendered by a script
/// with exactly this asymmetry, so a faithful port keeps it. The one
/// sample where it could matter (`T[n-1]`, feeding `N[n-1]`) sits right at
/// a course's own sinker — already the most heavily blended point in the
/// whole curve — and the golden comparison in `knit_golden.rs` is the
/// actual check that this doesn't visibly matter, not an assumption.
/// Tangent, normal and binormal arrays, one entry per sample.
type Frame = (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<[f64; 3]>);

fn rotation_minimizing_frame(p: &[[f64; 3]]) -> Frame {
    let n = p.len();
    let mut t = vec![[0.0; 3]; n];
    for i in 0..n {
        let next = p[(i + 1) % n];
        let prev = p[(i + n - 1) % n];
        t[i] = normalize(sub(next, prev));
    }

    let mut nrm = vec![[0.0; 3]; n];
    let reference = [0.0, 0.0, 1.0];
    nrm[0] = normalize(sub(reference, scale(t[0], dot(reference, t[0]))));

    for i in 0..n - 1 {
        let v1 = sub(p[(i + 1) % n], p[i]);
        let c1 = dot(v1, v1);
        let r_l = sub(nrm[i], scale(v1, 2.0 / c1 * dot(v1, nrm[i])));
        let t_l = sub(t[i], scale(v1, 2.0 / c1 * dot(v1, t[i])));
        let v2 = sub(t[i + 1], t_l);
        let c2 = dot(v2, v2);
        nrm[i + 1] = if c2 > 1e-12 {
            sub(r_l, scale(v2, 2.0 / c2 * dot(v2, r_l)))
        } else {
            r_l
        };
    }

    let b: Vec<[f64; 3]> = (0..n).map(|i| cross(t[i], nrm[i])).collect();

    // Close the loop: measure the twist mismatch at the seam and spread
    // its correction evenly across the whole course so the frame is
    // continuous across the tile edge.
    let n_end = nrm[n - 1];
    let b_end = b[n - 1];
    let ang = dot(nrm[0], b_end).atan2(dot(nrm[0], n_end));
    let mut n2 = vec![[0.0; 3]; n];
    for (i, slot) in n2.iter_mut().enumerate() {
        let corr = -ang * (i as f64) / (n as f64);
        *slot = add(scale(nrm[i], corr.cos()), scale(b[i], corr.sin()));
    }
    let b2: Vec<[f64; 3]> = (0..n).map(|i| cross(t[i], n2[i])).collect();

    (t, n2, b2)
}

/// One ply's worth of sample points for one course: `x` wrapped to
/// `[0, 1)` (matching [`course_centerline`]'s own convention), `y`/`z`
/// left as computed. `radius` is constant across every sample —
/// `knit_reference.py`'s `course_tubes()` never varies it along a course.
struct PlyCurve {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    wale: Vec<u32>,
    radius: f64,
    ply: u32,
}

/// `knit_reference.py`'s own `course_tubes()`: the plain centerline for
/// `plies <= 1`, or — for `plies > 1` — each ply's own helix around a
/// rotation-minimizing frame following that centerline, with ply packing
/// shared with the Thread node ([`crate::thread_shape::packing`]) and the
/// whole-turn-count twist rule from KNIT_NODE_SPEC.md §6 ("rounding to
/// whole turns keeps the tile seamless" — the same reason Thread rounds
/// its own `turns_from_angle`).
fn course_tubes(
    g: &Geometry,
    params: &KnitParams,
    course: u32,
    samples_per_stitch: u32,
) -> Vec<PlyCurve> {
    let c = course_centerline(g, course, samples_per_stitch);
    if params.plies <= 1 {
        return vec![PlyCurve {
            x: c.x,
            y: c.y,
            z: c.z,
            wale: c.wale,
            radius: g.yarn_radius,
            ply: 0,
        }];
    }

    let n = c.x.len();
    let xu = unwrap_period(&c.x, 1.0);
    let p: Vec<[f64; 3]> = (0..n).map(|i| [xu[i], c.y[i], c.z[i]]).collect();
    let (_t, nrm, b) = rotation_minimizing_frame(&p);

    // Arc length along the course, tile-width-corrected at the true
    // closing point (unlike the frame's own tangent array above — see
    // rotation_minimizing_frame's doc comment for why that asymmetry is
    // intentional, not fixed here).
    let mut seg = vec![0.0f64; n];
    for i in 0..n - 1 {
        seg[i] = dist(p[i], p[i + 1]);
    }
    seg[n - 1] = dist(p[n - 1], add(p[0], [1.0, 0.0, 0.0]));
    let mut s_arc = vec![0.0f64; n];
    for i in 1..n {
        s_arc[i] = s_arc[i - 1] + seg[i - 1];
    }
    let length = s_arc[n - 1] + seg[n - 1];

    let (ap, rp) = crate::thread_shape::packing(g.yarn_radius as f32, params.plies);
    let (ap, rp) = (ap as f64, rp as f64);
    let turns = ((params.twist_angle.to_radians().tan() * length / (2.0 * PI * ap)).round() as i64)
        .max(1) as f64;
    let sgn = if params.ply_direction == crate::thread_shape::TwistDirection::S {
        1.0
    } else {
        -1.0
    };

    (0..params.plies)
        .map(|k| {
            let mut qx = vec![0.0; n];
            let mut qy = vec![0.0; n];
            let mut qz = vec![0.0; n];
            for i in 0..n {
                let th = sgn * 2.0 * PI * turns * s_arc[i] / length
                    + 2.0 * PI * k as f64 / params.plies as f64;
                let (cs, sn) = (th.cos(), th.sin());
                let off = add(scale(nrm[i], ap * cs), scale(b[i], ap * sn));
                let q = add(p[i], off);
                qx[i] = q[0].rem_euclid(1.0);
                qy[i] = q[1];
                qz[i] = q[2];
            }
            PlyCurve {
                x: qx,
                y: qy,
                z: qz,
                wale: c.wale.clone(),
                radius: rp,
                ply: k,
            }
        })
        .collect()
}

fn push_capsule_chain(
    segments: &mut Vec<crate::tube_raster::Segment>,
    course: u32,
    wales: u32,
    curve: &PlyCurve,
) {
    let n = curve.x.len();
    let radius = curve.radius as f32;
    for i in 0..n {
        let j = (i + 1) % n;
        let mut x1 = curve.x[j] as f32;
        if j == 0 {
            // Closing segment: back to this same course's own first
            // sample, one period of t further along — unwrap it so the
            // capsule is short, not a false chord across the tile.
            x1 += 1.0;
        }
        segments.push(crate::tube_raster::Segment {
            p0: [curve.x[i] as f32, curve.y[i] as f32, curve.z[i] as f32],
            r0: radius,
            p1: [x1, curve.y[j] as f32, curve.z[j] as f32],
            r1: radius,
            tag: (course * wales + curve.wale[i]) * 8 + curve.ply,
        });
    }
}

fn pack_stitch_params(resolution: u32, wales: u32, courses: u32, seed: u32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&wales.to_le_bytes());
    bytes[8..12].copy_from_slice(&courses.to_le_bytes());
    bytes[12..16].copy_from_slice(&seed.to_le_bytes());
    bytes
}

fn pack_f32_slice(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn pack_u32_slice(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Second pass: Stitch ID and Stitch Type, reading `cell_key` back from the
/// first ([`tube_raster`]) pass.
fn dispatch_stitch_info(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    keys: &KnitOutputKeys,
    cell_key: &str,
    resolution: u32,
    g: &Geometry,
    seed: u32,
) {
    let shader_source = crate::shaders::full_source("knit_stitch_info.wgsl");
    let pipeline = gpu.compute_pipeline("knit_stitch_info", &shader_source, "main");

    let mut widths_cumsum = Vec::with_capacity(g.widths.len());
    let mut cumulative = 0.0f64;
    for &width in &g.widths {
        cumulative += width;
        widths_cumsum.push(cumulative as f32);
    }

    let mut chart_is_knit = vec![0u32; (g.courses * g.wales) as usize];
    for course in 0..g.courses {
        let row = &g.chart[(course as usize) % g.chart_height];
        for wale in 0..g.wales {
            let is_knit = row[(wale as usize) % g.chart_width];
            chart_is_knit[(course * g.wales + wale) as usize] = is_knit as u32;
        }
    }

    let param_bytes = pack_stitch_params(resolution, g.wales, g.courses, seed);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("knit_stitch_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let widths_cumsum_bytes = pack_f32_slice(&widths_cumsum);
    let widths_cumsum_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("knit_stitch_widths_cumsum"),
        size: widths_cumsum_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&widths_cumsum_buffer, 0, &widths_cumsum_bytes);

    let chart_is_knit_bytes = pack_u32_slice(&chart_is_knit);
    let chart_is_knit_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("knit_stitch_chart_is_knit"),
        size: chart_is_knit_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&chart_is_knit_buffer, 0, &chart_is_knit_bytes);

    let cell_view = pool
        .get_or_create(
            &gpu.device,
            cell_key,
            resolution,
            wgpu::TextureFormat::R32Uint,
        )
        .clone();

    let stitch_id_view = pool
        .get_or_create(
            &gpu.device,
            keys.stitch_id,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let stitch_type_view = pool
        .get_or_create(
            &gpu.device,
            keys.stitch_type,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("knit_stitch_info_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: widths_cumsum_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: chart_is_knit_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&cell_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&stitch_id_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&stitch_type_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("knit_stitch_info_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("knit_stitch_info_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

/// Builds one capsule per sample gap in every course, for every ply (plus
/// the segment closing each course's own loop back to its first sample —
/// see [`push_capsule_chain`]), hands them to [`crate::tube_raster::dispatch`]
/// for Height/Mask/Ply ID, then runs a second pass
/// ([`dispatch_stitch_info`]) for Stitch ID/Stitch Type — two separate GPU
/// submissions on the same queue, which is enough on its own to guarantee
/// the second pass's reads see the first pass's writes (no explicit barrier
/// needed; this is also why the two passes aren't fused into one shader —
/// `tube_raster` stays generic, chart/hash logic stays in Knit's own pass).
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    keys: KnitOutputKeys,
    resolution: u32,
    params: &KnitParams,
) {
    let g = resolve(params);
    let sps = samples_per_stitch(resolution, g.wales);

    let mut segments = Vec::new();
    for course in 0..g.courses {
        for curve in course_tubes(&g, params, course, sps) {
            push_capsule_chain(&mut segments, course, g.wales, &curve);
        }
    }

    let cell_key = format!("{}__cell", keys.height);
    crate::tube_raster::dispatch(
        gpu,
        pool,
        crate::tube_raster::RasterOutputKeys {
            height: keys.height,
            mask: keys.mask,
            ply_id: keys.ply_id,
            cell: &cell_key,
        },
        resolution,
        &segments,
        params.plies.max(1),
        g.zmax as f32,
    );

    dispatch_stitch_info(gpu, pool, &keys, &cell_key, resolution, &g, params.seed);
}

#[cfg(test)]
mod tests {
    use super::*;

    const CURVES_JSON: &str = include_str!("../../../tests/golden/knit/curves.json");

    fn golden_params(case: &serde_json::Value) -> KnitParams {
        let p = &case["params"];
        KnitParams {
            preset: KnitPreset::from_name(match p["preset"].as_str().unwrap() {
                "jersey" => "Jersey",
                "reverse_jersey" => "Reverse Jersey",
                "purl" => "Purl",
                "rib_1x1" => "1x1 Rib",
                "rib_2x2" => "2x2 Rib",
                "rib_2x1" => "2x1 Rib",
                "rib_6x3" => "6x3 Rib",
                "interlock" => "Interlock",
                other => panic!("unknown preset {other:?}"),
            })
            .unwrap(),
            chart: None,
            side: match p["side"].as_str().unwrap() {
                "front" => Side::Front,
                "back" => Side::Back,
                other => panic!("unknown side {other:?}"),
            },
            wales: p["wales"].as_u64().unwrap() as u32,
            courses: p["courses"].as_u64().map(|v| v as u32),
            roundness: p["roundness"].as_f64().unwrap(),
            loop_height: p["loop_height"].as_f64(),
            depth: p["depth"].as_f64().unwrap(),
            tension: Tension::from_name(match p["tension"].as_str().unwrap() {
                "loose" => "Loose",
                "normal" => "Normal",
                "tight" => "Tight",
                other => panic!("unknown tension {other:?}"),
            })
            .unwrap(),
            thickness: p["thickness"].as_f64(),
            plies: p["plies"].as_u64().unwrap() as u32,
            ply_direction: crate::thread_shape::TwistDirection::from_name(
                p["ply_direction"].as_str().unwrap(),
            )
            .unwrap(),
            twist_angle: p["twist_angle"].as_f64().unwrap(),
            rib_depth: p["rib_depth"].as_f64().unwrap(),
            rib_compaction: p["rib_compaction"].as_f64().unwrap(),
            seed: p["seed"].as_u64().unwrap() as u32,
        }
    }

    fn assert_close(actual: f64, expected: f64, tol: f64, context: &str) {
        assert!(
            (actual - expected).abs() <= tol,
            "{context}: actual={actual} expected={expected} diff={}",
            (actual - expected).abs()
        );
    }

    #[test]
    fn resolve_matches_golden_geometry() {
        let golden: serde_json::Value = serde_json::from_str(CURVES_JSON).unwrap();
        for (name, case) in golden.as_object().unwrap() {
            let params = golden_params(case);
            let g = resolve(&params);
            let resolved = &case["resolved"];
            assert_eq!(
                g.wales,
                resolved["wales"].as_u64().unwrap() as u32,
                "{name}: wales"
            );
            assert_eq!(
                g.courses,
                resolved["courses"].as_u64().unwrap() as u32,
                "{name}: courses"
            );
            assert_close(
                g.yarn_radius,
                resolved["R"].as_f64().unwrap(),
                1e-9,
                &format!("{name}: R"),
            );
            assert_close(
                g.loop_half_height,
                resolved["h"].as_f64().unwrap(),
                1e-9,
                &format!("{name}: h"),
            );
            assert_close(
                g.depth,
                resolved["d"].as_f64().unwrap(),
                1e-9,
                &format!("{name}: d"),
            );
            assert_close(
                g.course_spacing,
                resolved["c"].as_f64().unwrap(),
                1e-9,
                &format!("{name}: c"),
            );
            assert_close(
                g.zmax,
                resolved["zmax"].as_f64().unwrap(),
                1e-9,
                &format!("{name}: zmax"),
            );
            for (j, expected) in resolved["widths"].as_array().unwrap().iter().enumerate() {
                assert_close(
                    g.widths[j],
                    expected.as_f64().unwrap(),
                    1e-9,
                    &format!("{name}: widths[{j}]"),
                );
            }
            for (j, expected) in resolved["zoff"].as_array().unwrap().iter().enumerate() {
                assert_close(
                    g.zoff[j],
                    expected.as_f64().unwrap(),
                    1e-9,
                    &format!("{name}: zoff[{j}]"),
                );
            }
            for (j, expected) in resolved["stagger"].as_array().unwrap().iter().enumerate() {
                assert_close(
                    g.stagger[j],
                    expected.as_f64().unwrap(),
                    1e-9,
                    &format!("{name}: stagger[{j}]"),
                );
            }
            for (i, expected) in resolved["czoff"].as_array().unwrap().iter().enumerate() {
                assert_close(
                    g.czoff[i],
                    expected.as_f64().unwrap(),
                    1e-9,
                    &format!("{name}: czoff[{i}]"),
                );
            }
        }
    }

    #[test]
    fn course_centerline_matches_golden_samples() {
        let golden: serde_json::Value = serde_json::from_str(CURVES_JSON).unwrap();
        for (name, case) in golden.as_object().unwrap() {
            let params = golden_params(case);
            let g = resolve(&params);
            let samples_per_stitch = case["samples_per_stitch"].as_u64().unwrap() as u32;
            for course_key in ["0", "1"] {
                let course: u32 = course_key.parse().unwrap();
                let expected = &case["courses"][course_key];
                let got = course_centerline(&g, course, samples_per_stitch);
                let ex = expected["x"].as_array().unwrap();
                let ey = expected["y"].as_array().unwrap();
                let ez = expected["z"].as_array().unwrap();
                let ew = expected["wale"].as_array().unwrap();
                assert_eq!(
                    got.x.len(),
                    ex.len(),
                    "{name} course {course}: sample count"
                );
                for i in 0..ex.len() {
                    assert_close(
                        got.x[i],
                        ex[i].as_f64().unwrap(),
                        1e-5,
                        &format!("{name} course {course}: x[{i}]"),
                    );
                    assert_close(
                        got.y[i],
                        ey[i].as_f64().unwrap(),
                        1e-5,
                        &format!("{name} course {course}: y[{i}]"),
                    );
                    assert_close(
                        got.z[i],
                        ez[i].as_f64().unwrap(),
                        1e-5,
                        &format!("{name} course {course}: z[{i}]"),
                    );
                    assert_eq!(
                        got.wale[i],
                        ew[i].as_u64().unwrap() as u32,
                        "{name} course {course}: wale[{i}]"
                    );
                }
            }
        }
    }

    #[test]
    fn every_preset_name_round_trips() {
        for name in KnitPreset::NAMES {
            assert!(
                KnitPreset::from_name(name).is_some(),
                "{name:?} should parse"
            );
        }
        assert!(KnitPreset::from_name("Nonsense").is_none());
    }

    #[test]
    fn every_tension_name_round_trips() {
        for name in Tension::NAMES {
            assert!(Tension::from_name(name).is_some(), "{name:?} should parse");
        }
        assert!(Tension::from_name("Nonsense").is_none());
    }

    /// Spec §10 acceptance: "Wales 7 with chart KKPP gives 8."
    #[test]
    fn wales_snap_to_a_multiple_of_chart_width() {
        let g = resolve(&KnitParams {
            chart: Some(vec!["KKPP".to_string()]),
            wales: 7,
            ..KnitParams::default()
        });
        assert_eq!(g.wales, 8);
    }

    /// Spec §10 acceptance: "Auto courses for jersey with 8 wales gives
    /// 11."
    #[test]
    fn auto_courses_for_default_jersey_gives_eleven() {
        let g = resolve(&KnitParams::default());
        assert_eq!(g.wales, 8);
        assert_eq!(g.courses, 11);
    }

    /// Spec §10 acceptance: "N = 2 gives a_p = r_p = 0.5 R. N = 3 gives
    /// r_p = 0.464 R." Same packing() Thread already has and already
    /// tests — checked again here only to confirm Knit is actually
    /// calling the shared one, not a quiet reimplementation.
    #[test]
    fn ply_packing_matches_thread_nodes_own() {
        let (a, r) = crate::thread_shape::packing(1.0, 2);
        assert!((a - 0.5).abs() < 1e-4 && (r - 0.5).abs() < 1e-4);
        let (_a, r) = crate::thread_shape::packing(1.0, 3);
        assert!((r - 0.464).abs() < 1e-3);
    }

    // --- Step 2: dispatch() / tube_raster integration ---

    struct Rendered {
        height: Vec<u8>,
        mask: Vec<u8>,
        ply_id: Vec<u8>,
        stitch_id: Vec<u8>,
        stitch_type: Vec<u8>,
    }

    fn render(params: &KnitParams, resolution: u32) -> Rendered {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(
            &mut gpu,
            &mut pool,
            KnitOutputKeys {
                height: "height",
                mask: "mask",
                ply_id: "ply_id",
                stitch_id: "stitch_id",
                stitch_type: "stitch_type",
            },
            resolution,
            params,
        );
        Rendered {
            height: gpu.read_rgba8(pool.texture("height").unwrap(), resolution),
            mask: gpu.read_rgba8(pool.texture("mask").unwrap(), resolution),
            ply_id: gpu.read_rgba8(pool.texture("ply_id").unwrap(), resolution),
            stitch_id: gpu.read_rgba8(pool.texture("stitch_id").unwrap(), resolution),
            stitch_type: gpu.read_rgba8(pool.texture("stitch_type").unwrap(), resolution),
        }
    }

    #[test]
    fn dispatch_is_deterministic() {
        let params = KnitParams::default();
        let a = render(&params, crate::test_support::RESOLUTION);
        let b = render(&params, crate::test_support::RESOLUTION);
        assert_eq!(a.height, b.height);
        assert_eq!(a.mask, b.mask);
        assert_eq!(a.ply_id, b.ply_id);
    }

    #[test]
    fn dispatch_tiles_seamlessly() {
        let r = render(&KnitParams::default(), crate::test_support::RESOLUTION);
        crate::test_support::assert_tiles_seamlessly(&r.height, crate::test_support::RESOLUTION);
    }

    #[test]
    fn mask_is_strictly_binary_and_covers_most_of_the_fabric() {
        let r = render(&KnitParams::default(), crate::test_support::RESOLUTION);
        assert!(
            r.mask.iter().step_by(4).all(|&v| v == 0 || v == 255),
            "every Mask pixel should be exactly 0 or 255"
        );
        let coverage = r.mask.iter().step_by(4).filter(|&&v| v == 255).count() as f32
            / r.mask.len() as f32
            * 4.0;
        assert!(
            coverage > 0.5,
            "a plain-knit fabric should cover well over half the tile (got {coverage})"
        );
    }

    #[test]
    fn height_is_zero_wherever_mask_is_zero() {
        let r = render(&KnitParams::default(), crate::test_support::RESOLUTION);
        for i in (0..r.mask.len()).step_by(4) {
            if r.mask[i] == 0 {
                assert_eq!(
                    r.height[i], 0,
                    "background (mask=0) should always be height=0"
                );
            }
        }
    }

    /// `knit_reference.py`'s own `ply01 = mask.astype(float)` fallback for
    /// `plies <= 1` — Ply ID should read identically to Mask.
    #[test]
    fn ply_id_matches_mask_for_a_single_ply() {
        let r = render(&KnitParams::default(), crate::test_support::RESOLUTION);
        assert_eq!(r.ply_id, r.mask);
    }

    // --- Step 4: plies / rotation-minimizing frame ---

    #[test]
    fn unwrap_period_removes_a_false_wrap_jump() {
        let values = vec![0.9, 0.95, 0.01, 0.05]; // wraps 0.95 -> 0.01
        let unwrapped = unwrap_period(&values, 1.0);
        assert!((unwrapped[0] - 0.9).abs() < 1e-9);
        assert!((unwrapped[1] - 0.95).abs() < 1e-9);
        // continuing past 1.0, not snapping back down
        assert!((unwrapped[2] - 1.01).abs() < 1e-9);
        assert!((unwrapped[3] - 1.05).abs() < 1e-9);
    }

    #[test]
    fn unwrap_period_is_a_no_op_on_an_already_continuous_sequence() {
        let values = vec![0.1, 0.2, 0.3, 0.4];
        let unwrapped = unwrap_period(&values, 1.0);
        assert_eq!(unwrapped, values);
    }

    /// A rotation-minimizing frame's N and B must stay perpendicular to T
    /// (and to each other) at every sample — the double-reflection method's
    /// whole point is propagating an orthonormal frame without drift.
    #[test]
    fn rotation_minimizing_frame_stays_orthonormal() {
        let params = KnitParams::default();
        let g = resolve(&params);
        let c = course_centerline(&g, 0, 64);
        let n = c.x.len();
        let xu = unwrap_period(&c.x, 1.0);
        let p: Vec<[f64; 3]> = (0..n).map(|i| [xu[i], c.y[i], c.z[i]]).collect();
        let (t, nrm, b) = rotation_minimizing_frame(&p);
        for i in 0..n {
            assert!(dot(t[i], nrm[i]).abs() < 1e-6, "T.N at {i}");
            assert!(dot(t[i], b[i]).abs() < 1e-6, "T.B at {i}");
            assert!(dot(nrm[i], b[i]).abs() < 1e-6, "N.B at {i}");
            assert!((dot(nrm[i], nrm[i]) - 1.0).abs() < 1e-6, "|N| at {i}");
        }
    }

    #[test]
    fn dispatch_with_plies_is_deterministic_and_tiles_seamlessly() {
        let params = KnitParams {
            wales: 6,
            plies: 2,
            ..KnitParams::default()
        };
        let a = render(&params, crate::test_support::RESOLUTION);
        let b = render(&params, crate::test_support::RESOLUTION);
        assert_eq!(a.height, b.height);
        assert_eq!(a.ply_id, b.ply_id);
        crate::test_support::assert_tiles_seamlessly(&a.height, crate::test_support::RESOLUTION);
    }

    /// `course_tubes()` returns the plain centerline untouched for
    /// `plies <= 1`, so Ply Twist's S/Z sign (and Twist Angle) literally
    /// never gets read there — not a bug, but easy to mistake for one when
    /// testing at the default Plies 1 (reported exactly that way: "S/Z
    /// doesn't create a change").
    #[test]
    fn ply_direction_changes_output_only_with_plies_2_or_more() {
        let render_with = |plies, dir| {
            render(
                &KnitParams {
                    plies,
                    ply_direction: dir,
                    ..KnitParams::default()
                },
                crate::test_support::RESOLUTION,
            )
        };
        let s1 = render_with(1, crate::thread_shape::TwistDirection::S);
        let z1 = render_with(1, crate::thread_shape::TwistDirection::Z);
        assert_eq!(
            s1.height, z1.height,
            "S vs Z should have no effect at Plies 1 (no ply arrangement to change)"
        );

        let s2 = render_with(2, crate::thread_shape::TwistDirection::S);
        let z2 = render_with(2, crate::thread_shape::TwistDirection::Z);
        assert_ne!(
            s2.height, z2.height,
            "S vs Z should visibly change the render at Plies 2+"
        );
    }

    #[test]
    fn ply_id_shows_every_ply_for_a_plied_yarn() {
        let params = KnitParams {
            wales: 6,
            plies: 3,
            ..KnitParams::default()
        };
        let r = render(&params, crate::test_support::RESOLUTION);
        let distinct: std::collections::HashSet<u8> = r
            .ply_id
            .iter()
            .step_by(4)
            .zip(r.mask.iter().step_by(4))
            .filter(|(_, &m)| m > 0)
            .map(|(&v, _)| v)
            .collect();
        assert!(
            distinct.len() >= 2,
            "3 plies should show at least 2 distinct Ply ID values where drawn (got {})",
            distinct.len()
        );
    }

    // --- Step 5: Stitch ID / Stitch Type ---

    #[test]
    fn stitch_type_is_strictly_binary() {
        let r = render(&KnitParams::default(), crate::test_support::RESOLUTION);
        assert!(
            r.stitch_type.iter().step_by(4).all(|&v| v == 0 || v == 255),
            "every Stitch Type pixel should be exactly 0 or 255"
        );
    }

    #[test]
    fn stitch_type_is_all_knit_for_plain_jersey() {
        let r = render(&KnitParams::default(), crate::test_support::RESOLUTION);
        assert!(r.stitch_type.iter().step_by(4).all(|&v| v == 255));
    }

    #[test]
    fn stitch_type_is_all_purl_for_reverse_jersey() {
        let r = render(
            &KnitParams {
                preset: KnitPreset::ReverseJersey,
                ..KnitParams::default()
            },
            crate::test_support::RESOLUTION,
        );
        assert!(r.stitch_type.iter().step_by(4).all(|&v| v == 0));
    }

    /// Spec §7: "Holes take the stitch cell they sit in" — Stitch Type
    /// should be defined even where Mask is 0, not left blank. A rib has
    /// both all-knit and all-purl wales, so a chart with uniform Stitch
    /// Type everywhere (like the two tests above) wouldn't catch a
    /// gap-fallback bug the way this one can.
    #[test]
    fn stitch_type_is_defined_in_mask_gaps_for_a_rib() {
        let r = render(
            &KnitParams {
                preset: KnitPreset::Rib2x2,
                wales: 12,
                ..KnitParams::default()
            },
            crate::test_support::RESOLUTION,
        );
        let gap_values: std::collections::HashSet<u8> = r
            .stitch_type
            .iter()
            .step_by(4)
            .zip(r.mask.iter().step_by(4))
            .filter(|(_, &m)| m == 0)
            .map(|(&v, _)| v)
            .collect();
        assert!(!gap_values.is_empty(), "test assumes rib has visible gaps");
        assert!(
            gap_values.len() >= 2,
            "a rib's gaps should show both Stitch Types depending on which \
             wale's gap they sit in (got {gap_values:?})"
        );
    }

    #[test]
    fn stitch_id_shows_variety_across_the_tile() {
        let r = render(&KnitParams::default(), crate::test_support::RESOLUTION);
        let distinct: std::collections::HashSet<u8> =
            r.stitch_id.iter().step_by(4).copied().collect();
        assert!(
            distinct.len() > 4,
            "Stitch ID should vary across multiple stitch cells, got {} distinct values",
            distinct.len()
        );
    }

    #[test]
    fn stitch_id_is_deterministic_and_seed_dependent() {
        let a = render(
            &KnitParams {
                seed: 1,
                ..KnitParams::default()
            },
            crate::test_support::RESOLUTION,
        );
        let b = render(
            &KnitParams {
                seed: 1,
                ..KnitParams::default()
            },
            crate::test_support::RESOLUTION,
        );
        assert_eq!(a.stitch_id, b.stitch_id);
        let c = render(
            &KnitParams {
                seed: 2,
                ..KnitParams::default()
            },
            crate::test_support::RESOLUTION,
        );
        assert_ne!(a.stitch_id, c.stitch_id);
    }

    // --- Step 6: from_graph() — the UI-facing boundary ---

    fn graph_params(
        entries: &[(&str, mc_core::ParamValue)],
    ) -> std::collections::BTreeMap<String, mc_core::ParamValue> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    /// CLAUDE.md §12: no path a user can trigger should panic. A mismatched
    /// row length in hand-typed Chart text used to reach `resolve()`'s own
    /// `assert!` directly — this is the regression test for the fix
    /// (`from_graph` now falls back to the preset's own chart instead).
    #[test]
    fn from_graph_does_not_panic_on_malformed_chart_text() {
        let params = graph_params(&[(
            "chart",
            mc_core::ParamValue::Text("KP/KKP".to_string()), // mismatched row lengths
        )]);
        let knit_params = KnitParams::from_graph(&params);
        assert!(knit_params.chart.is_none());
        let _ = resolve(&knit_params); // must not panic
    }

    #[test]
    fn from_graph_empty_chart_text_means_use_the_preset() {
        let params = graph_params(&[("chart", mc_core::ParamValue::Text(String::new()))]);
        assert!(KnitParams::from_graph(&params).chart.is_none());
    }

    #[test]
    fn from_graph_parses_a_well_formed_custom_chart() {
        let params = graph_params(&[("chart", mc_core::ParamValue::Text("kkpp/ppkk".to_string()))]);
        let knit_params = KnitParams::from_graph(&params);
        assert_eq!(
            knit_params.chart,
            Some(vec!["KKPP".to_string(), "PPKK".to_string()])
        );
    }

    #[test]
    fn from_graph_zero_thickness_and_loop_height_mean_derive_from_tension() {
        let params = graph_params(&[
            ("thickness", mc_core::ParamValue::Float(0.0)),
            ("loop_height", mc_core::ParamValue::Float(0.0)),
        ]);
        let knit_params = KnitParams::from_graph(&params);
        assert!(knit_params.thickness.is_none());
        assert!(knit_params.loop_height.is_none());
    }

    #[test]
    fn from_graph_nonzero_thickness_and_loop_height_are_real_overrides() {
        let params = graph_params(&[
            ("thickness", mc_core::ParamValue::Float(0.2)),
            ("loop_height", mc_core::ParamValue::Float(1.0)),
        ]);
        let knit_params = KnitParams::from_graph(&params);
        assert!((knit_params.thickness.unwrap() - 0.2).abs() < 1e-6);
        assert!((knit_params.loop_height.unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn from_graph_auto_courses_bool_controls_courses_option() {
        let auto = graph_params(&[("auto_courses", mc_core::ParamValue::Bool(true))]);
        assert!(KnitParams::from_graph(&auto).courses.is_none());

        let manual = graph_params(&[
            ("auto_courses", mc_core::ParamValue::Bool(false)),
            ("courses", mc_core::ParamValue::Int(20)),
        ]);
        assert_eq!(KnitParams::from_graph(&manual).courses, Some(20));
    }

    #[test]
    fn from_graph_round_trips_every_enum_and_scalar_param() {
        let params = graph_params(&[
            ("preset", mc_core::ParamValue::Enum("2x2 Rib".to_string())),
            ("side", mc_core::ParamValue::Enum("Back".to_string())),
            ("wales", mc_core::ParamValue::Int(12)),
            ("roundness", mc_core::ParamValue::Float(1.2)),
            ("depth", mc_core::ParamValue::Float(1.0)),
            ("tension", mc_core::ParamValue::Enum("Tight".to_string())),
            ("plies", mc_core::ParamValue::Int(3)),
            ("ply_direction", mc_core::ParamValue::Enum("Z".to_string())),
            ("twist_angle", mc_core::ParamValue::Float(40.0)),
            ("rib_depth", mc_core::ParamValue::Float(2.0)),
            ("rib_compaction", mc_core::ParamValue::Float(0.6)),
            ("seed", mc_core::ParamValue::Seed(42)),
        ]);
        let knit_params = KnitParams::from_graph(&params);
        assert_eq!(knit_params.preset, KnitPreset::Rib2x2);
        assert_eq!(knit_params.side, Side::Back);
        assert_eq!(knit_params.wales, 12);
        assert!((knit_params.roundness - 1.2).abs() < 1e-6);
        assert!((knit_params.depth - 1.0).abs() < 1e-6);
        assert_eq!(knit_params.tension, Tension::Tight);
        assert_eq!(knit_params.plies, 3);
        assert_eq!(
            knit_params.ply_direction,
            crate::thread_shape::TwistDirection::Z
        );
        assert!((knit_params.twist_angle - 40.0).abs() < 1e-6);
        assert!((knit_params.rib_depth - 2.0).abs() < 1e-6);
        assert!((knit_params.rib_compaction - 0.6).abs() < 1e-6);
        assert_eq!(knit_params.seed, 42);
    }

    #[test]
    fn every_side_name_round_trips() {
        for name in Side::NAMES {
            assert!(Side::from_name(name).is_some(), "{name:?} should parse");
        }
        assert!(Side::from_name("Nonsense").is_none());
    }

    #[test]
    fn custom_preset_has_no_extra_modifiers() {
        assert_eq!(KnitPreset::Custom.name(), "Custom");
        let g = resolve(&KnitParams {
            preset: KnitPreset::Custom,
            chart: Some(vec!["KP".to_string()]),
            ..KnitParams::default()
        });
        // Jersey-equivalent neutral behavior: no stagger, and the rib rule
        // only fires because THIS chart has both all-K and all-P wales, not
        // because of any preset-specific extra.
        assert!(g.stagger.iter().all(|&s| s == 0.0));
    }

    /// Picking a preset resets Wales to `canonical_wales()` — it should
    /// always be an exact multiple of that preset's own chart width, or
    /// `resolve()`'s own snapping would silently round it up to a
    /// different number than what the UI just set, right after the user
    /// picked it.
    #[test]
    fn canonical_wales_is_a_multiple_of_its_own_chart_width() {
        for name in KnitPreset::NAMES {
            let preset = KnitPreset::from_name(name).unwrap();
            let chart_width = preset.default_chart_rows()[0].len() as u32;
            let wales = preset.canonical_wales();
            assert_eq!(
                wales % chart_width,
                0,
                "{name}: canonical_wales {wales} isn't a multiple of its own chart width \
                 {chart_width}"
            );
        }
    }
}
