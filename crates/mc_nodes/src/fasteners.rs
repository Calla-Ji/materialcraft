// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Path / Fasteners.
//!
//! Places rivets/bolts/screws evenly along any `Path` (typically Panel
//! Layout's own fastener output), in 6 head shapes, seated on an optional
//! base height (`docs/hardsurface_nodes/HARDSURFACE_NODES.md` §3.7).
//!
//! **CPU-only, deliberately** — each fastener's compositing is
//! sequential and stateful (`cur = height[window]` reads whatever
//! earlier fasteners already painted there, so draw order is part of the
//! reference's own defined behavior), a scatter/read-after-write pattern
//! unlike the uniform independent-per-pixel map every GPU node in this
//! codebase assumes. Fastener counts are bounded (hundreds, not
//! millions) and each only touches a small local window around its own
//! center, so a plain CPU loop is cheap — likely cheaper than Distance
//! Field's own already-accepted full-image CPU pass (`distance_field.rs`'s
//! own doc comment has the same category of justification).

use mc_core::Path;

use crate::stitch::{hash01, round_half_even};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastenerKind {
    RivetDome,
    RivetFlush,
    HexBolt,
    ScrewSlot,
    ScrewCross,
    SocketCap,
}

impl FastenerKind {
    pub const NAMES: [&'static str; 6] = [
        "Rivet Dome",
        "Rivet Flush",
        "Hex Bolt",
        "Screw Slot",
        "Screw Cross",
        "Socket Cap",
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Rivet Dome" => Some(Self::RivetDome),
            "Rivet Flush" => Some(Self::RivetFlush),
            "Hex Bolt" => Some(Self::HexBolt),
            "Screw Slot" => Some(Self::ScrewSlot),
            "Screw Cross" => Some(Self::ScrewCross),
            "Socket Cap" => Some(Self::SocketCap),
            _ => None,
        }
    }
}

pub struct FastenersParams {
    pub kind: FastenerKind,
    /// Tile units — target distance between fasteners along the path.
    pub spacing: f32,
    pub head_radius: f32,
    /// Relative to the bevel depth units Bevel Profile uses (1.0 = a
    /// full panel depth).
    pub head_height: f32,
    pub seed: u32,
    /// 0..1 — probability a given fastener is absent (deterministic per
    /// position, not re-rolled between evaluations).
    pub missing: f32,
    /// Each fastener gets its own random rotation when on; off keeps
    /// every head upright.
    pub rotate: bool,
}

impl Default for FastenersParams {
    fn default() -> Self {
        Self {
            kind: FastenerKind::RivetDome,
            spacing: 0.05,
            head_radius: 0.006,
            head_height: 0.6,
            seed: 1,
            missing: 0.0,
            rotate: true,
        }
    }
}

impl FastenersParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            kind: params
                .get("kind")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => FastenerKind::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.kind),
            spacing: params
                .get("spacing")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.spacing),
            head_radius: params
                .get("head_radius")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.head_radius),
            head_height: params
                .get("head_height")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.head_height),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
            missing: params
                .get("missing")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.missing),
            rotate: params
                .get("rotate")
                .and_then(mc_core::ParamValue::as_bool)
                .unwrap_or(defaults.rotate),
        }
    }
}

/// Every section of `path` (`crate::path::sections`, the same corner-
/// splitting Stitch already uses) gets `ceil(length / spacing)` evenly
/// spaced points starting at the section's own start — so every corner
/// gets a fastener — plus the end point for an open section. Positions
/// within `0.5 * spacing` of an earlier one are dropped (wrap-aware).
/// Ported from `hs_reference.py`'s `fastener_positions`.
pub(crate) fn fastener_positions(path: &Path, spacing: f64) -> Vec<[f64; 2]> {
    const CORNER_THRESHOLD_DEG: f64 = 30.0;
    let mut raw = Vec::new();
    for sec in crate::path::sections(path, CORNER_THRESHOLD_DEG) {
        let length = sec.length();
        let n = ((length / spacing - 1e-9).ceil() as i64).max(1);
        let extra = if sec.closed || sec.end_corner.is_some() {
            0
        } else {
            1
        };
        for i in 0..(n + extra) {
            let (p, _, _) = sec.at(i as f64 * length / n as f64);
            raw.push(p);
        }
    }

    let mut keep: Vec<[f64; 2]> = Vec::new();
    for p in raw {
        let mut ok = true;
        for q in &keep {
            let mut dx = p[0] - q[0];
            let mut dy = p[1] - q[1];
            if path.wrap {
                dx -= round_half_even(dx);
                dy -= round_half_even(dy);
            }
            if (dx * dx + dy * dy).sqrt() < 0.5 * spacing {
                ok = false;
                break;
            }
        }
        if ok {
            keep.push(p);
        }
    }
    keep
}

/// Anti-aliased inside-coverage of an SDF value — the same `coverage()`
/// convention used everywhere else in this package, kept local since
/// this module works in `f64` throughout (precision headroom for the
/// per-fastener math below; nothing here needs `pub(crate)` sharing).
fn coverage(sdf: f64, res: f64) -> f64 {
    (0.5 - sdf * res).clamp(0.0, 1.0)
}

/// iq regular hexagon SDF (flat-top), `r` = inradius.
fn sd_hexagon(px: f64, py: f64, r: f64) -> f64 {
    const KX: f64 = -0.866_025_404;
    const KY: f64 = 0.5;
    const KZ: f64 = 0.577_350_269;
    let mut px = px.abs();
    let mut py = py.abs();
    let dot = KX * px + KY * py;
    let m = 2.0 * dot.min(0.0);
    px -= m * KX;
    py -= m * KY;
    px -= px.clamp(-KZ * r, KZ * r);
    py -= r;
    (px * px + py * py).sqrt() * py.signum()
}

/// Height of one fastener head in local coordinates (center at origin,
/// already rotated), radius `r`, height `h`. Ported from
/// `hs_reference.py`'s `fastener_head`.
fn fastener_head(kind: FastenerKind, lx: f64, ly: f64, r: f64, h: f64, res: f64) -> f64 {
    let rad = (lx * lx + ly * ly).sqrt();
    let disc = rad - r;
    let cov = coverage(disc, res);
    match kind {
        FastenerKind::RivetDome => h * (1.0 - (rad / r).powi(2)).clamp(0.0, 1.0).sqrt() * cov,
        FastenerKind::RivetFlush => {
            let ring = (1.0 - (rad - 0.9 * r).abs() / (0.08 * r)).clamp(0.0, 1.0);
            (0.08 * h - 0.25 * h * ring) * cov
        }
        FastenerKind::HexBolt => {
            let washer = 0.18 * h * cov;
            let hx = sd_hexagon(lx, ly, 0.78 * r);
            let top = h * (-hx / (0.12 * r)).clamp(0.0, 1.0).sqrt() * coverage(hx, res);
            washer.max(top)
        }
        FastenerKind::ScrewSlot => {
            let pan = h * 0.8 * (1.0 - (rad / r).powi(4)).clamp(0.0, 1.0).sqrt() * cov;
            if ly.abs() < 0.11 * r && rad < 0.95 * r {
                pan * 0.35
            } else {
                pan
            }
        }
        FastenerKind::ScrewCross => {
            let pan = h * 0.8 * (1.0 - (rad / r).powi(4)).clamp(0.0, 1.0).sqrt() * cov;
            if (ly.abs() < 0.1 * r || lx.abs() < 0.1 * r) && rad < 0.55 * r {
                let cone = 0.25 * h + 0.5 * h * (rad / (0.55 * r));
                pan.min(cone)
            } else {
                pan
            }
        }
        FastenerKind::SocketCap => {
            let cap = h * (-disc / (0.12 * r)).clamp(0.0, 1.0) * cov;
            let hole = sd_hexagon(lx, ly, 0.42 * r);
            if hole < 0.0 {
                cap * 0.3
            } else {
                cap
            }
        }
    }
}

pub struct FastenersOutputKeys<'a> {
    pub height: &'a str,
    pub mask: &'a str,
    pub fastener_id: &'a str,
}

pub fn dispatch(
    gpu: &mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    path: &Path,
    base_key: Option<&str>,
    keys: FastenersOutputKeys,
    resolution: u32,
    params: &FastenersParams,
) {
    let res = resolution as usize;
    let res_f = f64::from(resolution);

    let base: Vec<f32> = match base_key {
        Some(key) => {
            let texture = pool
                .texture(key)
                .unwrap_or_else(|| panic!("fasteners: no texture at base key {key:?}"));
            gpu.read_rgba16f(texture, resolution)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| p[0])
                .collect()
        }
        None => vec![0.0f32; res * res],
    };

    let mut height = base.clone();
    let mut mask = vec![0.0f32; res * res];
    let mut fastener_id = vec![0.0f32; res * res];

    let spacing = f64::from(params.spacing).max(1e-6);
    let positions = fastener_positions(path, spacing);
    let head_radius = f64::from(params.head_radius).max(1e-6);
    let head_height = f64::from(params.head_height);
    let rpx = (head_radius * res_f).ceil() as i64 + 2;
    let res_i = resolution as i64;

    for (k, p) in positions.iter().enumerate() {
        let kk = k as u32;
        if hash01(kk, 21, params.seed) < f64::from(params.missing) {
            continue;
        }
        let cx = p[0] * res_f - 0.5;
        let cy = p[1] * res_f - 0.5;
        let rcx = round_half_even(cx) as i64;
        let rcy = round_half_even(cy) as i64;
        let b0 = f64::from(
            base[(rcy.rem_euclid(res_i) as usize) * res + (rcx.rem_euclid(res_i) as usize)],
        );

        let angle = if params.rotate {
            2.0 * std::f64::consts::PI * hash01(kk, 22, params.seed)
        } else {
            0.0
        };
        let (sin_a, cos_a) = angle.sin_cos();

        for oy in -rpx..=rpx {
            for ox in -rpx..=rpx {
                let raw_lx = (rcx as f64 + ox as f64 - cx) / res_f;
                let raw_ly = (rcy as f64 + oy as f64 - cy) / res_f;
                let (lx, ly) = if params.rotate {
                    (
                        raw_lx * cos_a + raw_ly * sin_a,
                        -raw_lx * sin_a + raw_ly * cos_a,
                    )
                } else {
                    (raw_lx, raw_ly)
                };
                let rad = (lx * lx + ly * ly).sqrt();
                let cov = coverage(rad - head_radius, res_f);
                if cov <= 0.0 {
                    // A no-op blend — skip the write (same result, cheaper).
                    continue;
                }
                let hd = fastener_head(params.kind, lx, ly, head_radius, head_height, res_f);

                let ix = (rcx + ox).rem_euclid(res_i) as usize;
                let iy = (rcy + oy).rem_euclid(res_i) as usize;
                let idx = iy * res + ix;

                let cur = f64::from(height[idx]);
                let top = if params.kind == FastenerKind::RivetFlush {
                    b0 + hd
                } else {
                    cur.max(b0 + hd)
                };
                height[idx] = (cur * (1.0 - cov) + top * cov) as f32;
                mask[idx] = mask[idx].max(cov as f32);
                if cov > 0.5 {
                    fastener_id[idx] = hash01(kk, 23, params.seed) as f32;
                }
            }
        }
    }

    gpu.write_rgba16f(pool, keys.height, resolution, &to_pixels(&height));
    gpu.write_rgba16f(pool, keys.mask, resolution, &to_pixels(&mask));
    gpu.write_rgba16f(pool, keys.fastener_id, resolution, &to_pixels(&fastener_id));
}

fn to_pixels(values: &[f32]) -> Vec<[f32; 4]> {
    values.iter().map(|&v| [v, v, v, 1.0]).collect()
}

#[cfg(test)]
mod tests {
    use super::{dispatch, fastener_positions, FastenerKind, FastenersOutputKeys, FastenersParams};
    use crate::test_support::{golden_path, read_png16_as_unit_floats};
    use mc_core::{Path, Point, SubPath};

    const RES: u32 = 512;

    fn default_panel_path() -> Path {
        let params = crate::panel_layout::PanelLayoutParams::default();
        let rects = crate::panel_layout::panel_rects(&params);
        crate::panel_layout::fastener_path(&rects, &params)
    }

    #[test]
    fn matches_the_positions_golden() {
        let path = default_panel_path();
        let got = fastener_positions(&path, 0.045);

        let text =
            std::fs::read_to_string(golden_path("hardsurface", "fastener_positions.json")).unwrap();
        let golden: serde_json::Value = serde_json::from_str(&text).unwrap();
        let want = golden["positions"].as_array().unwrap();

        assert_eq!(got.len(), want.len(), "position count mismatch");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let w = w.as_array().unwrap();
            let wx = w[0].as_f64().unwrap();
            let wy = w[1].as_f64().unwrap();
            assert!((g[0] - wx).abs() < 1e-6, "position {i}: x {} vs {wx}", g[0]);
            assert!((g[1] - wy).abs() < 1e-6, "position {i}: y {} vs {wy}", g[1]);
        }
    }

    struct Outputs {
        height: Vec<f32>,
        mask: Vec<f32>,
    }

    fn run(params: &FastenersParams) -> Outputs {
        let gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(
            &gpu,
            &mut pool,
            &default_panel_path(),
            None,
            FastenersOutputKeys {
                height: "height",
                mask: "mask",
                fastener_id: "fastener_id",
            },
            RES,
            params,
        );
        let read = |key: &str| -> Vec<f32> {
            gpu.read_rgba16f(pool.texture(key).unwrap(), RES)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| p[0])
                .collect()
        };
        Outputs {
            height: read("height"),
            mask: read("mask"),
        }
    }

    /// HARDSURFACE_NODES.md §6: mean abs under 0.002, max under 0.02 — max
    /// may be exceeded only on 1-px AA edges, at most 0.5% of pixels.
    fn assert_matches_golden(got: &[f32], want: &[f32], label: &str) {
        let diffs: Vec<f32> = got
            .iter()
            .zip(want.iter())
            .map(|(&g, &w)| (g - w).abs())
            .collect();
        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        let exceeding = diffs.iter().filter(|&&d| d > 0.02).count();
        let frac = exceeding as f32 / diffs.len() as f32;
        assert!(mean < 0.002, "{label}: mean abs diff {mean} exceeds 0.002");
        assert!(
            frac < 0.005,
            "{label}: {frac} of pixels exceed 0.02 abs diff (limit 0.005)"
        );
    }

    #[test]
    fn matches_every_kind_golden() {
        for (kind, name) in [
            (FastenerKind::RivetDome, "rivet_dome"),
            (FastenerKind::RivetFlush, "rivet_flush"),
            (FastenerKind::HexBolt, "hex_bolt"),
            (FastenerKind::ScrewSlot, "screw_slot"),
            (FastenerKind::ScrewCross, "screw_cross"),
            (FastenerKind::SocketCap, "socket_cap"),
        ] {
            let out = run(&FastenersParams {
                kind,
                spacing: 0.045,
                head_radius: 0.006,
                head_height: 0.3,
                seed: 1,
                missing: 0.0,
                rotate: true,
            });

            // Golden heights are encoded -0.1..0.4 (HARDSURFACE_NODES.md
            // §6) — encode my own raw output the same way rather than
            // decoding the golden, so the comparison stays in one
            // normalized space.
            let encoded_height: Vec<f32> = out
                .height
                .iter()
                .map(|&h| ((h - -0.1) / 0.5).clamp(0.0, 1.0))
                .collect();
            let (want_height, _) = read_png16_as_unit_floats(&golden_path(
                "hardsurface",
                &format!("fasteners_{name}_height.png"),
            ));
            assert_matches_golden(&encoded_height, &want_height, &format!("{name}_height"));

            let (want_mask, _) = read_png16_as_unit_floats(&golden_path(
                "hardsurface",
                &format!("fasteners_{name}_mask.png"),
            ));
            assert_matches_golden(&out.mask, &want_mask, &format!("{name}_mask"));
        }
    }

    #[test]
    fn missing_one_leaves_the_mask_empty() {
        let out = run(&FastenersParams {
            missing: 1.0,
            ..FastenersParams::default()
        });
        assert!(
            out.mask.iter().all(|&v| v == 0.0),
            "missing = 1.0 should skip every fastener"
        );
    }

    #[test]
    fn rotate_changes_the_result() {
        let base = FastenersParams {
            kind: FastenerKind::HexBolt,
            spacing: 0.045,
            head_radius: 0.01,
            head_height: 0.3,
            seed: 1,
            missing: 0.0,
            rotate: true,
        };
        let rotated = run(&base);
        let unrotated = run(&FastenersParams {
            rotate: false,
            ..base
        });
        assert_ne!(
            rotated.height, unrotated.height,
            "rotate on vs off should disagree for a non-radially-symmetric head"
        );
    }

    /// Every non-flush kind should seat at or above a non-zero `base` at
    /// its own center (`max(current, b0 + hd)`); `rivet_flush` is the one
    /// documented exception (its ring groove can cut below the surface).
    #[test]
    fn seats_on_top_of_a_nonzero_base() {
        let path = Path {
            subpaths: vec![SubPath::new(
                vec![Point::new([0.1, 0.5]), Point::new([0.9, 0.5])],
                false,
            )],
            wrap: true,
        };
        let base_value = 0.2f32;
        let gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        gpu.write_rgba16f(
            &mut pool,
            "base",
            RES,
            &vec![[base_value, base_value, base_value, 1.0]; (RES * RES) as usize],
        );

        for (kind, _) in [
            (FastenerKind::RivetDome, "rivet_dome"),
            (FastenerKind::HexBolt, "hex_bolt"),
            (FastenerKind::ScrewSlot, "screw_slot"),
            (FastenerKind::ScrewCross, "screw_cross"),
            (FastenerKind::SocketCap, "socket_cap"),
        ] {
            dispatch(
                &gpu,
                &mut pool,
                &path,
                Some("base"),
                FastenersOutputKeys {
                    height: "height",
                    mask: "mask",
                    fastener_id: "fastener_id",
                },
                RES,
                &FastenersParams {
                    kind,
                    spacing: 0.1,
                    head_radius: 0.02,
                    head_height: 0.3,
                    seed: 1,
                    missing: 0.0,
                    rotate: false,
                },
            );
            let height: Vec<f32> = gpu
                .read_rgba16f(pool.texture("height").unwrap(), RES)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| p[0])
                .collect();
            let min = height.iter().copied().fold(f32::INFINITY, f32::min);
            assert!(
                min >= base_value - 1e-3,
                "{kind:?}: height should never drop below base {base_value}, got min {min}"
            );
        }
    }

    #[test]
    fn from_graph_round_trips() {
        let mut params = std::collections::BTreeMap::new();
        params.insert(
            "kind".to_string(),
            mc_core::ParamValue::Enum("Hex Bolt".to_string()),
        );
        params.insert("spacing".to_string(), mc_core::ParamValue::Float(0.1));
        params.insert("head_radius".to_string(), mc_core::ParamValue::Float(0.02));
        params.insert("head_height".to_string(), mc_core::ParamValue::Float(0.5));
        params.insert("seed".to_string(), mc_core::ParamValue::Seed(9));
        params.insert("missing".to_string(), mc_core::ParamValue::Float(0.1));
        params.insert("rotate".to_string(), mc_core::ParamValue::Bool(false));
        let parsed = FastenersParams::from_graph(&params);
        assert_eq!(parsed.kind, FastenerKind::HexBolt);
        assert_eq!(parsed.spacing, 0.1);
        assert_eq!(parsed.head_radius, 0.02);
        assert_eq!(parsed.head_height, 0.5);
        assert_eq!(parsed.seed, 9);
        assert_eq!(parsed.missing, 0.1);
        assert!(!parsed.rotate);

        let defaults = FastenersParams::from_graph(&std::collections::BTreeMap::new());
        assert_eq!(defaults.kind, FastenerKind::RivetDome);
        assert!(defaults.rotate);
    }
}
