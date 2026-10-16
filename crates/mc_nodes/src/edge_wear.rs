// SPDX-License-Identifier: MIT OR Apache-2.0
//! Converters / Edge Wear.
//!
//! Paints chipping on convex edges and a cavity mask for dirt/recesses
//! (`docs/hardsurface_nodes/HARDSURFACE_NODES.md` §3.8) — from an `Sdf`
//! (preferred, exact) or, if none is connected, from any `height` map
//! via a Laplacian-of-Gaussian curvature estimate.
//!
//! **CPU-only, deliberately**: the SDF branch alone is pure per-pixel
//! math and could be a GPU shader, but the height-map fallback needs
//! `gaussian_laplace` (a real neighbor-dependent convolution) *and* a
//! 99th-percentile normalization over the *entire* image before any
//! pixel's final value is known — a global reduction a uniform compute
//! shader can't do in one pass. Rather than splitting this node across
//! two different dispatch architectures depending on which input is
//! connected, the whole node stays CPU — same justified-exception
//! category as Distance Field/Fasteners, see their own doc comments.

use crate::stitch::hash01;

#[derive(Clone, Copy)]
enum Axis {
    X,
    Y,
}

/// Tileable fBm value noise (bilinear lattice value noise, smoothstep-
/// interpolated — distinct from this codebase's existing gradient-based
/// Perlin noise). Not a graph node of its own: `docs/hardsurface_nodes/
/// HARDSURFACE_NODES.md`'s own node list has no entry for it, it only
/// exists as Edge Wear's own chip pattern. Ported from `hs_reference.py`'s
/// `value_noise`.
fn value_noise(resolution: u32, cells: u32, seed: u32, octaves: u32, gain: f32) -> Vec<f32> {
    let res = resolution as usize;
    let res_f = resolution as f32;
    let mut total = vec![0.0f32; res * res];
    let mut amp = 1.0f32;
    let mut norm = 0.0f32;

    for o in 0..octaves {
        let n = cells.wrapping_mul(1u32 << o);
        let n_f = n as f32;
        let n_i = i64::from(n);
        // Rust panics on integer overflow in debug builds where Python's
        // own masked hash wouldn't — wrapping ops only, to match.
        let salt = seed.wrapping_mul(131).wrapping_add(o);

        for y in 0..res {
            let gy = (y as f32 + 0.5) / res_f * n_f;
            let iy = gy.floor() as i64;
            let fy = gy - iy as f32;
            let uy = fy * fy * (3.0 - 2.0 * fy);

            for x in 0..res {
                let gx = (x as f32 + 0.5) / res_f * n_f;
                let ix = gx.floor() as i64;
                let fx = gx - ix as f32;
                let ux = fx * fx * (3.0 - 2.0 * fx);

                let v = |dx: i64, dy: i64| -> f32 {
                    let xi = (ix + dx).rem_euclid(n_i) as u32;
                    let yi = (iy + dy).rem_euclid(n_i) as u32;
                    hash01(xi, yi, salt) as f32
                };
                let v00 = v(0, 0);
                let v10 = v(1, 0);
                let v01 = v(0, 1);
                let v11 = v(1, 1);
                let a = v00 + (v10 - v00) * ux;
                let b = v01 + (v11 - v01) * ux;
                total[y * res + x] += amp * (a + (b - a) * uy);
            }
        }
        norm += amp;
        amp *= gain;
    }

    total.iter().map(|&t| t / norm).collect()
}

/// `scipy.ndimage`'s own 1D Gaussian derivative kernel (`order` 0 or 2
/// only — nothing else is ever called here), discretely renormalized
/// (`sum = 1` for order 0) before the order-2 polynomial weight
/// `(x^2 - sigma^2) / sigma^4` is applied — hand-verified by expanding
/// scipy's own recursive polynomial construction for `order = 2`, not
/// guessed; it's the standard closed-form 1D "Mexican hat" kernel.
fn gaussian_kernel1d(sigma: f32, order: u32, radius: i32) -> Vec<f32> {
    let sigma2 = sigma * sigma;
    let phi: Vec<f32> = (-radius..=radius)
        .map(|x| (-0.5 / sigma2 * (x * x) as f32).exp())
        .collect();
    let sum: f32 = phi.iter().sum();
    let phi: Vec<f32> = phi.iter().map(|&p| p / sum).collect();

    if order == 0 {
        return phi;
    }
    (-radius..=radius)
        .zip(phi.iter())
        .map(|(x, &p)| {
            let xf = x as f32;
            (xf * xf - sigma2) / (sigma2 * sigma2) * p
        })
        .collect()
}

/// One 1D correlation pass along a row or column, wrap-indexed. Safe to
/// treat as a convolution with no kernel flip: every kernel used here
/// (`gaussian_kernel1d`'s order-0 and order-2 outputs) is symmetric.
fn convolve1d_wrap(data: &[f32], resolution: usize, axis: Axis, kernel: &[f32]) -> Vec<f32> {
    let radius = (kernel.len() / 2) as i64;
    let res_i = resolution as i64;
    let mut out = vec![0.0f32; data.len()];
    for y in 0..resolution {
        for x in 0..resolution {
            let mut acc = 0.0f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let d = k as i64 - radius;
                let (xx, yy) = match axis {
                    Axis::X => ((x as i64 + d).rem_euclid(res_i) as usize, y),
                    Axis::Y => (x, (y as i64 + d).rem_euclid(res_i) as usize),
                };
                acc += kv * data[yy * resolution + xx];
            }
            out[y * resolution + x] = acc;
        }
    }
    out
}

/// `scipy.ndimage.gaussian_laplace(data, sigma, mode="wrap")` — sums the
/// two separable passes (order-2 along one axis, order-0 along the
/// other) `generic_laplace` combines. `truncate = 4.0` is scipy's own
/// default.
fn gaussian_laplace(data: &[f32], resolution: usize, sigma: f32) -> Vec<f32> {
    let radius = (4.0 * sigma + 0.5) as i32;
    let k0 = gaussian_kernel1d(sigma, 0, radius);
    let k2 = gaussian_kernel1d(sigma, 2, radius);

    let pass_y2 = convolve1d_wrap(data, resolution, Axis::Y, &k2);
    let term_y2x0 = convolve1d_wrap(&pass_y2, resolution, Axis::X, &k0);

    let pass_y0 = convolve1d_wrap(data, resolution, Axis::Y, &k0);
    let term_y0x2 = convolve1d_wrap(&pass_y0, resolution, Axis::X, &k2);

    term_y2x0
        .iter()
        .zip(term_y0x2.iter())
        .map(|(&a, &b)| a + b)
        .collect()
}

/// `numpy.percentile`'s own default ("linear") interpolation method —
/// not a naive bucket count, which would disagree with numpy whenever
/// `0.99 * (n - 1)` isn't a whole number.
fn percentile99(values: &[f32]) -> f32 {
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = 0.99 * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = (idx - lo as f64) as f32;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub struct EdgeWearParams {
    /// SDF branch only, tile units.
    pub band: f32,
    pub amount: f32,
    /// SDF branch only.
    pub falloff: f32,
    pub chip_cells: u32,
    pub softness: f32,
    pub seed: u32,
    /// SDF branch only, tile units.
    pub cavity_width: f32,
    /// Height-map fallback only, pixels.
    pub curvature_sigma: f32,
    pub chip_depth: f32,
}

impl Default for EdgeWearParams {
    fn default() -> Self {
        Self {
            band: 0.012,
            amount: 0.5,
            falloff: 1.5,
            chip_cells: 24,
            softness: 0.08,
            seed: 7,
            cavity_width: 0.006,
            curvature_sigma: 1.5,
            chip_depth: 0.04,
        }
    }
}

impl EdgeWearParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            band: params
                .get("band")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.band),
            amount: params
                .get("amount")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.amount),
            falloff: params
                .get("falloff")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.falloff),
            chip_cells: params
                .get("chip_cells")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.chip_cells),
            softness: params
                .get("softness")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.softness),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
            cavity_width: params
                .get("cavity_width")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.cavity_width),
            curvature_sigma: params
                .get("curvature_sigma")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.curvature_sigma),
            chip_depth: params
                .get("chip_depth")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.chip_depth),
        }
    }
}

pub struct EdgeWearOutputKeys<'a> {
    pub wear: &'a str,
    pub cavity: &'a str,
    pub height: &'a str,
}

fn to_pixels(values: &[f32]) -> Vec<[f32; 4]> {
    values.iter().map(|&v| [v, v, v, 1.0]).collect()
}

pub fn dispatch(
    gpu: &mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    sdf_key: Option<&str>,
    height_key: Option<&str>,
    keys: EdgeWearOutputKeys,
    resolution: u32,
    params: &EdgeWearParams,
) {
    let res = resolution as usize;
    let res_f = resolution as f32;

    let sdf: Option<Vec<f32>> = sdf_key.map(|key| {
        let texture = pool
            .texture(key)
            .unwrap_or_else(|| panic!("edge_wear: no texture at sdf key {key:?}"));
        gpu.read_rgba32f(texture, resolution)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| p[0])
            .collect()
    });
    let height: Option<Vec<f32>> = height_key.map(|key| {
        let texture = pool
            .texture(key)
            .unwrap_or_else(|| panic!("edge_wear: no texture at height key {key:?}"));
        gpu.read_rgba16f(texture, resolution)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| p[0])
            .collect()
    });

    let noise = value_noise(resolution, params.chip_cells, params.seed, 4, 0.5);

    let (wear, cavity) = if let Some(sdf) = &sdf {
        let mut wear = vec![0.0f32; res * res];
        let mut cavity = vec![0.0f32; res * res];
        for i in 0..res * res {
            let s = sdf[i];
            let t = (-s / params.band).clamp(0.0, 1.0);
            let e = (1.0 - t).powf(params.falloff);
            let inside = crate::distance_field::coverage(s, res_f);
            let sig = 2.0 * params.amount * e - noise[i];
            wear[i] = smoothstep(0.0, params.softness, sig) * inside;
            cavity[i] = smoothstep(params.cavity_width, 0.0, s) * (1.0 - inside);
        }
        (wear, cavity)
    } else if let Some(height) = &height {
        let lap = gaussian_laplace(height, res, params.curvature_sigma);
        let conv: Vec<f32> = lap.iter().map(|&l| (-l).max(0.0)).collect();
        let conc: Vec<f32> = lap.iter().map(|&l| l.max(0.0)).collect();
        let p99_conv = percentile99(&conv) + 1e-12;
        let p99_conc = percentile99(&conc) + 1e-12;
        let mut wear = vec![0.0f32; res * res];
        let mut cavity = vec![0.0f32; res * res];
        for i in 0..res * res {
            let e = (conv[i] / p99_conv).clamp(0.0, 1.0);
            let sig = 2.0 * params.amount * e - noise[i];
            wear[i] = smoothstep(0.0, params.softness, sig);
            cavity[i] = (conc[i] / p99_conc).clamp(0.0, 1.0);
        }
        (wear, cavity)
    } else {
        (vec![0.0f32; res * res], vec![0.0f32; res * res])
    };

    let chipped_height: Vec<f32> = match &height {
        Some(h) => h
            .iter()
            .zip(wear.iter())
            .map(|(&hv, &w)| hv - params.chip_depth * w)
            .collect(),
        None => vec![0.0f32; res * res],
    };

    gpu.write_rgba16f(pool, keys.wear, resolution, &to_pixels(&wear));
    gpu.write_rgba16f(pool, keys.cavity, resolution, &to_pixels(&cavity));
    gpu.write_rgba16f(pool, keys.height, resolution, &to_pixels(&chipped_height));
}

#[cfg(test)]
mod tests {
    use super::{dispatch, gaussian_laplace, value_noise, EdgeWearOutputKeys, EdgeWearParams};
    use crate::test_support::{golden_path, read_png16_as_unit_floats};

    const RES: u32 = 512;

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
    fn value_noise_matches_the_golden() {
        let got = value_noise(RES, 24, 7, 4, 0.5);
        let (want, _) =
            read_png16_as_unit_floats(&golden_path("hardsurface", "value_noise_c24_s7.png"));
        assert_matches_golden(&got, &want, "value_noise");
    }

    struct Outputs {
        wear: Vec<f32>,
        cavity: Vec<f32>,
        height: Vec<f32>,
    }

    fn run(
        gpu: &mc_gpu::Context,
        pool: &mut mc_gpu::TexturePool,
        sdf_key: Option<&str>,
        height_key: Option<&str>,
        params: &EdgeWearParams,
    ) -> Outputs {
        dispatch(
            gpu,
            pool,
            sdf_key,
            height_key,
            EdgeWearOutputKeys {
                wear: "wear",
                cavity: "cavity",
                height: "height_out",
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
            wear: read("wear"),
            cavity: read("cavity"),
            height: read("height_out"),
        }
    }

    #[test]
    fn sdf_branch_matches_the_goldens() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::panel_layout::dispatch(
            &mut gpu,
            &mut pool,
            crate::panel_layout::PanelLayoutOutputKeys {
                sdf: "panel_sdf",
                panel_id: "panel_id",
                panel_index: "panel_index",
                uv_u: "uv_u",
                uv_v: "uv_v",
                seam: "seam",
            },
            RES,
            &crate::panel_layout::PanelLayoutParams::default(),
        );

        let out = run(
            &gpu,
            &mut pool,
            Some("panel_sdf"),
            None,
            &EdgeWearParams::default(),
        );
        let (want_wear, _) =
            read_png16_as_unit_floats(&golden_path("hardsurface", "wear_from_sdf.png"));
        assert_matches_golden(&out.wear, &want_wear, "wear_from_sdf");
        let (want_cavity, _) =
            read_png16_as_unit_floats(&golden_path("hardsurface", "cavity_from_sdf.png"));
        assert_matches_golden(&out.cavity, &want_cavity, "cavity_from_sdf");
    }

    #[test]
    fn height_branch_matches_the_golden() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::panel_layout::dispatch(
            &mut gpu,
            &mut pool,
            crate::panel_layout::PanelLayoutOutputKeys {
                sdf: "panel_sdf",
                panel_id: "panel_id",
                panel_index: "panel_index",
                uv_u: "uv_u",
                uv_v: "uv_v",
                seam: "seam",
            },
            RES,
            &crate::panel_layout::PanelLayoutParams::default(),
        );
        // `wear_from_height.png` = edge_wear(height = bevel_height * 0.5).
        // `height` scales linearly with `depth` (base = 0), so Depth: 0.5
        // reproduces that exact array directly, no separate scale step.
        crate::bevel_profile::dispatch(
            &mut gpu,
            &mut pool,
            "panel_sdf",
            crate::bevel_profile::BevelProfileOutputKeys {
                height: "bevel_height",
                edge: "bevel_edge",
                top: "bevel_top",
                u: "bevel_u",
            },
            RES,
            &crate::bevel_profile::BevelProfileParams {
                preset: crate::bevel_profile::BevelPreset::Round,
                width: 0.006,
                wall: 0.35,
                depth: 0.5,
                base: 0.0,
                inward: true,
                ..crate::bevel_profile::BevelProfileParams::default()
            },
        );

        let out = run(
            &gpu,
            &mut pool,
            None,
            Some("bevel_height"),
            &EdgeWearParams::default(),
        );
        let (want_wear, _) =
            read_png16_as_unit_floats(&golden_path("hardsurface", "wear_from_height.png"));
        assert_matches_golden(&out.wear, &want_wear, "wear_from_height");
    }

    #[test]
    fn height_output_is_blank_unless_a_height_input_exists() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::panel_layout::dispatch(
            &mut gpu,
            &mut pool,
            crate::panel_layout::PanelLayoutOutputKeys {
                sdf: "panel_sdf",
                panel_id: "panel_id",
                panel_index: "panel_index",
                uv_u: "uv_u",
                uv_v: "uv_v",
                seam: "seam",
            },
            RES,
            &crate::panel_layout::PanelLayoutParams::default(),
        );
        let out = run(
            &gpu,
            &mut pool,
            Some("panel_sdf"),
            None,
            &EdgeWearParams::default(),
        );
        assert!(
            out.height.iter().all(|&v| v == 0.0),
            "no height input should leave the Height output blank"
        );

        gpu.write_rgba16f(
            &mut pool,
            "flat_height",
            RES,
            &vec![[0.2, 0.2, 0.2, 1.0]; (RES * RES) as usize],
        );
        let out = run(
            &gpu,
            &mut pool,
            Some("panel_sdf"),
            Some("flat_height"),
            &EdgeWearParams::default(),
        );
        let differs = out
            .height
            .iter()
            .zip(std::iter::repeat(0.2f32))
            .any(|(&h, base)| (h - base).abs() > 1e-6);
        assert!(
            differs,
            "with a height input and nonzero wear somewhere, chipped height should differ from the raw input"
        );
    }

    /// A raised bump (a local height maximum) is a convex edge — its
    /// Laplacian should be negative at the peak, so `conv = max(-lap, 0)`
    /// should be clearly positive there and near-zero far away. Checked
    /// directly before trusting the full pipeline's golden match alone to
    /// catch a flipped sign.
    #[test]
    fn gaussian_laplace_is_negative_at_a_convex_peak() {
        const N: usize = 64;
        let mut height = vec![0.0f32; N * N];
        let (cx, cy, sigma) = (32.0f32, 32.0f32, 6.0f32);
        for y in 0..N {
            for x in 0..N {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                height[y * N + x] = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
            }
        }
        let lap = gaussian_laplace(&height, N, 1.5);
        let center = lap[32 * N + 32];
        let corner = lap[2 * N + 2];
        assert!(
            center < 0.0,
            "a bump's peak should have negative Laplacian, got {center}"
        );
        assert!(
            (-center) > (-corner).max(0.0) + 0.01,
            "the peak's convexity should be clearly stronger than the flat corner's \
             (center={center}, corner={corner})"
        );
    }

    #[test]
    fn from_graph_round_trips() {
        let mut params = std::collections::BTreeMap::new();
        params.insert("band".to_string(), mc_core::ParamValue::Float(0.02));
        params.insert("amount".to_string(), mc_core::ParamValue::Float(0.8));
        params.insert("falloff".to_string(), mc_core::ParamValue::Float(2.0));
        params.insert("chip_cells".to_string(), mc_core::ParamValue::Int(12));
        params.insert("softness".to_string(), mc_core::ParamValue::Float(0.1));
        params.insert("seed".to_string(), mc_core::ParamValue::Seed(3));
        params.insert("cavity_width".to_string(), mc_core::ParamValue::Float(0.01));
        params.insert(
            "curvature_sigma".to_string(),
            mc_core::ParamValue::Float(2.0),
        );
        params.insert("chip_depth".to_string(), mc_core::ParamValue::Float(0.1));
        let parsed = EdgeWearParams::from_graph(&params);
        assert_eq!(parsed.band, 0.02);
        assert_eq!(parsed.amount, 0.8);
        assert_eq!(parsed.falloff, 2.0);
        assert_eq!(parsed.chip_cells, 12);
        assert_eq!(parsed.softness, 0.1);
        assert_eq!(parsed.seed, 3);
        assert_eq!(parsed.cavity_width, 0.01);
        assert_eq!(parsed.curvature_sigma, 2.0);
        assert_eq!(parsed.chip_depth, 0.1);

        let defaults = EdgeWearParams::from_graph(&std::collections::BTreeMap::new());
        assert_eq!(defaults.chip_cells, 24);
        assert_eq!(defaults.seed, 7);
    }
}
