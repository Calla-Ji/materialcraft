// SPDX-License-Identifier: MIT OR Apache-2.0
//! Converters / Redistance.
//!
//! `redistance(sdf) = distance_field(coverage(sdf))`
//! (`docs/hardsurface_nodes/HARDSURFACE_NODES.md` §3.2) — rebuilds an exact
//! field from any SDF's own zero contour. Booleans built from min/max (and
//! warps) keep their inputs' hidden interior edges, which a later Bevel
//! would otherwise show as a false groove where two shapes meet; this
//! fixes that for any SDF, not just Shape Boolean's own `Exact` option
//! (which does the same thing internally, just automatically).
//!
//! **CPU-only**, same deliberate, temporary exception Distance Field
//! itself is — see `distance_field::dispatch`'s own doc comment for the
//! full reasoning. This node does no new math of its own: it's a thin
//! wrapper reusing `distance_field::coverage`/`compute_distance_field`
//! directly.

use std::collections::BTreeMap;

use mc_core::ParamValue;

use crate::distance_field::{compute_distance_field, coverage, DistanceFieldParams};

pub struct RedistanceParams {
    pub wrap: bool,
}

impl Default for RedistanceParams {
    fn default() -> Self {
        Self { wrap: true }
    }
}

impl RedistanceParams {
    pub fn from_graph(params: &BTreeMap<String, ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            wrap: params
                .get("wrap")
                .and_then(ParamValue::as_bool)
                .unwrap_or(defaults.wrap),
        }
    }
}

pub fn dispatch(
    gpu: &mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    sdf_key: &str,
    output_key: &str,
    resolution: u32,
    params: &RedistanceParams,
) {
    let texture = pool
        .texture(sdf_key)
        .unwrap_or_else(|| panic!("redistance: no texture at sdf key {sdf_key:?}"));
    // `Sdf` sockets are `Rgba32Float` (HARDSURFACE_NODES.md §2) — see
    // `distance_field::dispatch`'s own output write for why.
    let raw = gpu.read_rgba32f(texture, resolution);
    let res_f = resolution as f32;
    let mask: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| coverage(p[0], res_f))
        .collect();

    let sdf = compute_distance_field(
        &mask,
        resolution as usize,
        &DistanceFieldParams {
            wrap: params.wrap,
            threshold: 0.5,
            max_distance: 0.0,
        },
    );

    let pixels: Vec<[f32; 4]> = sdf.iter().map(|&d| [d, d, d, 1.0]).collect();
    gpu.write_rgba32f(pool, output_key, resolution, &pixels);
}

#[cfg(test)]
mod tests {
    use super::{dispatch, RedistanceParams};
    use crate::test_support::{golden_path, read_npy_f32_2d};

    const RES: u32 = 512;

    /// `redistance_union.npy` = `redistance(shape_boolean(a, b, "union",
    /// "hard"))` — build the hard union directly (plain `min(a,b)`, no need
    /// for the real Shape Boolean node to test this node in isolation),
    /// redistance it, and compare within Distance Field's own px-based
    /// tolerance.
    #[test]
    fn matches_the_redistance_union_golden() {
        let gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let a = read_npy_f32_2d(&golden_path("hardsurface", "shape_a.npy"));
        let b = read_npy_f32_2d(&golden_path("hardsurface", "shape_b.npy"));
        let union: Vec<[f32; 4]> = a
            .iter()
            .zip(b.iter())
            .map(|(&av, &bv)| {
                let m = av.min(bv);
                [m, m, m, 1.0]
            })
            .collect();
        gpu.write_rgba32f(&mut pool, "union", RES, &union);

        dispatch(
            &gpu,
            &mut pool,
            "union",
            "redistanced",
            RES,
            &RedistanceParams::default(),
        );
        let out: Vec<f32> = gpu
            .read_rgba32f(pool.texture("redistanced").unwrap(), RES)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| p[0])
            .collect();

        let want = read_npy_f32_2d(&golden_path("hardsurface", "redistance_union.npy"));
        let mut errs: Vec<f64> = out
            .iter()
            .zip(want.iter())
            .map(|(&g, &w)| ((g - w) as f64 * RES as f64).abs())
            .collect();
        let mean = errs.iter().sum::<f64>() / errs.len() as f64;
        errs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let p99 = errs[(errs.len() as f64 * 0.99) as usize];
        assert!(mean < 0.05, "mean error {mean} px");
        assert!(p99 < 0.3, "p99 error {p99} px");
    }

    #[test]
    fn from_graph_round_trips() {
        let mut params = std::collections::BTreeMap::new();
        params.insert("wrap".to_string(), mc_core::ParamValue::Bool(false));
        let parsed = RedistanceParams::from_graph(&params);
        assert!(!parsed.wrap);

        let defaults = RedistanceParams::from_graph(&std::collections::BTreeMap::new());
        assert!(defaults.wrap);
    }
}
