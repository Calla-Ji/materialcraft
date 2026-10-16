// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Panel Layout.
//!
//! A tileable guillotine subdivision (`docs/hardsurface_nodes/
//! HARDSURFACE_NODES.md` §3.5) — no flood fill, so none of the seams or
//! tiling workarounds Substance artists complain about. Every pixel on the
//! torus belongs to exactly one panel, found directly from a recursive
//! rect split rather than painted in.
//!
//! Ported from `hs_reference.py`'s `PanelParams`/`panel_rects`/
//! `sd_round_box`/`panel_layout`. The spec itself is explicit that
//! `panel_rects` must match its own JSON goldens *exactly* (integer and
//! hash logic) before the rasters are trusted at all — this module keeps
//! that split: [`panel_rects`] is pure `f64` CPU math with no GPU
//! involved, verified on its own against 5 seeds, and [`dispatch`] (the
//! GPU rasterization) is a separate, later concern built on top of it.

use std::collections::BTreeMap;

use mc_core::{ParamValue, Path, Point, SubPath};

use crate::shaders::full_source;
use crate::stitch::{hash01, round_half_even};

const NODE_KIND: &str = "panel_layout";

pub struct PanelLayoutParams {
    pub seed: u32,
    pub grid_div: u32,
    pub max_depth: u32,
    pub min_depth: u32,
    pub offset: bool,
    /// Geometry-sensitive values kept as `f64`, not `f32`: `panel_rects`'s
    /// own hash-threshold comparisons (`hash01(...) < split_chance`) are
    /// discontinuous, so rounding a value through `f32` first (as every
    /// other `Float` param in this codebase does, via `ParamValue`) risks
    /// a different split decision than the Python reference's own
    /// `float64` arithmetic at the same default value. `from_graph` still
    /// goes through the usual `f32` UI param — a real but inconsequential
    /// precision loss for an artist's slider; tests construct this struct
    /// directly from the golden JSON's own `f64`s instead, bypassing that
    /// path entirely, for real bit-level reproducibility.
    pub min_size: f64,
    pub split_chance: f64,
    pub long_axis_bias: f64,
    pub gap: f64,
    pub corner_radius: f64,
    pub fastener_margin: f64,
}

impl Default for PanelLayoutParams {
    fn default() -> Self {
        Self {
            seed: 3,
            grid_div: 16,
            max_depth: 6,
            min_depth: 3,
            offset: true,
            min_size: 0.125,
            split_chance: 0.6,
            long_axis_bias: 0.8,
            gap: 0.006,
            corner_radius: 0.004,
            fastener_margin: 0.014,
        }
    }
}

impl PanelLayoutParams {
    pub fn from_graph(params: &BTreeMap<String, ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            seed: params
                .get("seed")
                .and_then(ParamValue::as_seed)
                .unwrap_or(defaults.seed),
            grid_div: params
                .get("grid_div")
                .and_then(ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.grid_div),
            max_depth: params
                .get("max_depth")
                .and_then(ParamValue::as_int)
                .map(|v| v.max(0) as u32)
                .unwrap_or(defaults.max_depth),
            min_depth: params
                .get("min_depth")
                .and_then(ParamValue::as_int)
                .map(|v| v.max(0) as u32)
                .unwrap_or(defaults.min_depth),
            offset: params
                .get("offset")
                .and_then(ParamValue::as_bool)
                .unwrap_or(defaults.offset),
            min_size: params
                .get("min_size")
                .and_then(ParamValue::as_float)
                .map(f64::from)
                .unwrap_or(defaults.min_size),
            split_chance: params
                .get("split_chance")
                .and_then(ParamValue::as_float)
                .map(f64::from)
                .unwrap_or(defaults.split_chance),
            long_axis_bias: params
                .get("long_axis_bias")
                .and_then(ParamValue::as_float)
                .map(f64::from)
                .unwrap_or(defaults.long_axis_bias),
            gap: params
                .get("gap")
                .and_then(ParamValue::as_float)
                .map(f64::from)
                .unwrap_or(defaults.gap),
            corner_radius: params
                .get("corner_radius")
                .and_then(ParamValue::as_float)
                .map(f64::from)
                .unwrap_or(defaults.corner_radius),
            fastener_margin: params
                .get("fastener_margin")
                .and_then(ParamValue::as_float)
                .map(f64::from)
                .unwrap_or(defaults.fastener_margin),
        }
    }
}

/// One leaf of the guillotine subdivision, tile units — may extend past
/// `[0, 1)`; pixels are assigned with modular local coordinates, so the
/// layout always tiles (see [`dispatch`]'s own rasterization).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PanelRect {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
    pub index: usize,
}

/// Recursive guillotine subdivision covering the torus exactly once —
/// ported verbatim from `hs_reference.py`'s `panel_rects` (the module doc
/// comment explains why this stays `f64` throughout).
pub(crate) fn panel_rects(params: &PanelLayoutParams) -> Vec<PanelRect> {
    let g = 1.0 / f64::from(params.grid_div);
    let (ox, oy) = if params.offset {
        let ox = (hash01(params.seed, 1, 0) * f64::from(params.grid_div)).floor() * g;
        let oy = (hash01(params.seed, 2, 0) * f64::from(params.grid_div)).floor() * g;
        (ox, oy)
    } else {
        (0.0, 0.0)
    };
    let mut out = Vec::new();
    subdivide(ox, oy, ox + 1.0, oy + 1.0, 0, 0, params, g, &mut out);
    out
}

#[allow(clippy::too_many_arguments)]
fn subdivide(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    depth: u32,
    path_id: u32,
    params: &PanelLayoutParams,
    g: f64,
    out: &mut Vec<PanelRect>,
) {
    let w = x1 - x0;
    let h = y1 - y0;
    let k = path_id;
    let can_x = w >= 2.0 * params.min_size - 1e-9;
    let can_y = h >= 2.0 * params.min_size - 1e-9;
    let want = depth < params.min_depth || hash01(params.seed, k, 3) < params.split_chance;
    if depth >= params.max_depth || !(can_x || can_y) || !want {
        let index = out.len();
        out.push(PanelRect {
            x0,
            y0,
            x1,
            y1,
            index,
        });
        return;
    }

    let longer_x = w >= h;
    let pick_long = hash01(params.seed, k, 4) < params.long_axis_bias;
    let mut split_x = if pick_long { longer_x } else { !longer_x };
    if split_x && !can_x {
        split_x = false;
    }
    if !split_x && !can_y {
        split_x = true;
    }
    let (lo, hi) = if split_x { (x0, x1) } else { (y0, y1) };
    let t = 0.3 + 0.4 * hash01(params.seed, k, 5);
    let mut cut = lo + round_half_even((t * (hi - lo)) / g) * g;
    cut = cut.max(lo + params.min_size).min(hi - params.min_size);
    cut = lo + round_half_even((cut - lo) / g) * g;
    if cut - lo < params.min_size - 1e-9 || hi - cut < params.min_size - 1e-9 {
        let index = out.len();
        out.push(PanelRect {
            x0,
            y0,
            x1,
            y1,
            index,
        });
        return;
    }

    if split_x {
        subdivide(x0, y0, cut, y1, depth + 1, 2 * k + 1, params, g, out);
        subdivide(cut, y0, x1, y1, depth + 1, 2 * k + 2, params, g, out);
    } else {
        subdivide(x0, y0, x1, cut, depth + 1, 2 * k + 1, params, g, out);
        subdivide(x0, cut, x1, y1, depth + 1, 2 * k + 2, params, g, out);
    }
}

/// The fastener path: one closed, 4-point corner subpath per rect wide
/// and tall enough to survive the `gap/2 + fastener_margin` inset —
/// ported from `panel_layout`'s own `subpaths` loop. [`Point::new`]'s
/// defaults (zero handles, `Corner` kind) already match the reference's
/// `Point((x,y),(0,0),(0,0),"corner")` exactly.
pub(crate) fn fastener_path(rects: &[PanelRect], params: &PanelLayoutParams) -> Path {
    let m = params.gap / 2.0 + params.fastener_margin;
    let mut subpaths = Vec::new();
    for rect in rects {
        if (rect.x1 - rect.x0) > 2.0 * m && (rect.y1 - rect.y0) > 2.0 * m {
            let points = vec![
                Point::new([rect.x0 + m, rect.y0 + m]),
                Point::new([rect.x1 - m, rect.y0 + m]),
                Point::new([rect.x1 - m, rect.y1 - m]),
                Point::new([rect.x0 + m, rect.y1 - m]),
            ];
            subpaths.push(SubPath::new(points, true));
        }
    }
    Path {
        subpaths,
        wrap: true,
    }
}

pub struct PanelLayoutOutputKeys<'a> {
    pub sdf: &'a str,
    pub panel_id: &'a str,
    pub panel_index: &'a str,
    pub uv_u: &'a str,
    pub uv_v: &'a str,
    pub seam: &'a str,
}

/// Rasterizes `panel_rects(params)` into `keys`' six textures and returns
/// the fastener [`Path`] — the caller (`registry.rs`) is the one that
/// knows how to wire a `Path`-typed output into the graph (the established
/// `computed_path_result` pattern Image Trace/Path Simplify already use);
/// this function only computes the value.
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    keys: PanelLayoutOutputKeys,
    resolution: u32,
    params: &PanelLayoutParams,
) -> Path {
    let rects = panel_rects(params);

    let shader_source = full_source("panel_layout.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, rects.len() as u32, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("panel_layout_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let rect_bytes = pack_rects(&rects);
    let rect_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("panel_layout_rects"),
        size: rect_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&rect_buffer, 0, &rect_bytes);

    let sdf_view = pool
        .get_or_create(
            &gpu.device,
            keys.sdf,
            resolution,
            wgpu::TextureFormat::Rgba32Float,
        )
        .clone();
    let panel_id_view = pool
        .get_or_create(
            &gpu.device,
            keys.panel_id,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let panel_index_view = pool
        .get_or_create(
            &gpu.device,
            keys.panel_index,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let uv_u_view = pool
        .get_or_create(
            &gpu.device,
            keys.uv_u,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let uv_v_view = pool
        .get_or_create(
            &gpu.device,
            keys.uv_v,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let seam_view = pool
        .get_or_create(
            &gpu.device,
            keys.seam,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("panel_layout_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rect_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&sdf_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&panel_id_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&panel_index_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&uv_u_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&uv_v_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&seam_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("panel_layout_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("panel_layout_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));

    fastener_path(&rects, params)
}

fn pack_rects(rects: &[PanelRect]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rects.len() * 20);
    for r in rects {
        bytes.extend_from_slice(&(r.x0 as f32).to_le_bytes());
        bytes.extend_from_slice(&(r.y0 as f32).to_le_bytes());
        bytes.extend_from_slice(&(r.x1 as f32).to_le_bytes());
        bytes.extend_from_slice(&(r.y1 as f32).to_le_bytes());
        bytes.extend_from_slice(&(r.index as f32).to_le_bytes());
    }
    bytes
}

fn pack_params(resolution: u32, rect_count: u32, params: &PanelLayoutParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&rect_count.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.seed.to_le_bytes());
    bytes[12..16].copy_from_slice(&(params.gap as f32).to_le_bytes());
    bytes[16..20].copy_from_slice(&(params.corner_radius as f32).to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, fastener_path, panel_rects, PanelLayoutOutputKeys, PanelLayoutParams};
    use crate::stitch::hash01;
    use crate::test_support::{golden_path, read_npy_f32_2d, read_png16_as_unit_floats};

    const RES: u32 = 512;

    fn params_from_json(v: &serde_json::Value) -> PanelLayoutParams {
        let p = &v["params"];
        PanelLayoutParams {
            seed: p["seed"].as_u64().unwrap() as u32,
            grid_div: p["grid_div"].as_u64().unwrap() as u32,
            max_depth: p["max_depth"].as_u64().unwrap() as u32,
            min_depth: p["min_depth"].as_u64().unwrap() as u32,
            offset: p["offset"].as_bool().unwrap(),
            min_size: p["min_size"].as_f64().unwrap(),
            split_chance: p["split_chance"].as_f64().unwrap(),
            long_axis_bias: p["long_axis_bias"].as_f64().unwrap(),
            gap: p["gap"].as_f64().unwrap(),
            corner_radius: p["corner_radius"].as_f64().unwrap(),
            fastener_margin: p["fastener_margin"].as_f64().unwrap(),
        }
    }

    fn check_against_golden(file_name: &str) {
        let text = std::fs::read_to_string(golden_path("hardsurface", file_name))
            .unwrap_or_else(|err| panic!("failed to read {file_name}: {err}"));
        let golden: serde_json::Value = serde_json::from_str(&text).unwrap();
        let params = params_from_json(&golden);
        let rects = panel_rects(&params);

        let want = golden["rects"].as_array().unwrap();
        assert_eq!(rects.len(), want.len(), "{file_name}: rect count mismatch");
        for (i, (got, want)) in rects.iter().zip(want.iter()).enumerate() {
            let want = want.as_array().unwrap();
            let want_x0 = want[0].as_f64().unwrap();
            let want_y0 = want[1].as_f64().unwrap();
            let want_x1 = want[2].as_f64().unwrap();
            let want_y1 = want[3].as_f64().unwrap();
            let want_index = want[4].as_u64().unwrap() as usize;
            assert!(
                (got.x0 - want_x0).abs() < 1e-6,
                "{file_name} rect {i}: x0 {} vs {want_x0}",
                got.x0
            );
            assert!(
                (got.y0 - want_y0).abs() < 1e-6,
                "{file_name} rect {i}: y0 {} vs {want_y0}",
                got.y0
            );
            assert!(
                (got.x1 - want_x1).abs() < 1e-6,
                "{file_name} rect {i}: x1 {} vs {want_x1}",
                got.x1
            );
            assert!(
                (got.y1 - want_y1).abs() < 1e-6,
                "{file_name} rect {i}: y1 {} vs {want_y1}",
                got.y1
            );
            assert_eq!(got.index, want_index, "{file_name} rect {i}: index");
        }
    }

    #[test]
    fn matches_the_default_seed_golden() {
        check_against_golden("panel_rects.json");
    }

    #[test]
    fn matches_every_other_seed_golden() {
        for file_name in [
            "panel_rects_seed1.json",
            "panel_rects_seed2.json",
            "panel_rects_seed4.json",
            "panel_rects_seed5.json",
        ] {
            check_against_golden(file_name);
        }
    }

    struct Outputs {
        sdf: Vec<f32>,
        panel_id: Vec<f32>,
        panel_index: Vec<f32>,
        uv_u: Vec<f32>,
        uv_v: Vec<f32>,
        seam: Vec<f32>,
    }

    fn run(params: &PanelLayoutParams) -> Outputs {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(
            &mut gpu,
            &mut pool,
            PanelLayoutOutputKeys {
                sdf: "sdf",
                panel_id: "panel_id",
                panel_index: "panel_index",
                uv_u: "uv_u",
                uv_v: "uv_v",
                seam: "seam",
            },
            RES,
            params,
        );
        let read16 = |key: &str| -> Vec<f32> {
            gpu.read_rgba16f(pool.texture(key).unwrap(), RES)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| p[0])
                .collect()
        };
        let sdf: Vec<f32> = gpu
            .read_rgba32f(pool.texture("sdf").unwrap(), RES)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| p[0])
            .collect();
        Outputs {
            sdf,
            panel_id: read16("panel_id"),
            panel_index: read16("panel_index"),
            uv_u: read16("uv_u"),
            uv_v: read16("uv_v"),
            seam: read16("seam"),
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
    fn matches_the_raster_goldens() {
        let out = run(&PanelLayoutParams::default());

        let want_sdf = read_npy_f32_2d(&golden_path("hardsurface", "panel_sdf.npy"));
        assert_matches_golden(&out.sdf, &want_sdf, "sdf");

        let (want_id, _) = read_png16_as_unit_floats(&golden_path("hardsurface", "panel_id.png"));
        assert_matches_golden(&out.panel_id, &want_id, "panel_id");

        let (want_u, _) = read_png16_as_unit_floats(&golden_path("hardsurface", "panel_uv_u.png"));
        assert_matches_golden(&out.uv_u, &want_u, "uv_u");

        let (want_v, _) = read_png16_as_unit_floats(&golden_path("hardsurface", "panel_uv_v.png"));
        assert_matches_golden(&out.uv_v, &want_v, "uv_v");
    }

    /// No separate golden for `seam` — it's trivially derived from `sdf`,
    /// so the invariant itself is the test.
    #[test]
    fn seam_matches_one_minus_coverage_of_sdf() {
        let out = run(&PanelLayoutParams::default());
        let res_f = RES as f32;
        for (i, (&s, &seam)) in out.sdf.iter().zip(out.seam.iter()).enumerate() {
            let cov = (0.5 - s * res_f).clamp(0.0, 1.0);
            let expect = 1.0 - cov;
            assert!(
                (seam - expect).abs() < 2e-3,
                "seam mismatch at {i}: got {seam} want {expect}"
            );
        }
    }

    /// No separate golden for `panel_index` either — cross-checked against
    /// `panel_id` through their own known relationship instead
    /// (`hash01(index, 11, seed)`), confirming the index output is really
    /// right rather than just "some index".
    #[test]
    fn panel_id_matches_hash01_of_panel_index() {
        let params = PanelLayoutParams::default();
        let out = run(&params);
        for (i, (&idx, &id)) in out.panel_index.iter().zip(out.panel_id.iter()).enumerate() {
            let expect = hash01(idx.round() as u32, 11, params.seed) as f32;
            assert!(
                (id - expect).abs() < 2e-3,
                "panel_id mismatch at {i} (index {idx}): got {id} want {expect}"
            );
        }
    }

    #[test]
    fn fastener_path_has_one_closed_rect_per_large_enough_panel() {
        let params = PanelLayoutParams::default();
        let rects = panel_rects(&params);
        let path = fastener_path(&rects, &params);
        assert!(path.wrap);

        let m = params.gap / 2.0 + params.fastener_margin;
        let eligible = rects
            .iter()
            .filter(|r| (r.x1 - r.x0) > 2.0 * m && (r.y1 - r.y0) > 2.0 * m)
            .count();
        assert_eq!(
            path.subpaths.len(),
            eligible,
            "one subpath per rect wide/tall enough to survive the inset"
        );

        for sp in &path.subpaths {
            assert!(sp.closed);
            assert_eq!(sp.points.len(), 4);
            for p in &sp.points {
                assert_eq!(p.h_in, [0.0, 0.0]);
                assert_eq!(p.h_out, [0.0, 0.0]);
                assert_eq!(p.kind, mc_core::PointKind::Corner);
            }
        }
    }

    #[test]
    fn from_graph_round_trips() {
        let mut params = std::collections::BTreeMap::new();
        params.insert("seed".to_string(), mc_core::ParamValue::Seed(7));
        params.insert("grid_div".to_string(), mc_core::ParamValue::Int(8));
        params.insert("max_depth".to_string(), mc_core::ParamValue::Int(4));
        params.insert("min_depth".to_string(), mc_core::ParamValue::Int(1));
        params.insert("offset".to_string(), mc_core::ParamValue::Bool(false));
        params.insert("min_size".to_string(), mc_core::ParamValue::Float(0.2));
        params.insert("split_chance".to_string(), mc_core::ParamValue::Float(0.5));
        params.insert(
            "long_axis_bias".to_string(),
            mc_core::ParamValue::Float(0.7),
        );
        params.insert("gap".to_string(), mc_core::ParamValue::Float(0.01));
        params.insert(
            "corner_radius".to_string(),
            mc_core::ParamValue::Float(0.002),
        );
        params.insert(
            "fastener_margin".to_string(),
            mc_core::ParamValue::Float(0.02),
        );
        let parsed = PanelLayoutParams::from_graph(&params);
        assert_eq!(parsed.seed, 7);
        assert_eq!(parsed.grid_div, 8);
        assert_eq!(parsed.max_depth, 4);
        assert_eq!(parsed.min_depth, 1);
        assert!(!parsed.offset);
        assert!((parsed.min_size - 0.2).abs() < 1e-6);

        let defaults = PanelLayoutParams::from_graph(&std::collections::BTreeMap::new());
        assert_eq!(defaults.seed, 3);
        assert_eq!(defaults.grid_div, 16);
        assert!(defaults.offset);
    }
}
