// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Path / Path to SDF.
//!
//! Turns the Embroidery package's existing `Path` (`PortType::Path`,
//! already used by Path/Stitch/Path to Mask) into a real `Sdf`
//! (`docs/hardsurface_nodes/HARDSURFACE_NODES.md` §3.6) — Fill (closed
//! subpaths, even-odd rule) or Stroke (distance to any subpath, minus
//! half width). This is what lets hard-surface geometry be hand-drawn or
//! image-traced instead of only procedural: engraved lines, cutouts,
//! custom panel shapes.

use mc_core::{Path, SubPath};

use crate::shaders::full_source;

const NODE_KIND: &str = "path_to_sdf";

/// The spec's own stated tolerance for flattening curves into line
/// segments here — not a user param.
const FLATTEN_TOLERANCE: f64 = 1e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathSdfMode {
    Fill,
    Stroke,
}

impl PathSdfMode {
    pub const NAMES: [&'static str; 2] = ["Fill", "Stroke"];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Fill" => Some(Self::Fill),
            "Stroke" => Some(Self::Stroke),
            _ => None,
        }
    }

    fn as_u32(self) -> u32 {
        match self {
            Self::Fill => 0,
            Self::Stroke => 1,
        }
    }
}

pub struct PathToSdfParams {
    pub mode: PathSdfMode,
    /// Stroke mode only, tile units.
    pub width: f32,
}

impl Default for PathToSdfParams {
    fn default() -> Self {
        Self {
            mode: PathSdfMode::Fill,
            width: 0.012,
        }
    }
}

impl PathToSdfParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            mode: params
                .get("mode")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => PathSdfMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.mode),
            width: params
                .get("width")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.width),
        }
    }
}

/// Flattens every segment of `sp` to a polyline via `kurbo::flatten` at
/// `tolerance` — segment by segment (not the whole subpath as one
/// `BezPath`) so the "drop this segment's last point, except the
/// subpath's very last point if open" bookkeeping stays the same shape
/// as `path_reference.py`'s own `path_polylines` (which does the same
/// thing with a fixed sample count instead of `kurbo`'s adaptive one —
/// the spec itself calls for this substitution).
pub(crate) fn flatten_subpath(sp: &SubPath, tolerance: f64) -> Vec<[f64; 2]> {
    let segs = crate::path::segments(sp);
    let n = segs.len();
    let mut out = Vec::new();
    for (i, seg) in segs.iter().enumerate() {
        let mut seg_pts: Vec<[f64; 2]> = Vec::new();
        let els = [
            kurbo::PathEl::MoveTo(seg.p0),
            kurbo::PathEl::CurveTo(seg.p1, seg.p2, seg.p3),
        ];
        kurbo::flatten(els, tolerance, |el| match el {
            kurbo::PathEl::MoveTo(p) | kurbo::PathEl::LineTo(p) => seg_pts.push([p.x, p.y]),
            _ => {}
        });
        let is_last_segment_of_open_path = i == n - 1 && !sp.closed;
        if !is_last_segment_of_open_path {
            seg_pts.pop();
        }
        out.extend(seg_pts);
    }
    out
}

/// Every subpath's flattened polyline turned into `(a, b, closed)` edges —
/// closed subpaths wrap their last point back to the first; open ones
/// don't.
pub(crate) fn edges_for_path(path: &Path, tolerance: f64) -> Vec<([f64; 2], [f64; 2], bool)> {
    let mut edges = Vec::new();
    for sp in &path.subpaths {
        let poly = flatten_subpath(sp, tolerance);
        if poly.len() < 2 {
            continue;
        }
        if sp.closed {
            for i in 0..poly.len() {
                edges.push((poly[i], poly[(i + 1) % poly.len()], true));
            }
        } else {
            for w in poly.windows(2) {
                edges.push((w[0], w[1], false));
            }
        }
    }
    edges
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    path: &Path,
    params: &PathToSdfParams,
) {
    let edges = edges_for_path(path, FLATTEN_TOLERANCE);

    let shader_source = full_source("path_to_sdf.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, edges.len() as u32, path.wrap, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("path_to_sdf_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let edge_bytes = pack_edges(&edges);
    let edge_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("path_to_sdf_edges"),
        // An empty path is legal (unconnected input) but wgpu rejects a
        // zero-sized buffer outright.
        size: (edge_bytes.len() as u64).max(20),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !edge_bytes.is_empty() {
        gpu.queue.write_buffer(&edge_buffer, 0, &edge_bytes);
    }

    // `Sdf` sockets are `Rgba32Float` (HARDSURFACE_NODES.md §2).
    let sdf_view = pool
        .get_or_create(
            &gpu.device,
            key,
            resolution,
            wgpu::TextureFormat::Rgba32Float,
        )
        .clone();

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("path_to_sdf_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: edge_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&sdf_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("path_to_sdf_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("path_to_sdf_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

fn pack_edges(edges: &[([f64; 2], [f64; 2], bool)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(edges.len() * 20);
    for &(a, b, closed) in edges {
        bytes.extend_from_slice(&(a[0] as f32).to_le_bytes());
        bytes.extend_from_slice(&(a[1] as f32).to_le_bytes());
        bytes.extend_from_slice(&(b[0] as f32).to_le_bytes());
        bytes.extend_from_slice(&(b[1] as f32).to_le_bytes());
        bytes.extend_from_slice(&(if closed { 1.0f32 } else { 0.0f32 }).to_le_bytes());
    }
    bytes
}

fn pack_params(resolution: u32, edge_count: u32, wrap: bool, params: &PathToSdfParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&edge_count.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.mode.as_u32().to_le_bytes());
    bytes[12..16].copy_from_slice(&(wrap as u32).to_le_bytes());
    bytes[16..20].copy_from_slice(&params.width.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, PathSdfMode, PathToSdfParams};
    use crate::test_support::{golden_path, read_npy_f32_2d};
    use mc_core::{Path, Point, PointKind, SubPath};

    const RES: u32 = 256;

    /// `path_reference.py`'s `demo_path()` — the ornament every golden
    /// image uses. Ported once, verbatim: every field is a plain number,
    /// no randomness involved.
    fn demo_path() -> Path {
        let p =
            |pos: [f64; 2], h_in: [f64; 2], h_out: [f64; 2], kind: PointKind, width: f32| Point {
                pos,
                h_in,
                h_out,
                kind,
                width,
            };
        let wave = SubPath::new(
            vec![
                p(
                    [0.00, 0.24],
                    [-0.08, 0.0],
                    [0.08, 0.0],
                    PointKind::Symmetric,
                    0.7,
                ),
                p(
                    [0.25, 0.14],
                    [-0.07, 0.0],
                    [0.07, 0.0],
                    PointKind::Smooth,
                    1.0,
                ),
                p(
                    [0.50, 0.04],
                    [-0.04, 0.12],
                    [0.04, 0.12],
                    PointKind::Corner,
                    1.3,
                ),
                p(
                    [0.75, 0.34],
                    [-0.07, 0.0],
                    [0.07, 0.0],
                    PointKind::Smooth,
                    1.0,
                ),
                p(
                    [1.00, 0.24],
                    [-0.08, 0.0],
                    [0.08, 0.0],
                    PointKind::Symmetric,
                    0.7,
                ),
            ],
            false,
        );
        let leaf = SubPath::new(
            vec![
                p(
                    [0.12, 0.66],
                    [0.12, 0.13],
                    [0.12, -0.17],
                    PointKind::Corner,
                    0.7,
                ),
                p(
                    [0.62, 0.66],
                    [-0.12, -0.17],
                    [-0.12, 0.13],
                    PointKind::Corner,
                    0.7,
                ),
            ],
            true,
        );
        let (cx, cy, rad): (f64, f64, f64) = (0.82, 0.70, 0.11);
        let k = 0.5523 * rad;
        let mut circle_pts = Vec::new();
        for ang in 0..4 {
            let th = ang as f64 * std::f64::consts::FRAC_PI_2;
            let pos = [cx + rad * th.cos(), cy + rad * th.sin()];
            let tng = [-th.sin() * k, th.cos() * k];
            circle_pts.push(p(pos, [-tng[0], -tng[1]], tng, PointKind::Symmetric, 1.0));
        }
        let circle = SubPath::new(circle_pts, true);
        Path {
            subpaths: vec![wave, leaf, circle],
            wrap: true,
        }
    }

    fn run(path: &Path, params: &PathToSdfParams) -> Vec<f32> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "sdf", RES, path, params);
        gpu.read_rgba32f(pool.texture("sdf").unwrap(), RES)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| p[0])
            .collect()
    }

    /// Distance Field's own px-based tolerance (mean < 0.05, p99 < 0.3) —
    /// `path_fill_sdf.npy`/`path_stroke_sdf.npy` are raw `.npy` SDF
    /// arrays with no dedicated tolerance row of their own in §6, same
    /// category of artifact as `df_circle_sdf.npy`/`redistance_union.npy`.
    fn assert_matches_golden(got: &[f32], want: &[f32], label: &str) {
        let mut errs: Vec<f64> = got
            .iter()
            .zip(want.iter())
            .map(|(&g, &w)| ((g - w) as f64 * f64::from(RES)).abs())
            .collect();
        let mean = errs.iter().sum::<f64>() / errs.len() as f64;
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99 = errs[(errs.len() as f64 * 0.99) as usize];
        assert!(mean < 0.05, "{label}: mean error {mean} px");
        assert!(p99 < 0.3, "{label}: p99 error {p99} px");
    }

    #[test]
    fn fill_mode_matches_the_golden() {
        let out = run(
            &demo_path(),
            &PathToSdfParams {
                mode: PathSdfMode::Fill,
                ..PathToSdfParams::default()
            },
        );
        let want = read_npy_f32_2d(&golden_path("hardsurface", "path_fill_sdf.npy"));
        assert_matches_golden(&out, &want, "fill");
    }

    #[test]
    fn stroke_mode_matches_the_golden() {
        let out = run(
            &demo_path(),
            &PathToSdfParams {
                mode: PathSdfMode::Stroke,
                width: 0.012,
            },
        );
        let want = read_npy_f32_2d(&golden_path("hardsurface", "path_stroke_sdf.npy"));
        assert_matches_golden(&out, &want, "stroke");
    }

    /// A point deep inside the closed leaf/circle should read negative; a
    /// point far from every subpath should read positive — confirms the
    /// even-odd sign isn't accidentally inverted, independent of the
    /// golden's own overall accuracy number.
    #[test]
    fn fill_mode_sign_is_negative_inside_positive_outside() {
        let out = run(&demo_path(), &PathToSdfParams::default());
        let idx = |x: f64, y: f64| -> usize {
            let px = (x * f64::from(RES)) as usize;
            let py = (y * f64::from(RES)) as usize;
            py * RES as usize + px
        };
        // Leaf center (~0.37, 0.66) and circle center (0.82, 0.70).
        assert!(
            out[idx(0.37, 0.66)] < 0.0,
            "leaf center should be inside (negative)"
        );
        assert!(
            out[idx(0.82, 0.70)] < 0.0,
            "circle center should be inside (negative)"
        );
        // A corner, far from every subpath.
        assert!(
            out[idx(0.98, 0.02)] > 0.0,
            "a far corner should be outside (positive)"
        );
    }

    /// `wrap` must actually change the result for a subpath straddling
    /// the tile seam — a small synthetic path, not `demo_path()` (which
    /// doesn't necessarily cross the seam itself).
    #[test]
    fn wrap_changes_the_result_near_a_seam_crossing_subpath() {
        let path = Path {
            subpaths: vec![SubPath::new(
                vec![Point::new([-0.02, 0.5]), Point::new([0.3, 0.5])],
                false,
            )],
            wrap: true,
        };
        let no_wrap = Path {
            wrap: false,
            ..path.clone()
        };
        let params = PathToSdfParams {
            mode: PathSdfMode::Stroke,
            width: 0.02,
        };
        let wrapped = run(&path, &params);
        let unwrapped = run(&no_wrap, &params);
        assert_ne!(
            wrapped, unwrapped,
            "wrap on vs off should disagree near a seam-straddling subpath"
        );
    }

    #[test]
    fn from_graph_round_trips() {
        let mut params = std::collections::BTreeMap::new();
        params.insert(
            "mode".to_string(),
            mc_core::ParamValue::Enum("Stroke".to_string()),
        );
        params.insert("width".to_string(), mc_core::ParamValue::Float(0.05));
        let parsed = PathToSdfParams::from_graph(&params);
        assert_eq!(parsed.mode, PathSdfMode::Stroke);
        assert_eq!(parsed.width, 0.05);

        let defaults = PathToSdfParams::from_graph(&std::collections::BTreeMap::new());
        assert_eq!(defaults.mode, PathSdfMode::Fill);
    }
}
