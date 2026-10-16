// SPDX-License-Identifier: MIT OR Apache-2.0
// path_to_sdf.wgsl — Generators / Path / Path to SDF
// (docs/hardsurface_nodes/HARDSURFACE_NODES.md §3.6)
//
// `edges` is every subpath flattened to line segments on the CPU
// (`mc_nodes::path_to_sdf::edges_for_path`, `kurbo::flatten` at tolerance
// 1e-4) — a dynamic-length storage buffer, the same per-pixel-loop-over-
// CPU-geometry pattern `tube_raster.wgsl`/`panel_layout.wgsl` already use.
//
// Output: `sdf` — `fill`: even-odd inside/outside, magnitude = nearest
//                  distance to *any* edge (open subpaths still pull the
//                  magnitude even though they can't flip the sign).
//                 `stroke`: nearest distance to any edge, minus width/2 —
//                  unsigned geometry, sign comes only from being on the
//                  near vs. far side of that offset tube.
// Params:
//   mode  — 0 fill, 1 stroke
//   wrap  — tests the 3x3 shifted copies of every edge when nonzero, so
//           a subpath crossing the tile seam still reads correctly.
//   width — stroke mode only.

struct PathToSdfParams {
    resolution: u32,
    edge_count: u32,
    mode: u32,
    wrap: u32,
    width: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct EdgeData {
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    closed: f32,
}

@group(0) @binding(0) var<uniform> params: PathToSdfParams;
@group(0) @binding(1) var<storage, read> edges: array<EdgeData>;
@group(0) @binding(2) var sdf_out: texture_storage_2d<rgba32float, write>;

fn seg_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let denom = max(dot(ab, ab), 1e-18);
    let t = clamp(dot(p - a, ab) / denom, 0.0, 1.0);
    let c = a + t * ab;
    return length(p - c);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let res_f = f32(params.resolution);
    let p = (vec2<f32>(id.xy) + 0.5) / res_f;

    var best: f32 = 1e20;
    var crossings: u32 = 0u;
    let shift_range: i32 = select(0, 1, params.wrap != 0u);

    for (var i: u32 = 0u; i < params.edge_count; i = i + 1u) {
        let e = edges[i];
        let is_closed = e.closed > 0.5;
        for (var sx: i32 = -shift_range; sx <= shift_range; sx = sx + 1) {
            for (var sy: i32 = -shift_range; sy <= shift_range; sy = sy + 1) {
                let shift = vec2<f32>(f32(sx), f32(sy));
                let a = vec2<f32>(e.ax, e.ay) + shift;
                let b = vec2<f32>(e.bx, e.by) + shift;
                best = min(best, seg_dist(p, a, b));

                if (params.mode == 0u && is_closed) {
                    let ya = a.y;
                    let yb = b.y;
                    let cond = (ya > p.y) != (yb > p.y);
                    if (cond) {
                        let xint = a.x + (p.y - ya) / (yb - ya + 1e-18) * (b.x - a.x);
                        if (p.x < xint) {
                            crossings = crossings + 1u;
                        }
                    }
                }
            }
        }
    }

    var sdf: f32;
    if (params.mode == 1u) {
        sdf = best - params.width * 0.5;
    } else {
        let inside = (crossings % 2u) == 1u;
        sdf = select(best, -best, inside);
    }

    textureStore(sdf_out, vec2<i32>(id.xy), vec4<f32>(sdf, sdf, sdf, 1.0));
}
