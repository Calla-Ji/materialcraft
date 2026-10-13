// SPDX-License-Identifier: MIT OR Apache-2.0
// tube_raster.wgsl — shared "tube along a curve" rasterizer.
//
// See mc_nodes::tube_raster's own module doc comment for why this exists as
// its own shader instead of living inside the Knit node, and for the full
// derivation of the binning/wrap-around scheme below. In short: a capsule
// (a round tube between two endpoints) is the union of spheres swept along
// it, and for a given pixel the exposed top surface of the highest capsule
// covering it is
//
//     height = z(u) + sqrt(r(u)^2 - d(u)^2)
//
// where u is the closest point along the capsule's XY-projected line to the
// pixel, d is the planar (XY-only) distance to that point, and z/r are
// linearly interpolated along the capsule at u (KNIT_NODE_SPEC.md §5).
//
// Coordinates: the tile is [0,1)^2, periodic in both axes. A capsule's own
// endpoint coordinates may sit slightly outside [0,1) (e.g. a course's
// closing segment, built by adding 1.0 to its first sample's x) — this is
// fine, because every pixel-vs-segment comparison goes through `min_image`,
// which finds the correct periodic copy regardless.
//
// Inputs: segments (every capsule in the whole tile, pre-binned on the CPU
//         into a uniform grid — see mc_nodes::tube_raster::bin_segments),
//         cell_offsets/cell_indices (that grid, as a CSR structure).
// Outputs: height — (z + zmax) / (2*zmax) of the winning capsule, else 0.
//          mask   — 1 wherever any capsule covers this pixel, else 0.
//          ply_id — (winning capsule's ply + 1) / total_plies, else 0.
//          cell   — the winning capsule's raw `tag` (R32Uint, not
//                   normalized — a caller decodes its own tag-packing
//                   scheme), or CELL_MISS where nothing was hit. Knit's own
//                   stitch-info pass reads this back to recover which wale
//                   and course a pixel's winning capsule belongs to.

struct Segment {
    x0: f32, y0: f32, z0: f32, r0: f32,
    x1: f32, y1: f32, z1: f32, r1: f32,
    tag: u32,
}

struct RasterParams {
    resolution: u32,
    grid_n: u32,
    total_plies: u32,
    zmax: f32,
}

@group(0) @binding(0) var<uniform> params: RasterParams;
@group(0) @binding(1) var<storage, read> segments: array<Segment>;
@group(0) @binding(2) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> cell_indices: array<u32>;
@group(0) @binding(4) var height_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var mask_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var ply_id_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var cell_out: texture_storage_2d<r32uint, write>;

const CELL_MISS: u32 = 0xFFFFFFFFu;

// Nearest periodic copy of a difference on a tile of period 1 (the
// "minimum image convention" — see this shader's own top comment).
fn min_image(d: f32) -> f32 {
    return d - round(d);
}

fn wrap_cell(i: i32, n: i32) -> u32 {
    return u32(((i % n) + n) % n);
}

// (height, hit) of one capsule against pixel (px, py), or (_, 0.0) on a miss.
fn eval_segment(seg: Segment, px: f32, py: f32) -> vec2<f32> {
    let dx0 = min_image(seg.x0 - px);
    let dy0 = min_image(seg.y0 - py);
    let ex = seg.x1 - seg.x0; // already a short, non-wrapping difference —
    let ey = seg.y1 - seg.y0; // both endpoints share the same periodic copy.
    let len2 = ex * ex + ey * ey;
    var u = 0.0;
    if len2 > 1e-12 {
        u = clamp(-(dx0 * ex + dy0 * ey) / len2, 0.0, 1.0);
    }
    let qx = dx0 + u * ex;
    let qy = dy0 + u * ey;
    let d2 = qx * qx + qy * qy;
    let r = mix(seg.r0, seg.r1, u);
    let r2 = r * r;
    if d2 >= r2 {
        return vec2<f32>(0.0, 0.0);
    }
    let z = mix(seg.z0, seg.z1, u) + sqrt(r2 - d2);
    return vec2<f32>(z, 1.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let px = (f32(id.x) + 0.5) / f32(params.resolution);
    let py = (f32(id.y) + 0.5) / f32(params.resolution);

    let n = i32(params.grid_n);
    let icx = i32(floor(px * f32(params.grid_n)));
    let icy = i32(floor(py * f32(params.grid_n)));

    var best_z = -1.0e9;
    var best_tag = 0u;
    var hit = false;

    for (var dy = -1; dy <= 1; dy = dy + 1) {
        let cy = wrap_cell(icy + dy, n);
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let cx = wrap_cell(icx + dx, n);
            let cell = cy * u32(n) + cx;
            let start = cell_offsets[cell];
            let end = cell_offsets[cell + 1u];
            for (var i = start; i < end; i = i + 1u) {
                let seg = segments[cell_indices[i]];
                let result = eval_segment(seg, px, py);
                if result.y > 0.5 && result.x > best_z {
                    best_z = result.x;
                    best_tag = seg.tag;
                    hit = true;
                }
            }
        }
    }

    var height = 0.0;
    var mask = 0.0;
    var ply_id = 0.0;
    if hit {
        height = clamp((best_z + params.zmax) / (2.0 * params.zmax), 0.0, 1.0);
        mask = 1.0;
        let ply = best_tag % 8u;
        ply_id = (f32(ply) + 1.0) / f32(params.total_plies);
    }

    textureStore(height_out, vec2<i32>(id.xy), vec4<f32>(height, height, height, 1.0));
    textureStore(mask_out, vec2<i32>(id.xy), vec4<f32>(mask, mask, mask, 1.0));
    textureStore(ply_id_out, vec2<i32>(id.xy), vec4<f32>(ply_id, ply_id, ply_id, 1.0));
    let cell_val = select(CELL_MISS, best_tag, hit);
    textureStore(cell_out, vec2<i32>(id.xy), vec4<u32>(cell_val, 0u, 0u, 0u));
}
