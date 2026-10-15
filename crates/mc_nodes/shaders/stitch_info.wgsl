// SPDX-License-Identifier: MIT OR Apache-2.0
// stitch_info.wgsl — per-pixel Height/Thread ID/Order/Direction for the
// Stitch node; a second pass run after tube_raster.wgsl has filled in
// height/mask/ply_id/cell for the same tile (mc_nodes::stitch::dispatch_render
// runs both in sequence on the same GPU queue, which guarantees this pass's
// reads see that one's writes — no explicit barrier needed, same convention
// knit_stitch_info.wgsl already established for Knit).
//
// Every covered pixel belongs to exactly one stitch, decoded from
// tube_raster's own `cell` output (packed the same way mc_nodes::knit's
// Segment tags are: order*8+ply, recovered here via cell/8). An uncovered
// pixel is plain background — unlike Knit there's no gap-fallback cell to
// fall into, every embroidery pixel either has a thread over it or it
// doesn't (STITCH_NODE_SPEC.md §8).
//
// Height needs its own re-normalization here rather than using
// tube_raster's own `height` output directly: that output is a SYMMETRIC
// (z+zmax)/(2*zmax) clamp (mapping z=0 to 0.5), but STITCH_NODE_SPEC.md §8
// wants height divided by `height_range * thread diameter` and clamped to
// [0,1] from zero — a different, one-sided shape (z<=0, e.g. a dipped
// hole, must map to 0, not ~0.5). So tube_raster's own height output is
// routed to a scratch key by the caller and inverted here
// (`raw_z = height01*2*zmax - zmax`) before Stitch's own formula is
// applied — `zmax` just needs to be generous enough to never actually
// clip a real z value, which mc_nodes::stitch::dispatch_render picks.
//
// stitch_reference.py's own hash01(), ported bit for bit (identical to
// knit_stitch_info.wgsl's copy — same algorithm, different node).

struct StitchInfoParams {
    resolution: u32,
    n_stitches: u32,
    seed: u32,
    zmax: f32,
    // height_range (tile units) * thread diameter (tile units) — Height's
    // own normalization divisor.
    height_norm: f32,
}

@group(0) @binding(0) var<uniform> params: StitchInfoParams;
@group(0) @binding(1) var<storage, read> thread_angles: array<f32>;
@group(0) @binding(2) var cell_in: texture_2d<u32>;
@group(0) @binding(3) var scratch_height_in: texture_2d<f32>;
@group(0) @binding(4) var height_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var thread_id_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var order_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var direction_out: texture_storage_2d<rgba16float, write>;

const CELL_MISS: u32 = 0xFFFFFFFFu;

// hash01() is now shared via common.wgsl (also used by Knit and Gabor
// Noise), which `full_source` always prepends.

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let tag = textureLoad(cell_in, coord, 0).r;

    if tag == CELL_MISS {
        textureStore(height_out, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(thread_id_out, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(order_out, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(direction_out, coord, vec4<f32>(0.5, 0.5, 0.5, 1.0));
        return;
    }

    let order = tag / 8u;

    let height01_sym = textureLoad(scratch_height_in, coord, 0).r;
    let raw_z = height01_sym * 2.0 * params.zmax - params.zmax;
    let height = clamp(raw_z / params.height_norm, 0.0, 1.0);

    let tid = hash01(order, 3u, params.seed);
    let ord = (f32(order) + 1.0) / f32(params.n_stitches);

    let theta = thread_angles[order];
    let r = 0.5 + 0.5 * cos(2.0 * theta);
    let g = 0.5 + 0.5 * sin(2.0 * theta);

    textureStore(height_out, coord, vec4<f32>(height, height, height, 1.0));
    textureStore(thread_id_out, coord, vec4<f32>(tid, tid, tid, 1.0));
    textureStore(order_out, coord, vec4<f32>(ord, ord, ord, 1.0));
    textureStore(direction_out, coord, vec4<f32>(r, g, 0.5, 1.0));
}
