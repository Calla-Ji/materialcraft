// SPDX-License-Identifier: MIT OR Apache-2.0
// knot_info.wgsl — per-pixel Height/Knot ID/Along/Direction for the
// Stitch node's French Knot — a second pass run after tube_raster.wgsl
// has filled in height/mask/ply_id/cell for the same tile
// (mc_nodes::french_knot::dispatch_render runs both in sequence on the
// same GPU queue, which guarantees this pass's reads see that one's
// writes — no explicit barrier needed, same convention
// knit_stitch_info.wgsl/stitch_info.wgsl already established).
//
// Strand ID isn't decoded here at all: it's exactly tube_raster's own
// `ply_id` output (dispatch_render routes it straight to that key), since
// "which of the n parallel strands is on top" is the same question Stitch
// already asks for its own plies.
//
// Every covered pixel belongs to exactly one segment, decoded from
// tube_raster's own `cell` output (packed as `sample_index * 8 + strand`
// — recovered here via `cell / 8u`). An uncovered pixel is plain
// background.
//
// Height needs its own re-normalization here rather than using
// tube_raster's own `height` output directly, for the same reason
// stitch_info.wgsl's own comment already gives: that output is a
// SYMMETRIC (z+zmax)/(2*zmax) clamp (mapping z=0 to 0.5), but
// FRENCH_KNOT_SPEC.md §3 wants height divided by `height_range * D` and
// clamped to [0,1] from zero — a one-sided shape. So tube_raster's own
// height output is routed to a scratch key by the caller and inverted
// here (`raw_z = height01*2*zmax - zmax`) before this node's own formula
// is applied.
//
// Knot ID is a plain uniform, not a per-segment lookup: single-knot
// placement only, so every covered pixel belongs to the same one knot.

struct KnotInfoParams {
    resolution: u32,
    n_segments: u32,
    zmax: f32,
    // height_range (bundle diameters) * D (bundle diameter, tile units) —
    // Height's own normalization divisor.
    height_norm: f32,
    knot_id: f32,
}

@group(0) @binding(0) var<uniform> params: KnotInfoParams;
@group(0) @binding(1) var<storage, read> along: array<f32>;
@group(0) @binding(2) var<storage, read> angle: array<f32>;
@group(0) @binding(3) var cell_in: texture_2d<u32>;
@group(0) @binding(4) var scratch_height_in: texture_2d<f32>;
@group(0) @binding(5) var height_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var knot_id_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var along_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var direction_out: texture_storage_2d<rgba16float, write>;

const CELL_MISS: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let tag = textureLoad(cell_in, coord, 0).r;

    if tag == CELL_MISS {
        textureStore(height_out, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(knot_id_out, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(along_out, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(direction_out, coord, vec4<f32>(0.5, 0.5, 0.5, 1.0));
        return;
    }

    let sample_index = tag / 8u;

    let height01_sym = textureLoad(scratch_height_in, coord, 0).r;
    let raw_z = height01_sym * 2.0 * params.zmax - params.zmax;
    let height = clamp(raw_z / params.height_norm, 0.0, 1.0);

    let t = along[sample_index];
    let theta = angle[sample_index];
    let r = 0.5 + 0.5 * cos(2.0 * theta);
    let g = 0.5 + 0.5 * sin(2.0 * theta);

    textureStore(height_out, coord, vec4<f32>(height, height, height, 1.0));
    textureStore(knot_id_out, coord, vec4<f32>(params.knot_id, params.knot_id, params.knot_id, 1.0));
    textureStore(along_out, coord, vec4<f32>(t, t, t, 1.0));
    textureStore(direction_out, coord, vec4<f32>(r, g, 0.5, 1.0));
}
