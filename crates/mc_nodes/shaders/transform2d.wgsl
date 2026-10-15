// SPDX-License-Identifier: MIT OR Apache-2.0
// transform2d.wgsl — Filters / Spatial / Transform 2D
//
// Inputs:  `input_tex`.
// Output:  `output` — `input_tex` offset, rotated, and scaled, wrap-
//          addressed so there's never an empty border.
//          Note: offset and integer-divisor scales stay exactly tileable;
//          an arbitrary rotation does *not* (only multiples of 90° do — the
//          same limitation every texture tool's Transform node has). Default
//          params (identity transform) are always tileable.
// Params:  offset_x, offset_y (-1..1), rotation (0..1, fraction of a turn),
//          scale_x, scale_y (how many times smaller/larger), flip_x, flip_y
//          (0/1 — mirror the input before offset/rotate/scale are applied),
//          tiling_mode (see `wraps_x`/`wraps_y` in common.wgsl — bit 0 wraps
//          X, bit 1 wraps Y; an axis that doesn't wrap clamps to its edge
//          instead, so content pushed past that edge clips rather than
//          reappearing on the opposite side).

struct Transform2DParams {
    resolution: u32,
    offset_x: f32,
    offset_y: f32,
    rotation: f32,
    scale_x: f32,
    scale_y: f32,
    flip_x: u32,
    flip_y: u32,
    tiling_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Transform2DParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);

    // Backward-map: find where in the input this output pixel's content
    // comes from, by applying the inverse offset/rotate/scale to `uv`.
    var p = uv - vec2<f32>(0.5, 0.5);
    if params.flip_x != 0u {
        p.x = -p.x;
    }
    if params.flip_y != 0u {
        p.y = -p.y;
    }
    p = p - vec2<f32>(params.offset_x, params.offset_y);
    let angle = -params.rotation * 6.283185307;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    p = vec2<f32>(p.x * cos_a - p.y * sin_a, p.x * sin_a + p.y * cos_a);
    p = p / vec2<f32>(max(params.scale_x, 0.0001), max(params.scale_y, 0.0001));
    p = p + vec2<f32>(0.5, 0.5);

    let size = i32(params.resolution);
    let sample_uv = vec2<f32>(
        tile_uv(p.x, wraps_x(params.tiling_mode)),
        tile_uv(p.y, wraps_y(params.tiling_mode)),
    );
    let coord = vec2<i32>(
        clamp(i32(sample_uv.x * f32(params.resolution)), 0, size - 1),
        clamp(i32(sample_uv.y * f32(params.resolution)), 0, size - 1),
    );
    textureStore(output, vec2<i32>(id.xy), textureLoad(input_tex, coord, 0));
}
