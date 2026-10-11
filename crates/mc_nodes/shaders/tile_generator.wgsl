// SPDX-License-Identifier: MIT OR Apache-2.0
// tile_generator.wgsl — Generators / Patterns / Pattern Generator
//
// Inputs:  none.
// Output:  `output` — a brick/grid pattern: a height-like value, bright on
//          each tile's raised face (softened at the edges by `bevel`), 0 in
//          the gap between tiles.
// Params:
//   rows, columns      — grid density
//   offset             — stagger alternate rows by this fraction of a
//                        column width (0 = plain grid, 0.5 = classic brick).
//                        NOTE: for the stagger to tile seamlessly top-to-
//                        bottom, `rows` must be even — an odd row count
//                        flips row parity across the vertical wrap, same
//                        constraint every brick generator has.
//   gap                — mortar gap width, as a fraction of a tile
//   bevel              — how much of the tile's edge fades from full
//                        brightness to the gap
//   size_jitter        — per-tile random shrink, 0..1
//   rotation_jitter     — per-tile random rotation, 0..1 (small angles —
//                        this is a per-tile *cosmetic* tilt, not a precise
//                        physically-fitted brick edge)
//   luminance_jitter   — per-tile random brightness variation, 0..1
//   seed               — changes the per-tile randomness

struct TileGeneratorParams {
    resolution: u32,
    rows: u32,
    columns: u32,
    seed: u32,
    offset: f32,
    gap: f32,
    bevel: f32,
    size_jitter: f32,
    rotation_jitter: f32,
    luminance_jitter: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: TileGeneratorParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);

    let row = floor(uv.y * f32(params.rows));
    let row_i = i32(row) % 2;
    var x = uv.x * f32(params.columns);
    if row_i != 0 {
        x = x + params.offset;
    }
    let col = floor(x);
    let local_x = fract(x) - 0.5;
    let local_y = fract(uv.y * f32(params.rows)) - 0.5;

    let columns_i = i32(params.columns);
    let wrapped_col = wrap_coord(i32(col), columns_i);
    let cell = vec2<i32>(wrapped_col, i32(row));

    let h = hash2(cell, params.seed);
    let h2 = hash2(cell, params.seed + 101u);
    let h3 = hash2(cell, params.seed + 211u);

    let size_scale = 1.0 - params.size_jitter * h;
    let angle = (h2 - 0.5) * params.rotation_jitter * 0.6;
    let luminance = 1.0 - params.luminance_jitter * h3;

    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let rotated = vec2<f32>(
        local_x * cos_a - local_y * sin_a,
        local_x * sin_a + local_y * cos_a,
    );

    let half_size = vec2<f32>(0.5 - params.gap) * size_scale;
    let d = abs(rotated) - half_size;
    let outside = length(max(d, vec2<f32>(0.0)));
    let inside = min(max(d.x, d.y), 0.0);
    let sdf = outside + inside;

    var value = 0.0;
    if sdf < 0.0 {
        let bevel_amount = clamp(-sdf / max(params.bevel, 0.0001), 0.0, 1.0);
        value = luminance * mix(0.5, 1.0, bevel_amount);
    }
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
