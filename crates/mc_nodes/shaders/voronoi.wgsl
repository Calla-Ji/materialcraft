// SPDX-License-Identifier: MIT OR Apache-2.0
// voronoi.wgsl — Generators / Noise / Voronoi
//
// Inputs:  none.
// Output:  `output` — tileable cellular noise (see `voronoi_f1f2` in
//          common.wgsl), broadcast to RGB.
// Params:
//   scale  — how many cells tile across the texture
//   jitter — how far each cell's point can move from the cell center (0..1)
//   seed   — changes the pattern without changing its character
//   mode   — 0 F1 (distance to nearest point), 1 F2 (second-nearest),
//            2 F2-F1 (cell interiors flat, bright near edges),
//            3 Edges (thin bright lines exactly at cell borders)

struct VoronoiParams {
    resolution: u32,
    scale: u32,
    mode: u32,
    seed: u32,
    jitter: f32,
    tiling_mode: u32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: VoronoiParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);
    let fs = voronoi_f1f2(
        uv,
        i32(params.scale),
        params.jitter,
        params.seed,
        wraps_x(params.tiling_mode),
        wraps_y(params.tiling_mode),
    );

    var value = fs.x;
    switch params.mode {
        case 0u: { value = fs.x; }
        case 1u: { value = fs.y; }
        case 2u: { value = fs.y - fs.x; }
        default: { value = 1.0 - clamp((fs.y - fs.x) / 0.05, 0.0, 1.0); }
    }
    value = clamp(value, 0.0, 1.0);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
