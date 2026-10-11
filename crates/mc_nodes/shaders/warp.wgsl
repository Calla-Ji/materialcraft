// SPDX-License-Identifier: MIT OR Apache-2.0
// warp.wgsl — Filters / Spatial / Warp
//
// Inputs:  `input_tex`  — the texture to distort.
//          `intensity_tex` — a height/intensity map; its local gradient
//          drives the distortion (push pixels "downhill"/"uphill").
// Output:  `output` — `input_tex` resampled at each pixel's position offset
//          by the intensity map's gradient, scaled by `amount` and wrap-
//          addressed so it stays tileable.
// Params:  amount — warp strength, in UV-space units.

struct WarpParams {
    resolution: u32,
    amount: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: WarpParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var intensity_tex: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let size = i32(params.resolution);
    let x = i32(id.x);
    let y = i32(id.y);

    let left = textureLoad(intensity_tex, vec2<i32>(wrap_coord(x - 1, size), y), 0).r;
    let right = textureLoad(intensity_tex, vec2<i32>(wrap_coord(x + 1, size), y), 0).r;
    let down = textureLoad(intensity_tex, vec2<i32>(x, wrap_coord(y - 1, size)), 0).r;
    let up = textureLoad(intensity_tex, vec2<i32>(x, wrap_coord(y + 1, size)), 0).r;

    // `gradient` here is a per-texel finite difference (Δheight over 1
    // texel), which shrinks as 1/resolution for a fixed-frequency input —
    // multiplying by `resolution` once just cancels that back out to the
    // resolution-independent continuous derivative (Δheight/ΔUV), not a
    // pixel offset. `amount` is applied to that continuous derivative to get
    // a resolution-independent UV-space offset, which *then* needs
    // converting to texel space (another factor of `resolution`) for this
    // shader's texel-indexed sampling loop below.
    let gradient = vec2<f32>(right - left, up - down) * 0.5;
    let gradient_duv = gradient * f32(params.resolution);
    let offset_uv = gradient_duv * params.amount;
    let offset_px = offset_uv * f32(params.resolution);
    let sample_coord = vec2<i32>(
        wrap_coord(x + i32(round(offset_px.x)), size),
        wrap_coord(y + i32(round(offset_px.y)), size),
    );
    textureStore(output, vec2<i32>(id.xy), textureLoad(input_tex, sample_coord, 0));
}
