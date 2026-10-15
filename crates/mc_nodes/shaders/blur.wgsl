// SPDX-License-Identifier: MIT OR Apache-2.0
// blur.wgsl — Filters / Spatial / Blur
//
// Inputs:  `input_tex`.
// Output:  `output` — one pass of a separable Gaussian blur, wrap-addressed
//          so it stays tileable. `mc_nodes::blur::dispatch` runs this twice
//          (horizontal, then vertical) to get a full 2D blur.
// Params:
//   direction — 0 = horizontal pass, 1 = vertical pass
//   radius_px — blur radius in pixels (converted from the node's UV-space
//               radius param on the Rust side, since params here are in
//               texel space for the sampling loop)
//   wrap — whether *this pass's* axis should wrap (tile) or clamp to the
//          edge texel instead — resolved from the node's Tiling Mode param
//          on the Rust side, since each pass only ever touches one axis.

struct BlurParams {
    resolution: u32,
    direction: u32,
    radius_px: f32,
    wrap: u32,
}

@group(0) @binding(0) var<uniform> params: BlurParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let size = i32(params.resolution);
    let sigma = max(params.radius_px, 0.5);
    // Fixed at 32 taps max — beyond sigma = 16px, the step between samples
    // grows instead of the tap count, so the sampled window keeps reaching
    // out to 2*sigma for any radius instead of plateauing at a fixed ±32
    // texel window (which is what made the blur stop visibly getting
    // stronger past a small radius).
    let step = max(sigma / 16.0, 1.0);
    let taps = clamp(i32(ceil(sigma * 2.0 / step)), 1, 32);

    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var i = -taps; i <= taps; i = i + 1) {
        let dist = f32(i) * step;
        let weight = exp(-(dist * dist) / (2.0 * sigma * sigma));
        var coord = vec2<i32>(id.xy);
        let wrap = params.wrap != 0u;
        if params.direction == 0u {
            coord.x = tile_coord(coord.x + i32(round(dist)), size, wrap);
        } else {
            coord.y = tile_coord(coord.y + i32(round(dist)), size, wrap);
        }
        sum = sum + textureLoad(input_tex, coord, 0) * weight;
        weight_sum = weight_sum + weight;
    }
    textureStore(output, vec2<i32>(id.xy), sum / weight_sum);
}
