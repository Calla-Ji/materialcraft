// SPDX-License-Identifier: MIT OR Apache-2.0
// directional_blur.wgsl — Filters / Spatial / Directional Blur
//
// Inputs:  `input_tex`.
// Output:  `output` — a Gaussian blur along a single fixed direction (motion-
//          blur-like), wrap-addressed so it stays tileable.
// Params:
//   angle     — direction of the blur, as a fraction of a full turn
//   radius_px — blur radius in pixels (converted from the node's UV-space
//               radius param on the Rust side)

struct DirectionalBlurParams {
    resolution: u32,
    angle: f32,
    radius_px: f32,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> params: DirectionalBlurParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

const PI: f32 = 3.14159265358979;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let size = i32(params.resolution);
    let theta = params.angle * 2.0 * PI;
    let dir = vec2<f32>(cos(theta), sin(theta));

    let sigma = max(params.radius_px, 0.5);
    // See blur.wgsl: step lets the sampled window keep reaching out to
    // 2*sigma for any radius instead of capping at a fixed ±32 texels.
    let step = max(sigma / 16.0, 1.0);
    let taps = clamp(i32(ceil(sigma * 2.0 / step)), 1, 32);

    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var i = -taps; i <= taps; i = i + 1) {
        let dist = f32(i) * step;
        let weight = exp(-(dist * dist) / (2.0 * sigma * sigma));
        let offset = dir * dist;
        let coord = vec2<i32>(
            wrap_coord(i32(id.x) + i32(round(offset.x)), size),
            wrap_coord(i32(id.y) + i32(round(offset.y)), size),
        );
        sum = sum + textureLoad(input_tex, coord, 0) * weight;
        weight_sum = weight_sum + weight;
    }
    textureStore(output, vec2<i32>(id.xy), sum / weight_sum);
}
