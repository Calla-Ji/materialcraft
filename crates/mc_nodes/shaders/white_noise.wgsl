// SPDX-License-Identifier: MIT OR Apache-2.0
// white_noise.wgsl — Generators / Noise / White Noise
//
// Inputs:  none (this is a generator node).
// Output:  `output` — a grid of fully uncorrelated random cells (no
//          interpolation between them, unlike Perlin/Simplex), broadcast to
//          RGB (alpha = 1).
// Params:  scale_x/scale_y — cells spanning the texture along each axis
//          independently (equal values give square cells, same as the old
//          single "scale"); seed — changes the pattern.

struct WhiteNoiseParams {
    resolution: u32,
    scale_x: u32,
    scale_y: u32,
    seed: u32,
}

@group(0) @binding(0) var<uniform> params: WhiteNoiseParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5, 0.5)) / f32(params.resolution);
    let cell = vec2<i32>(floor(uv * vec2<f32>(f32(params.scale_x), f32(params.scale_y))));
    let value = hash2(cell, params.seed);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
