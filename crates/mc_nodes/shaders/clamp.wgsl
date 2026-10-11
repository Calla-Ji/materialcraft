// SPDX-License-Identifier: MIT OR Apache-2.0
// clamp.wgsl — Filters / Adjustments / Clamp
//
// Inputs:  `input_tex` — the texture to clamp.
// Output:  `output` — each RGB channel clamped to [low, high]; alpha passed
//          through unchanged.
// Params:  low, high — the allowed output range.

struct ClampParams {
    resolution: u32,
    low: f32,
    high: f32,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> params: ClampParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let value = textureLoad(input_tex, vec2<i32>(id.xy), 0);
    let clamped = clamp(value.rgb, vec3<f32>(params.low), vec3<f32>(params.high));
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(clamped, value.a));
}
