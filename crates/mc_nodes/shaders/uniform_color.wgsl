// SPDX-License-Identifier: MIT OR Apache-2.0
// uniform_color.wgsl — Generators / Patterns / Uniform Color
//
// Inputs:  none.
// Output:  `output` — every pixel set to `color`.
// Params:  color (RGBA, 0..1 per channel).

struct UniformColorParams {
    resolution: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: UniformColorParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    textureStore(output, vec2<i32>(id.xy), params.color);
}
