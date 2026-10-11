// SPDX-License-Identifier: MIT OR Apache-2.0
// gradient_radial.wgsl — Generators / Patterns / Gradient Radial
//
// Inputs:  none.
// Output:  `output` — 0 at tile center, ramping to 1 at distance `radius`
//          from center (clamped beyond that). Periodic by construction: it's
//          built from `fract(uv)`, so each tile independently repeats the
//          same centered ramp.
// Params:  radius (0..1, distance from center where the ramp reaches 1).

struct GradientRadialParams {
    resolution: u32,
    radius: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: GradientRadialParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);
    let centered = fract(uv) - vec2<f32>(0.5, 0.5);
    let value = clamp(length(centered) / max(params.radius, 0.0001), 0.0, 1.0);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
