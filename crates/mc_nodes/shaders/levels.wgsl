// SPDX-License-Identifier: MIT OR Apache-2.0
// levels.wgsl — Filters / Adjustments / Levels
//
// Inputs:  `input_tex` — the texture to adjust.
// Output:  `output` — same resolution, with an input/output range remap and
//          a gamma curve applied per RGB channel (alpha passed through
//          unchanged).
// Params (via the `params` uniform):
//   in_low, in_high   — input range that gets stretched to fill 0..1
//   gamma             — midtone curve (1.0 = linear)
//   out_low, out_high — output range the stretched value is mapped into

struct LevelsParams {
    resolution: u32,
    in_low: f32,
    in_high: f32,
    gamma: f32,
    out_low: f32,
    out_high: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: LevelsParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

fn apply_levels(value: f32) -> f32 {
    let range = max(params.in_high - params.in_low, 0.0001);
    var v = clamp((value - params.in_low) / range, 0.0, 1.0);
    // gamma = 1.0 (the default, "no curve") skips `pow` entirely rather
    // than calling `pow(v, 1.0)` — mathematically a no-op, but GPU `pow` is
    // typically `exp2(log2(x) * y)` under the hood, which isn't guaranteed
    // bit-exact for y = 1.0 on every backend. Confirmed as a real
    // cross-platform CI failure: identical on Metal/Vulkan, off by enough
    // to flip an 8-bit-readback pixel on Windows' DX12 backend.
    if params.gamma != 1.0 {
        v = pow(v, 1.0 / max(params.gamma, 0.0001));
    }
    return mix(params.out_low, params.out_high, v);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }

    let input_value = textureLoad(input_tex, vec2<i32>(id.xy), 0);
    let adjusted = vec3<f32>(
        apply_levels(input_value.r),
        apply_levels(input_value.g),
        apply_levels(input_value.b),
    );
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(adjusted, input_value.a));
}
