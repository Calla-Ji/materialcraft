// SPDX-License-Identifier: MIT OR Apache-2.0
// clouds.wgsl — Generators / Noise / Clouds
//
// Inputs:  none (this is a generator node).
// Output:  `output` — an fBm preset tuned for a cloud-like look (soft white
//          masses over darker gaps, rather than Perlin/Fractal Sum's more
//          even noise), broadcast to RGB (alpha = 1).
// Params (via the `params` uniform):
//   scale       — how many noise cells tile across the texture
//   seed        — changes the pattern without changing its character
//   persistence — how much each of the 5 fixed octaves shrinks (derived from
//                 the node's artist-facing "Softness" on the Rust side)

const OCTAVES: u32 = 5u;

struct CloudsParams {
    resolution: u32,
    scale: u32,
    seed: u32,
    persistence: f32,
    tiling_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: CloudsParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }

    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5, 0.5)) / f32(params.resolution);

    let wrap_x = wraps_x(params.tiling_mode);
    let wrap_y = wraps_y(params.tiling_mode);

    var amplitude = 1.0;
    var max_amplitude = 0.0;
    var sum = 0.0;
    var period = i32(params.scale);
    for (var i = 0u; i < OCTAVES; i = i + 1u) {
        sum = sum + perlin_2d(uv, period, params.seed + i * 1013u, wrap_x, wrap_y) * amplitude;
        max_amplitude = max_amplitude + amplitude;
        amplitude = amplitude * params.persistence;
        period = period * 2;
    }

    let raw = clamp(sum / max_amplitude * 0.5 + 0.5, 0.0, 1.0);
    // Push mids toward white and darken the troughs, rather than the even
    // spread plain fBm has, so this actually reads as clouds instead of
    // generic noise.
    let value = smoothstep(0.25, 0.85, raw);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
