// SPDX-License-Identifier: MIT OR Apache-2.0
// perlin_noise.wgsl — Generators / Noise / Perlin Noise
//
// Inputs:  none (this is a generator node).
// Output:  `output` — a tileable grayscale height field, broadcast to RGB
//          (alpha = 1) so it can be previewed directly and read back as a
//          plain image without a separate conversion pass.
// Params (via the `params` uniform):
//   scale       — how many noise cells tile across the texture (bigger = finer detail)
//   octaves     — how many layers of noise are summed (fractal Brownian motion)
//   persistence — how much each octave's contribution shrinks, in (0, 1]
//   seed        — changes the random pattern without changing its statistics

struct PerlinParams {
    resolution: u32,
    scale: u32,
    octaves: u32,
    seed: u32,
    persistence: f32,
    tiling_mode: u32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: PerlinParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

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
    for (var i = 0u; i < params.octaves; i = i + 1u) {
        sum = sum + perlin_2d(uv, period, params.seed + i * 1013u, wrap_x, wrap_y) * amplitude;
        max_amplitude = max_amplitude + amplitude;
        amplitude = amplitude * params.persistence;
        period = period * 2;
    }

    let value = clamp(sum / max_amplitude * 0.5 + 0.5, 0.0, 1.0);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
