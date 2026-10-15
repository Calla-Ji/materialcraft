// SPDX-License-Identifier: MIT OR Apache-2.0
// anisotropic_noise.wgsl — Generators / Noise / Anisotropic Noise
//
// Inputs:  none (this is a generator node).
// Output:  `output` — a tileable grayscale height field, broadcast to RGB
//          (alpha = 1) so it can be previewed directly and read back as a
//          plain image without a separate conversion pass.
// Params (via the `params` uniform):
//   scale_x     — how many noise cells tile across the texture along X
//   scale_y     — how many noise cells tile across the texture along Y
//   octaves     — how many layers of noise are summed (fractal Brownian motion)
//   persistence — how much each octave's contribution shrinks, in (0, 1]
//   seed        — changes the random pattern without changing its statistics
//   sharpness   — contrast boost around mid-gray; 1 = unchanged, higher reads
//                 as crisper and less blurry by pushing values away from 0.5
//
// Unlike Perlin Noise, `scale_x` and `scale_y` are independent: setting one
// much higher than the other stretches the noise into streaks along the
// lower-frequency axis (brushed metal, wood grain, combed fabric). Equal
// scales reduce to plain isotropic Perlin noise. For a streak direction
// other than horizontal/vertical, chain into a Transform 2D node — its own
// rotation shares the same "exact multiples of 90°" tiling caveat, so
// re-deriving it here wouldn't add anything.

struct AnisotropicNoiseParams {
    resolution: u32,
    scale_x: u32,
    scale_y: u32,
    octaves: u32,
    seed: u32,
    persistence: f32,
    tiling_mode: u32,
    sharpness: f32,
}

@group(0) @binding(0) var<uniform> params: AnisotropicNoiseParams;
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
    var period = vec2<i32>(i32(params.scale_x), i32(params.scale_y));
    for (var i = 0u; i < params.octaves; i = i + 1u) {
        sum = sum + perlin_2d_aniso(uv, period, params.seed + i * 1013u, wrap_x, wrap_y) * amplitude;
        max_amplitude = max_amplitude + amplitude;
        amplitude = amplitude * params.persistence;
        period = period * 2;
    }

    let raw = sum / max_amplitude * 0.5 + 0.5;
    let value = clamp((raw - 0.5) * params.sharpness + 0.5, 0.0, 1.0);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
