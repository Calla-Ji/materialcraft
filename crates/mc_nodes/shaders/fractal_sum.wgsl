// SPDX-License-Identifier: MIT OR Apache-2.0
// fractal_sum.wgsl — Generators / Noise / Fractal Sum
//
// Inputs:  none.
// Output:  `output` — fractional Brownian motion (fBm): several octaves of
//          a chosen base noise, summed with each octave's frequency scaled
//          by `lacunarity` and amplitude scaled by `gain`.
// Params:
//   base        — 0 Perlin, 1 Voronoi (F1)
//   scale       — base noise cells tiling across the texture
//   octaves     — how many layers are summed
//   lacunarity  — frequency multiplier per octave (rounded to the nearest
//                 integer period each step, so every octave stays exactly
//                 tileable)
//   gain        — amplitude multiplier per octave
//   seed        — changes the pattern without changing its character

struct FractalSumParams {
    resolution: u32,
    base: u32,
    scale: u32,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    seed: u32,
    tiling_mode: u32,
}

@group(0) @binding(0) var<uniform> params: FractalSumParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);

    let wrap_x = wraps_x(params.tiling_mode);
    let wrap_y = wraps_y(params.tiling_mode);

    var amplitude = 1.0;
    var max_amplitude = 0.0;
    var sum = 0.0;
    var period = max(1, i32(params.scale));
    for (var i = 0u; i < params.octaves; i = i + 1u) {
        var layer: f32;
        if params.base == 0u {
            layer = perlin_2d(uv, period, params.seed + i * 1013u, wrap_x, wrap_y);
        } else {
            let fs = voronoi_f1f2(uv, period, 1.0, params.seed + i * 1013u, wrap_x, wrap_y);
            layer = fs.x * 2.0 - 1.0;
        }
        sum = sum + layer * amplitude;
        max_amplitude = max_amplitude + amplitude;
        amplitude = amplitude * params.gain;
        period = max(1, i32(round(f32(period) * params.lacunarity)));
    }

    let value = clamp(sum / max_amplitude * 0.5 + 0.5, 0.0, 1.0);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
