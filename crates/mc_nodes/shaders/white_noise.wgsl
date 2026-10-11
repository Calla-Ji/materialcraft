// white_noise.wgsl — Generators / Noise / White Noise
//
// Inputs:  none (this is a generator node).
// Output:  `output` — a grid of fully uncorrelated random cells (no
//          interpolation between them, unlike Perlin/Simplex), broadcast to
//          RGB (alpha = 1).
// Params:  scale — cells spanning the texture; seed — changes the pattern.

struct WhiteNoiseParams {
    resolution: u32,
    scale: u32,
    seed: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: WhiteNoiseParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5, 0.5)) / f32(params.resolution);
    let cell = vec2<i32>(floor(uv * f32(params.scale)));
    let value = hash2(cell, params.seed);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
