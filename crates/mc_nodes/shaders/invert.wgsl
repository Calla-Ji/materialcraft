// invert.wgsl — Filters / Adjustments / Invert
//
// Inputs:  `input_tex` — the texture to invert.
// Output:  `output` — 1 - input, per RGB channel; alpha passed through.

struct InvertParams {
    resolution: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: InvertParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let value = textureLoad(input_tex, vec2<i32>(id.xy), 0);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(1.0 - value.rgb, value.a));
}
