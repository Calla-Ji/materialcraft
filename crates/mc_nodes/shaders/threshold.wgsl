// threshold.wgsl — Filters / Blending & Masking / Threshold
//
// Inputs:  `input_tex` — the texture to threshold.
// Output:  `output` — a hard black/white mask: each RGB channel becomes 1.0
//          if it's >= `threshold`, else 0.0; alpha passed through unchanged.
// Params:  threshold — the cutoff value.

struct ThresholdParams {
    resolution: u32,
    threshold: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: ThresholdParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let value = textureLoad(input_tex, vec2<i32>(id.xy), 0);
    let mask = step(vec3<f32>(params.threshold), value.rgb);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(mask, value.a));
}
