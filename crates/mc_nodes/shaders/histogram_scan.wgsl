// SPDX-License-Identifier: MIT OR Apache-2.0
// histogram_scan.wgsl — Filters / Blending & Masking / Value Range
//
// Inputs:  `input_tex` — the texture to threshold/window.
// Output:  `output` — a ramp of width (1 - contrast) centered on `position`:
//          values below the ramp go to 0, above it go to 1, inside it blend.
//          At contrast = 1 this is a hard threshold at `position`.
// Params:
//   position — where the transition is centered, in 0..1
//   contrast — how sharp the transition is, in 0..1 (1 = hard edge)

struct HistogramScanParams {
    resolution: u32,
    position: f32,
    contrast: f32,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> params: HistogramScanParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

fn scan(value: f32) -> f32 {
    let band = max(1.0 - params.contrast, 0.001);
    return clamp((value - (params.position - band * 0.5)) / band, 0.0, 1.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let value = textureLoad(input_tex, vec2<i32>(id.xy), 0);
    let scanned = vec3<f32>(scan(value.r), scan(value.g), scan(value.b));
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(scanned, value.a));
}
