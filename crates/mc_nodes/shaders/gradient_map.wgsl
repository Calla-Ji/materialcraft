// gradient_map.wgsl — Filters / Adjustments / Gradient Map
//
// Inputs:  `input_tex` — a Gray texture (value read from .r).
// Output:  `output` — the input value looked up through a piecewise-linear
//          color gradient.
// Params:
//   stop_count — how many of the up to 8 stops are actually in use
//   positions  — each stop's position, packed 4-per-vec4 (positions[i/4][i%4])
//   colors     — each stop's RGBA color
// Stops must already be sorted by position ascending (done on the Rust
// side) — the lookup below assumes it.

struct GradientMapParams {
    resolution: u32,
    stop_count: u32,
    _pad0: u32,
    _pad1: u32,
    positions: array<vec4<f32>, 2>,
    colors: array<vec4<f32>, 8>,
}

@group(0) @binding(0) var<uniform> params: GradientMapParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

fn stop_pos(i: u32) -> f32 {
    return params.positions[i / 4u][i % 4u];
}

fn gradient_color(g: f32) -> vec4<f32> {
    let n = params.stop_count;
    if g <= stop_pos(0u) {
        return params.colors[0];
    }
    for (var i = 0u; i < n - 1u; i = i + 1u) {
        let p1 = stop_pos(i + 1u);
        if g <= p1 {
            let p0 = stop_pos(i);
            let t = clamp((g - p0) / max(p1 - p0, 0.0001), 0.0, 1.0);
            return mix(params.colors[i], params.colors[i + 1u], t);
        }
    }
    return params.colors[n - 1u];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let value = textureLoad(input_tex, vec2<i32>(id.xy), 0).r;
    textureStore(output, vec2<i32>(id.xy), gradient_color(value));
}
