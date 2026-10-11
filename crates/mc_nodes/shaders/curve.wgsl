// curve.wgsl — Filters / Adjustments / Curve
//
// Inputs:  `input_tex` — the texture to remap.
// Output:  `output` — each RGB channel passed through a piecewise-linear
//          curve defined by 5 fixed control points at x = 0, 0.25, 0.5,
//          0.75, 1.0 (CLAUDE.md's "simple 4-8 point curve", at the simple
//          end — piecewise-linear rather than a spline, so there's no
//          overshoot to reason about and the control points are exactly
//          where their sliders say). Alpha passed through unchanged.
// Params:  v0..v4 — output value at each of the 5 fixed input positions.

struct CurveParams {
    resolution: u32,
    v0: f32,
    v1: f32,
    v2: f32,
    v3: f32,
    v4: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: CurveParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

fn apply_curve(x: f32) -> f32 {
    let t = clamp(x, 0.0, 1.0) * 4.0;
    let seg = clamp(i32(floor(t)), 0, 3);
    let local = t - f32(seg);
    if seg == 0 {
        return mix(params.v0, params.v1, local);
    } else if seg == 1 {
        return mix(params.v1, params.v2, local);
    } else if seg == 2 {
        return mix(params.v2, params.v3, local);
    }
    return mix(params.v3, params.v4, local);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let value = textureLoad(input_tex, vec2<i32>(id.xy), 0);
    let adjusted = vec3<f32>(
        apply_curve(value.r),
        apply_curve(value.g),
        apply_curve(value.b),
    );
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(adjusted, value.a));
}
