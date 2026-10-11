// gradient_linear.wgsl — Generators / Patterns / Gradient Linear
//
// Inputs:  none.
// Output:  `output` — a ramp, 0 to 1 across the tile along `rotation`'s
//          direction (0 = left-to-right). This is exactly periodic
//          (f(uv + 1) = f(uv)) but has a visible seam at the wrap — same as
//          every other texture tool's "Linear Gradient" generator. It's
//          meant to be combined with other nodes (e.g. as a Blend mask), not
//          used bare as a final texture.
// Params:  rotation — 0..1, fraction of a full turn. Only multiples of 0.25
//          keep the tile's edges lining up with its neighbor's (same caveat,
//          and same reason, as Transform 2D's rotation).

struct GradientLinearParams {
    resolution: u32,
    rotation: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: GradientLinearParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);
    var p = uv - vec2<f32>(0.5, 0.5);
    let angle = -params.rotation * 6.283185307;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    p = vec2<f32>(p.x * cos_a - p.y * sin_a, p.x * sin_a + p.y * cos_a);
    let value = fract(p.x + 0.5);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
