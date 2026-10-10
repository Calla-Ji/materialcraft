// gradient_linear.wgsl — Generators / Patterns / Gradient Linear
//
// Inputs:  none.
// Output:  `output` — a left-to-right ramp, 0 at the left edge to 1 at the
//          right. This is exactly periodic (f(uv + 1) = f(uv)) but has a
//          visible seam at the wrap — same as every other texture tool's
//          "Linear Gradient" generator. It's meant to be combined with other
//          nodes (e.g. as a Blend mask), not used bare as a final texture.
//          (No rotation param: an arbitrary angle is *not* periodic over the
//          tile — only multiples of 90° are — so it's left out for v0.1
//          rather than silently breaking the tiling guarantee.)
// Params:  none.

struct GradientLinearParams {
    resolution: u32,
    _pad0: u32,
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
    let value = (f32(id.x) + 0.5) / f32(params.resolution);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
