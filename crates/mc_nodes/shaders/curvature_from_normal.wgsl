// curvature_from_normal.wgsl — Converters / Curvature from Normal
//
// Inputs:  `input_tex` — a tangent-space normal map (RGB, 0.5 = flat).
// Output:  `output` — the divergence of the normal's XY components: convex
//          edges (ridges) read bright, concave ones (crevices) read dark,
//          flat areas sit at mid-gray. Wrap-addressed so it stays tileable.
// Params:  intensity — multiplier on the curvature strength.

struct CurvatureFromNormalParams {
    resolution: u32,
    intensity: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: CurvatureFromNormalParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

fn normal_xy_at(x: i32, y: i32, size: i32) -> vec2<f32> {
    let c = textureLoad(input_tex, vec2<i32>(wrap_coord(x, size), wrap_coord(y, size)), 0);
    return c.rg * 2.0 - 1.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let size = i32(params.resolution);
    let x = i32(id.x);
    let y = i32(id.y);

    let left = normal_xy_at(x - 1, y, size);
    let right = normal_xy_at(x + 1, y, size);
    let down = normal_xy_at(x, y - 1, size);
    let up = normal_xy_at(x, y + 1, size);

    let divergence = (right.x - left.x) * 0.5 + (up.y - down.y) * 0.5;
    let value = clamp(divergence * params.intensity * 0.5 + 0.5, 0.0, 1.0);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(vec3<f32>(value), 1.0));
}
