// directional_warp.wgsl — Filters / Spatial / Push Warp
//
// Inputs:  `input_tex`     — the texture to distort.
//          `intensity_tex` — drives how far each pixel shifts, by its own
//          value (not its gradient, unlike plain Warp).
// Output:  `output` — `input_tex` resampled at each pixel's position offset
//          along a single fixed direction by an amount proportional to the
//          intensity map's value there, wrap-addressed so it stays tileable.
// Params:  angle — direction of the warp, as a fraction of a full turn.
//          amount — warp strength, in UV-space units.

struct DirectionalWarpParams {
    resolution: u32,
    angle: f32,
    amount: f32,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> params: DirectionalWarpParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var intensity_tex: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba8unorm, write>;

const PI: f32 = 3.14159265358979;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let size = i32(params.resolution);
    let theta = params.angle * 2.0 * PI;
    let dir = vec2<f32>(cos(theta), sin(theta));

    let intensity = textureLoad(intensity_tex, vec2<i32>(id.xy), 0).r;
    let offset_px = dir * (intensity - 0.5) * params.amount * f32(params.resolution);
    let sample_coord = vec2<i32>(
        wrap_coord(i32(id.x) + i32(round(offset_px.x)), size),
        wrap_coord(i32(id.y) + i32(round(offset_px.y)), size),
    );
    textureStore(output, vec2<i32>(id.xy), textureLoad(input_tex, sample_coord, 0));
}
