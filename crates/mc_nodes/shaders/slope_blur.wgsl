// slope_blur.wgsl — Filters / Spatial / Slope Blur
//
// Inputs:  `input_tex`   — the texture to blur.
//          `control_tex` — drives both the blur's direction (its local
//          gradient) and strength (its local slope/steepness) — steep areas
//          of the control map blur `input` more, flat areas don't blur it
//          at all. Key for organic/eroded looks.
// Output:  `output` — wrap-addressed so it stays tileable.
// Params:  radius_px — maximum blur radius in pixels, reached at the control
//          map's steepest point.

struct SlopeBlurParams {
    resolution: u32,
    radius_px: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: SlopeBlurParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var control_tex: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba8unorm, write>;

fn control_at(x: i32, y: i32, size: i32) -> f32 {
    return textureLoad(control_tex, vec2<i32>(wrap_coord(x, size), wrap_coord(y, size)), 0).r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let size = i32(params.resolution);
    let x = i32(id.x);
    let y = i32(id.y);

    let left = control_at(x - 1, y, size);
    let right = control_at(x + 1, y, size);
    let down = control_at(x, y - 1, size);
    let up = control_at(x, y + 1, size);
    let grad = vec2<f32>(right - left, up - down) * 0.5;
    let slope = length(grad);

    let sigma = params.radius_px * clamp(slope * 10.0, 0.0, 1.0);
    if sigma < 0.5 {
        textureStore(output, vec2<i32>(id.xy), textureLoad(input_tex, vec2<i32>(id.xy), 0));
        return;
    }

    let dir = grad / slope;
    let taps = clamp(i32(ceil(sigma * 2.0)), 1, 32);

    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var i = -taps; i <= taps; i = i + 1) {
        let weight = exp(-f32(i * i) / (2.0 * sigma * sigma));
        let offset = dir * f32(i);
        let coord = vec2<i32>(
            wrap_coord(x + i32(round(offset.x)), size),
            wrap_coord(y + i32(round(offset.y)), size),
        );
        sum = sum + textureLoad(input_tex, coord, 0) * weight;
        weight_sum = weight_sum + weight;
    }
    textureStore(output, vec2<i32>(id.xy), sum / weight_sum);
}
