// SPDX-License-Identifier: MIT OR Apache-2.0
// edge_detect.wgsl — Filters / Spatial / Edge Detect
//
// Inputs:  `input_tex`.
// Output:  `output` — the Sobel gradient magnitude of the input's luminance,
//          broadcast to RGB (alpha = 1); bright where the input changes
//          sharply, black over flat areas. Wrap-addressed so it stays
//          tileable.
// Params:  intensity — multiplier on the gradient magnitude.

struct EdgeDetectParams {
    resolution: u32,
    intensity: f32,
    tiling_mode: u32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: EdgeDetectParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

fn luminance_at(x: i32, y: i32, size: i32) -> f32 {
    let c = textureLoad(
        input_tex,
        vec2<i32>(
            tile_coord(x, size, wraps_x(params.tiling_mode)),
            tile_coord(y, size, wraps_y(params.tiling_mode)),
        ),
        0,
    );
    return dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let size = i32(params.resolution);
    let x = i32(id.x);
    let y = i32(id.y);

    let h00 = luminance_at(x - 1, y - 1, size);
    let h10 = luminance_at(x, y - 1, size);
    let h20 = luminance_at(x + 1, y - 1, size);
    let h01 = luminance_at(x - 1, y, size);
    let h21 = luminance_at(x + 1, y, size);
    let h02 = luminance_at(x - 1, y + 1, size);
    let h12 = luminance_at(x, y + 1, size);
    let h22 = luminance_at(x + 1, y + 1, size);

    let dx = (h20 + 2.0 * h21 + h22) - (h00 + 2.0 * h01 + h02);
    let dy = (h02 + 2.0 * h12 + h22) - (h00 + 2.0 * h10 + h20);

    let magnitude = clamp(length(vec2<f32>(dx, dy)) * params.intensity, 0.0, 1.0);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(vec3<f32>(magnitude), 1.0));
}
