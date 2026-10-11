// SPDX-License-Identifier: MIT OR Apache-2.0
// height_to_normal.wgsl — Converters / Height to Normal
//
// Inputs:  `input_tex` — a height/gray map.
// Output:  `output` — a tangent-space normal map encoded as RGB (0.5 = flat).
// Params:
//   intensity — how strongly height differences tilt the normal
//   flip_y    — 0 OpenGL (Y+, for Blender), 1 DirectX (Y-, for Unreal)

struct HeightToNormalParams {
    resolution: u32,
    intensity: f32,
    flip_y: u32,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> params: HeightToNormalParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

fn height_at(x: i32, y: i32, size: i32) -> f32 {
    return textureLoad(input_tex, vec2<i32>(wrap_coord(x, size), wrap_coord(y, size)), 0).r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let size = i32(params.resolution);
    let x = i32(id.x);
    let y = i32(id.y);

    // Sobel kernels for the height gradient.
    let h00 = height_at(x - 1, y - 1, size);
    let h10 = height_at(x, y - 1, size);
    let h20 = height_at(x + 1, y - 1, size);
    let h01 = height_at(x - 1, y, size);
    let h21 = height_at(x + 1, y, size);
    let h02 = height_at(x - 1, y + 1, size);
    let h12 = height_at(x, y + 1, size);
    let h22 = height_at(x + 1, y + 1, size);

    let dx = (h20 + 2.0 * h21 + h22) - (h00 + 2.0 * h01 + h02);
    let dy = (h02 + 2.0 * h12 + h22) - (h00 + 2.0 * h10 + h20);

    var ny = -dy * params.intensity;
    if params.flip_y == 1u {
        ny = -ny;
    }
    let normal = normalize(vec3<f32>(-dx * params.intensity, ny, 1.0));
    let color = normal * 0.5 + 0.5;
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}
