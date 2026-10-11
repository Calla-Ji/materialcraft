// height_to_ao.wgsl — Converters / Height to AO
//
// Inputs:  `input_tex` — a height map.
// Output:  `output` — a simple screen-space-style ambient occlusion map:
//          darker wherever a pixel sits lower than its surrounding ring of
//          neighbors (a valley), unchanged (white) on flat or raised areas.
//          Wrap-addressed so it stays tileable.
// Params:
//   radius_px — how far out the sample ring reaches, in pixels
//   intensity — multiplier on the occlusion strength

struct HeightToAoParams {
    resolution: u32,
    radius_px: f32,
    intensity: f32,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> params: HeightToAoParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

const PI: f32 = 3.14159265358979;
const SAMPLE_COUNT: u32 = 8u;

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
    let h = height_at(x, y, size);

    var occlusion = 0.0;
    for (var i = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let angle = f32(i) / f32(SAMPLE_COUNT) * 2.0 * PI;
        let dx = i32(round(cos(angle) * params.radius_px));
        let dy = i32(round(sin(angle) * params.radius_px));
        let neighbor = height_at(x + dx, y + dy, size);
        occlusion = occlusion + max(neighbor - h, 0.0);
    }
    occlusion = occlusion / f32(SAMPLE_COUNT);

    let ao = clamp(1.0 - occlusion * params.intensity, 0.0, 1.0);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(vec3<f32>(ao), 1.0));
}
