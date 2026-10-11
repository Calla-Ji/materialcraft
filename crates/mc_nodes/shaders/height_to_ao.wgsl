// height_to_ao.wgsl — Converters / Height to AO
//
// Inputs:  `input_tex` — a height map.
// Output:  `output` — a simple screen-space-style ambient occlusion map:
//          darker wherever a pixel sits lower than nearby higher neighbors
//          (a valley, or just inside a raised edge's shadow), unchanged
//          (white) on flat or raised areas, or anywhere no higher neighbor
//          is within reach. Wrap-addressed so it stays tileable.
// Params:
//   radius_px — how far out sampling reaches, in pixels
//   intensity — multiplier on the occlusion strength
//
// Samples a ring of directions at several distances out to radius_px
// (rather than one single fixed-distance ring) and weights closer samples
// more heavily. A single fixed-radius ring only darkens the one exact
// distance where it happens to cross a height boundary, which for a sharp
// edge (e.g. a disc's silhouette) reads as a thin, hard-edged *ring*
// tracing that boundary rather than a soft occlusion falloff — and makes
// the Radius/Intensity params only relocate or recolor that ring instead
// of visibly widening or softening a gradient. Multi-distance, distance-
// weighted sampling fixes both: every pixel within radius_px of a raised
// edge now picks up smoothly graduated occlusion (strongest at the edge,
// fading out toward the radius), so Radius controls how wide that fade is
// and Intensity controls how dark it gets, as an artist would expect.

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
const DIRECTIONS: u32 = 8u;
const STEPS: u32 = 4u;

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
    var weight_sum = 0.0;
    for (var dir = 0u; dir < DIRECTIONS; dir = dir + 1u) {
        let angle = f32(dir) / f32(DIRECTIONS) * 2.0 * PI;
        let dir_vec = vec2<f32>(cos(angle), sin(angle));
        for (var step = 1u; step <= STEPS; step = step + 1u) {
            // 0 (closest) .. 1 (full radius), so `weight` fades linearly
            // to zero right at the radius boundary instead of cutting off
            // sharply.
            let t = f32(step) / f32(STEPS);
            let dist = params.radius_px * t;
            let dx = i32(round(dir_vec.x * dist));
            let dy = i32(round(dir_vec.y * dist));
            let neighbor = height_at(x + dx, y + dy, size);
            let weight = 1.0 - t;
            occlusion = occlusion + max(neighbor - h, 0.0) * weight;
            weight_sum = weight_sum + weight;
        }
    }
    occlusion = occlusion / weight_sum;

    let ao = clamp(1.0 - occlusion * params.intensity, 0.0, 1.0);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(vec3<f32>(ao), 1.0));
}
