// shape.wgsl — Generators / Patterns / Shape
//
// Inputs:  none.
// Output:  `output` — a single shape centered in the tile, white inside and
//          black outside with a soft edge. Trivially tileable: the shape
//          sits fully inside the tile on a flat background, so neighboring
//          tiles meet background-to-background with no seam.
// Params:
//   kind     — 0 Disc, 1 Square, 2 Polygon, 3 Star
//   sides    — vertex/point count for Polygon and Star
//   size     — how large the shape is (0..1, fraction of the tile)
//   softness — how soft the edge is (0 = hard edge)
//   rotation — 0..1, fraction of a full turn

struct ShapeParams {
    resolution: u32,
    kind: u32,
    sides: u32,
    _pad0: u32,
    size: f32,
    softness: f32,
    rotation: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: ShapeParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

fn rotate(p: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

// Distance to a regular N-gon's edge along the radial direction — a cheap,
// commonly-used approximation (not an exact SDF), good enough for a mask.
fn polygon_dist(p: vec2<f32>, sides: f32) -> f32 {
    let angle = atan2(p.y, p.x);
    let sector = 6.283185307 / sides;
    let a = ((angle % sector) + sector) % sector - sector * 0.5;
    return length(p) * cos(a);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);
    var p = uv - vec2<f32>(0.5, 0.5);
    p = rotate(p, params.rotation * 6.283185307);

    var dist: f32;
    switch params.kind {
        case 0u: {
            // Disc
            dist = length(p);
        }
        case 1u: {
            // Square
            dist = max(abs(p.x), abs(p.y));
        }
        case 2u: {
            // Polygon
            dist = polygon_dist(p, f32(params.sides));
        }
        default: {
            // Star: radius target alternates between an outer tip and an
            // inner valley across each sector.
            let angle = atan2(p.y, p.x);
            let sector = 6.283185307 / f32(params.sides);
            let a = ((angle % sector) + sector) % sector - sector * 0.5;
            let t = abs(a) / (sector * 0.5);
            let local_radius = mix(1.0, 0.45, t);
            dist = length(p) / local_radius;
        }
    }

    let edge = params.size * 0.5;
    let softness = max(params.softness * 0.5, 0.0005);
    let value = 1.0 - smoothstep(edge - softness, edge + softness, dist);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
