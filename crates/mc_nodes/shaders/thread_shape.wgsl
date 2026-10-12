// SPDX-License-Identifier: MIT OR Apache-2.0
// thread_shape.wgsl — Generators / Patterns / Thread
//
// Inputs:  none.
// Output:  `output` — a single fiber cross-section, as one horizontal bar
//          spanning the full tile width, centered vertically: 0 outside the
//          fiber, up to 1 at its centerline. Deliberately *not* tiled/
//          repeated — this is the one-shape building block meant to be
//          wired into Fabric Generator's Fiber input, the same relationship
//          Shape has to Pattern Generator's own Shape input.
// Params:
//   fiber_type — 0 Tube (the only type today; more can be added the same
//                way Shape's own kinds were)
//   thickness  — how much of the tile's height the fiber fills, 0..1
//   roundness  — 0 a flat ribbon, 1 a round cross-section

struct ThreadShapeParams {
    resolution: u32,
    fiber_type: u32,
    thickness: f32,
    roundness: f32,
}

@group(0) @binding(0) var<uniform> params: ThreadShapeParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

const FIBER_TUBE: u32 = 0u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);
    let local_y = uv.y - 0.5;

    var value = 0.0;
    if params.fiber_type == FIBER_TUBE {
        let half_w = clamp(params.thickness, 0.0, 1.0) * 0.5;
        if half_w > 0.0001 {
            let d = clamp(abs(local_y) / half_w, 0.0, 1.0);
            let flat = 1.0 - d;
            let round_profile = sqrt(max(1.0 - d * d, 0.0));
            value = mix(flat, round_profile, clamp(params.roundness, 0.0, 1.0));
        }
    }

    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
