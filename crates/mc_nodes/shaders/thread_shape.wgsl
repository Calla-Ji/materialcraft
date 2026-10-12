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
//   fiber_type — 0 Tube, 1 Rope (see `rope_profile`'s own doc comment)
//   thickness  — how much of the tile's height the fiber fills, 0..1
//   roundness  — 0 a flat ribbon, 1 a round cross-section (Rope: applies to
//                each individual strand, not the rope as a whole)

struct ThreadShapeParams {
    resolution: u32,
    fiber_type: u32,
    thickness: f32,
    roundness: f32,
}

@group(0) @binding(0) var<uniform> params: ThreadShapeParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

const FIBER_TUBE: u32 = 0u;
const FIBER_ROPE: u32 = 1u;

// One strand's own cross-section, centered at `center` rather than always 0
// — `tube_profile` is this with `center` pinned to 0.
fn strand_profile(t: f32, center: f32, half_width: f32, roundness: f32) -> f32 {
    if half_width <= 0.0001 {
        return 0.0;
    }
    let d = clamp(abs(t - center) / half_width, 0.0, 1.0);
    let flat = 1.0 - d;
    let round_profile = sqrt(max(1.0 - d * d, 0.0));
    return mix(flat, round_profile, roundness);
}

// A classic 3-strand twisted rope's cross-section: three overlapping
// strands side by side, each its own (smaller) tube — the scalloped
// silhouette a bundle of 3 twisted strands has when viewed end-on. This
// doesn't render the twist itself (a visible spiral running *along* the
// thread's length), since `thread_shape`'s whole output is deliberately
// just one cross-section, constant along the thread's length, for Fabric
// Generator to sample cheaply at a single fixed point — see this file's own
// module doc comment. A real along-the-length twist would need a 2D fiber
// image and a different sampling model in Fabric Generator; this is the
// closest approximation within today's "one cross-section" architecture.
fn rope_profile(t: f32, half_width: f32, roundness: f32) -> f32 {
    if half_width <= 0.0001 {
        return 0.0;
    }
    let strand_half_width = half_width * 0.6;
    let strand_offset = half_width * 0.5;
    var value = strand_profile(t, -strand_offset, strand_half_width, roundness);
    value = max(value, strand_profile(t, 0.0, strand_half_width, roundness));
    value = max(value, strand_profile(t, strand_offset, strand_half_width, roundness));
    return value;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);
    let local_y = uv.y - 0.5;
    let half_w = clamp(params.thickness, 0.0, 1.0) * 0.5;
    let roundness = clamp(params.roundness, 0.0, 1.0);

    var value = 0.0;
    if params.fiber_type == FIBER_ROPE {
        value = rope_profile(local_y, half_w, roundness);
    } else {
        value = strand_profile(local_y, 0.0, half_w, roundness);
    }

    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
