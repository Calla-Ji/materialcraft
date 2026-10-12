// SPDX-License-Identifier: MIT OR Apache-2.0
// thread_shape.wgsl — Generators / Patterns / Thread
//
// Inputs:  none.
// Output:  `output` — a single fiber's own pattern: the Y axis is its
//          cross-section (0 outside the fiber, up to 1 at its densest), the
//          X axis is position *along* the fiber's length. For a single ply
//          (`ply_count` 1) nothing varies along X at all, same as every
//          version of this node before twist existed. For 2+ plies with
//          `twist_cycles` > 0, each strand's position in cross-section
//          space orbits the fiber's center as X advances — the actual
//          visual signature of a twisted/plied yarn (confirmed against real
//          yarn reference photos: a rope's identity comes from the visible
//          spiral running along its length, *not* from its cross-section
//          silhouette alone — a multi-strand cross-section with no spiral
//          just reads as a vague blob, not "rope"). `twist_cycles` is an
//          integer so this stays exactly periodic along X (value at x=0
//          equals x=1) — the same "must be periodic" requirement every
//          generator in this app has, since Fabric Generator (the only
//          intended consumer of this node's output) samples it with
//          wraparound once per grid cell a thread passes through.
// Params:
//   ply_count        — how many strands are twisted together. 1 is a plain
//                       single strand/tube; 2-4+ match 2-ply/3-ply/4-ply
//                       yarn.
//   thickness        — how much of the tile's height the whole fiber
//                       bundle fills, 0..1
//   roundness        — 0 a flat ribbon, 1 a round cross-section (applies to
//                       each individual strand)
//   twist_cycles     — how many full twist rotations repeat across this
//                       texture's own width. Only meaningful for 2+ plies;
//                       0 means straight, untwisted parallel plies.
//   twist_direction  — 0 S-twist, 1 Z-twist (standard textile terms for the
//                       two mirror-image spiral directions — just a sign
//                       flip on the rotation here). Only meaningful when
//                       twist_cycles > 0.

struct ThreadShapeParams {
    resolution: u32,
    ply_count: u32,
    thickness: f32,
    roundness: f32,
    twist_cycles: u32,
    twist_direction: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: ThreadShapeParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

const TWIST_Z: u32 = 1u;
const PI: f32 = 3.14159265358979;

// One strand's own cross-section, centered at `center` rather than always 0.
fn strand_profile(t: f32, center: f32, half_width: f32, roundness: f32) -> f32 {
    if half_width <= 0.0001 {
        return 0.0;
    }
    let d = clamp(abs(t - center) / half_width, 0.0, 1.0);
    let flat = 1.0 - d;
    let round_profile = sqrt(max(1.0 - d * d, 0.0));
    return mix(flat, round_profile, roundness);
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
    let ply_count = max(params.ply_count, 1u);

    if ply_count == 1u {
        let value = strand_profile(local_y, 0.0, half_w, roundness);
        textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
        return;
    }

    // Each strand orbits the bundle's center as `uv.x` advances — an
    // integer number of full turns (`twist_cycles`) across this texture's
    // own width, so it stays exactly periodic. Direction just flips the
    // rotation's sign, mirroring the spiral.
    let strand_half_width = half_w * 0.55;
    let orbit_radius = half_w * 0.5;
    var sign = 1.0;
    if params.twist_direction == TWIST_Z {
        sign = -1.0;
    }
    let angle_base = sign * f32(params.twist_cycles) * uv.x * 2.0 * PI;

    var value = 0.0;
    for (var k = 0u; k < ply_count; k = k + 1u) {
        let phase = 2.0 * PI * f32(k) / f32(ply_count);
        let center = orbit_radius * sin(angle_base + phase);
        value = max(value, strand_profile(local_y, center, strand_half_width, roundness));
    }

    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
