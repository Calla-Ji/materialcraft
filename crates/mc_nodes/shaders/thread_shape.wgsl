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
// Fades out over its outer 15% with a smoothstep instead of stopping dead
// at `half_width` — a hard-edged circular strand swept along a curving
// twist path draws a scalloped envelope with a visible cusp everywhere two
// strands' circular boundaries meet (the geometric reason a first pass at
// this read as "ragged" from a screenshot, not just an aesthetic guess);
// tapering each strand's own edge softens that cusp into a flowing curve
// without blurring away the strand structure itself.
fn strand_profile(t: f32, center: f32, half_width: f32, roundness: f32) -> f32 {
    if half_width <= 0.0001 {
        return 0.0;
    }
    let d = clamp(abs(t - center) / half_width, 0.0, 1.0);
    let flat = 1.0 - d;
    let round_profile = sqrt(max(1.0 - d * d, 0.0));
    let shape = mix(flat, round_profile, roundness);
    let taper = 1.0 - smoothstep(0.85, 1.0, d);
    return shape * taper;
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

    // Pack `ply_count` strands evenly around a circle of `orbit_radius`,
    // sized so every strand's own circle overlaps its neighbors enough
    // that the bundle's cross-section never shows bare space down its
    // centerline — unlike the first version of this (each strand nearly as
    // wide as the whole orbit), which read as a moire lattice, instead of
    // sizing strands from a "just touching" baseline, which turned out to
    // leave an actual gap: for N strands evenly spaced around radius R,
    // the single strand closest to crossing dead center (local_y == 0) can
    // be up to PI/N of orbit away from it at any moment — "just touching"
    // sizes the strand to reach exactly that distance and no further,
    // which is an exact knife-edge (reaches zero coverage, not a visible
    // hole, only at the one instant every strand is simultaneously at its
    // worst-case offset) that a screenshot showed reading as a real void,
    // not a subtle groove. `overlap` pushes every strand comfortably past
    // that minimum instead. `orbit_radius + strand_half_width` still
    // reaches exactly `half_w`, so the bundle still fills the thickness
    // the slider promises; the groove look between plies that real yarn
    // has still comes through from each strand's own profile naturally
    // falling off toward its edge, not from an artificial shortfall.
    let n = f32(ply_count);
    let worst_case_offset_angle = PI / n;
    let overlap = 1.18;
    let orbit_radius = half_w / (1.0 + overlap * sin(worst_case_offset_angle));
    let strand_half_width = orbit_radius * overlap * sin(worst_case_offset_angle);

    // Each strand orbits the bundle's center as `uv.x` advances — an
    // integer number of full turns (`twist_cycles`) across this texture's
    // own width, so it stays exactly periodic. Direction just flips the
    // rotation's sign, mirroring the spiral.
    var sign = 1.0;
    if params.twist_direction == TWIST_Z {
        sign = -1.0;
    }
    let angle_base = sign * f32(params.twist_cycles) * uv.x * 2.0 * PI;

    // A real twisted yarn's strands genuinely pass in front of and behind
    // one another as they spiral, not just side by side — combining them
    // with a plain `max()` of brightness only captures the sideways
    // (sin) motion and produces an interference lattice, not a braid.
    // `cos` of each strand's own orbit angle is the matching depth
    // component of that same sin/cos helix projection: the strand with
    // greater depth (closer to the viewer) dominates, and every strand
    // dims slightly the further back it sits.
    //
    // Blended by relative depth (a softmax-style weighted average, not a
    // hard winner-take-all pick) rather than letting the front strand cut
    // off the back one outright: a hard switch put a visible crease right
    // at the exact crossing point every time occlusion flipped from one
    // strand to the other — the "overcut" groove a screenshot showed,
    // looking like a carved notch rather than a soft valley. Away from a
    // crossing, one strand's depth dominates so heavily (`exp` of the
    // depth difference) the blend is indistinguishable from a hard pick —
    // plies still read as genuinely crossing over and under each other,
    // just without a seam at the handoff.
    var weighted_sum = 0.0;
    var weight_total = 0.0;
    let occlusion_sharpness = 4.0;
    for (var k = 0u; k < ply_count; k = k + 1u) {
        let phase = 2.0 * PI * f32(k) / n;
        let angle = angle_base + phase;
        let center = orbit_radius * sin(angle);
        let depth = cos(angle);
        let profile = strand_profile(local_y, center, strand_half_width, roundness);
        if profile > 0.0005 {
            let shade = mix(0.55, 1.0, depth * 0.5 + 0.5);
            let weight = exp(depth * occlusion_sharpness);
            weighted_sum = weighted_sum + weight * profile * shade;
            weight_total = weight_total + weight;
        }
    }
    var value = 0.0;
    if weight_total > 0.0001 {
        value = weighted_sum / weight_total;
    }

    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
