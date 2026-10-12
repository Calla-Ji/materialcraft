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

// Converts a signed "how far inside the strand" distance (positive inside,
// 0 at the edge, negative outside) into the same flat-ribbon/round-fiber
// brightness falloff every version of this node has used — kept as its own
// function so both the single-ply path and the multi-ply smooth-union path
// below can share it.
fn shape_from_edge(edge: f32, half_width: f32, roundness: f32) -> f32 {
    if half_width <= 0.0001 {
        return 0.0;
    }
    let d = clamp(1.0 - edge / half_width, 0.0, 1.0);
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
        let edge = half_w - abs(local_y);
        let value = shape_from_edge(edge, half_w, roundness);
        textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
        return;
    }

    // Pack `ply_count` strands evenly around a circle of `orbit_radius`,
    // sized so neighboring strands' circles just touch (a small deliberate
    // gap, not an overlap — a "tiny gap is ok" per the owner) rather than
    // each strand being nearly as wide as the whole orbit, which is what
    // read as a moire lattice in the first version of this. For N strands
    // evenly spaced around radius R, they touch when strand_radius ==
    // R * sin(PI / N); `orbit_radius + strand_half_width` reaching exactly
    // `half_w` keeps the whole bundle filling the thickness the slider
    // promises.
    let n = f32(ply_count);
    let touching_angle = PI / n;
    let orbit_radius = half_w / (1.0 + sin(touching_angle));
    let strand_half_width = orbit_radius * sin(touching_angle);

    // Each strand orbits the bundle's center as `uv.x` advances — an
    // integer number of full turns (`twist_cycles`) across this texture's
    // own width, so it stays exactly periodic. Direction just flips the
    // rotation's sign, mirroring the spiral.
    var sign = 1.0;
    if params.twist_direction == TWIST_Z {
        sign = -1.0;
    }
    let angle_base = sign * f32(params.twist_cycles) * uv.x * 2.0 * PI;

    // Two things combine the strands, and they're deliberately different
    // operations:
    //
    // Shape (does thread cover this pixel at all, and how much): a true
    // hard max() of each strand's own "how far inside am I" distance draws
    // a kink everywhere the dominant strand switches — a sine curve has a
    // continuous slope, but max() of several sine curves does not, so the
    // combined outline's slope jumps at every handoff. That's the ragged,
    // over-curvy silhouette a screenshot showed, not just an edge-alpha
    // issue — softening each strand's own edge alone (a prior attempt)
    // couldn't fix it, since the kink is in the *position* of the boundary,
    // not its sharpness. A smooth maximum (the standard SDF smooth-union:
    // log(sum(exp(x / softness))) * softness, which reduces to the regular
    // max as softness -> 0) keeps every contributing curve's own continuous
    // slope instead of switching between them, so the silhouette — both
    // each ply's own rounded top and the valley between two plies — comes
    // out as one flowing curve. Unlike inflating the strands to force
    // coverage (a prior attempt, which fused the plies into what read as
    // one connected tube with no visible separation), this only rounds the
    // *shape* of the valley — it can still reach all the way down to a true
    // gap between plies, which is the point.
    //
    // Shading (how bright is the winning strand): blended by relative
    // depth, `cos` of each strand's own orbit angle standing in for toward/
    // away-from-viewer depth (the other half of the sin/cos helix
    // projection `sin` already provides for position) — a soft pick
    // instead of a hard one, so the front/back handoff doesn't itself
    // create a second seam on top of the shape one above.
    let softness = max(strand_half_width * 0.18, 0.0001);
    var shape_sum_exp = 0.0;
    var shade_weighted_sum = 0.0;
    var shade_weight_total = 0.0;
    let occlusion_sharpness = 4.0;
    for (var k = 0u; k < ply_count; k = k + 1u) {
        let phase = 2.0 * PI * f32(k) / n;
        let angle = angle_base + phase;
        let center = orbit_radius * sin(angle);
        let depth = cos(angle);
        let edge = strand_half_width - abs(local_y - center);

        shape_sum_exp = shape_sum_exp + exp(edge / softness);

        let shade = mix(0.55, 1.0, depth * 0.5 + 0.5);
        let presence_weight = exp(edge / softness);
        let depth_weight = exp(depth * occlusion_sharpness);
        let weight = presence_weight * depth_weight;
        shade_weighted_sum = shade_weighted_sum + weight * shade;
        shade_weight_total = shade_weight_total + weight;
    }
    let combined_edge = log(shape_sum_exp) * softness;
    let shape = shape_from_edge(combined_edge, strand_half_width, roundness);
    // `shape` is already ~0 wherever every strand's edge weight underflowed
    // toward 0 together (e.g. near the texture's own top/bottom border) —
    // this floor just keeps that division finite instead of 0/0 = NaN; it
    // never visibly affects a pixel where any strand actually contributes.
    let blended_shade = shade_weighted_sum / max(shade_weight_total, 1e-20);
    let value = shape * blended_shade;

    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
