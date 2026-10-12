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

    // One combined weight per strand decides both how much it contributes
    // to the *shape* (does thread cover this pixel, how much) and to the
    // *shading* (how bright) — tried as two separate operations before
    // (geometric presence alone for shape, depth alone for shading) and
    // that was wrong: with 3+ strands, several can be geometrically present
    // at the same pixel at once with no clear front/back winner, and a
    // depth-blind shape union lets *all* of them bulge the silhouette at
    // once — the exact moire/lattice interference pattern this node
    // started with, now reappearing for ply_count 3+ even though the
    // ply_count 2 case (where at most one strand is ever meaningfully
    // ahead of the other) looked fine. A real twisted yarn doesn't show
    // every strand's cross-section at every point — only the one currently
    // facing the viewer does, with its neighbors tucked behind it — so
    // depth has to gate the *shape* too, not just the brightness.
    //
    // `priority = edge / softness + depth * occlusion_sharpness` folds
    // "how geometrically present" and "how far forward" into one score per
    // strand; the softmax-weighted average of the strands' own edges by
    // that score is a smooth maximum biased toward whichever strand is
    // both present and in front. Two strands with a similar edge value
    // (an actual crossing) still blend smoothly between them — keeping the
    // continuous-slope silhouette the smooth-union rewrite was for — but a
    // strand that's merely geometrically nearby while *behind* a strand
    // that's already dominant no longer gets an independent vote in the
    // shape, which is what was missing.
    let softness = max(strand_half_width * 0.18, 0.0001);
    let occlusion_sharpness = 4.0;
    var weight_sum = 0.0;
    var edge_weighted_sum = 0.0;
    var shade_weighted_sum = 0.0;
    for (var k = 0u; k < ply_count; k = k + 1u) {
        let phase = 2.0 * PI * f32(k) / n;
        let angle = angle_base + phase;
        let center = orbit_radius * sin(angle);
        let depth = cos(angle);
        let edge = strand_half_width - abs(local_y - center);
        let shade = mix(0.55, 1.0, depth * 0.5 + 0.5);

        let priority = edge / softness + depth * occlusion_sharpness;
        let weight = exp(priority);
        weight_sum = weight_sum + weight;
        edge_weighted_sum = edge_weighted_sum + weight * edge;
        shade_weighted_sum = shade_weighted_sum + weight * shade;
    }
    // `weight_sum` is a sum of `exp(...)` terms, so it's always strictly
    // positive — no 0/0 risk the way a hard presence gate would have.
    let combined_edge = edge_weighted_sum / weight_sum;
    let blended_shade = shade_weighted_sum / weight_sum;
    let shape = shape_from_edge(combined_edge, strand_half_width, roundness);
    let value = shape * blended_shade;

    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
