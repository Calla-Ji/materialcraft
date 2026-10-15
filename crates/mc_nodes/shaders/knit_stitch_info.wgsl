// SPDX-License-Identifier: MIT OR Apache-2.0
// knit_stitch_info.wgsl — per-pixel Stitch ID / Stitch Type for the Knit
// node; a second pass run after tube_raster.wgsl has filled in
// height/mask/ply_id/cell for the same tile (mc_nodes::knit::dispatch runs
// both in sequence on the same GPU queue, which guarantees this pass's
// reads see that one's writes — no explicit barrier needed).
//
// Every pixel belongs to exactly one "stitch cell" (wale, course) — either
// the capsule that actually covers it, decoded from tube_raster's own
// `cell` output (packed the same way mc_nodes::knit's Segment tags are:
// (course*wales+wale)*8+ply, recovered here via cell/8), or — in a gap
// between loops — the cell a geometric fallback says the pixel sits in
// (KNIT_NODE_SPEC.md §7: "Holes take the stitch cell they sit in"), ported
// from knit_reference.py's own render().
//
// Stitch ID uses knit_reference.py's own hash01() integer hash, bit for
// bit (KNIT_NODE_SPEC.md §7: "use the same integer hash in WGSL").
//
// No Colour sampling here: the spec's own pixelate() (per-stitch nearest
// sampling of a Colour input) was cut from the shipped node as redundant
// with Uniform Color + Blend + a generic Color output — see
// mc_nodes::knit's own module doc comment.

struct StitchParams {
    resolution: u32,
    wales: u32,
    courses: u32,
    seed: u32,
}

@group(0) @binding(0) var<uniform> params: StitchParams;
@group(0) @binding(1) var<storage, read> widths_cumsum: array<f32>;
@group(0) @binding(2) var<storage, read> chart_is_knit: array<u32>;
@group(0) @binding(3) var cell_in: texture_2d<u32>;
@group(0) @binding(4) var stitch_id_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var stitch_type_out: texture_storage_2d<rgba16float, write>;

const CELL_MISS: u32 = 0xFFFFFFFFu;

// knit_reference.py's own hash01() — now shared via common.wgsl (also used
// by Stitch and Gabor Noise), which `full_source` always prepends.

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let px = (f32(id.x) + 0.5) / f32(params.resolution);
    let py = (f32(id.y) + 0.5) / f32(params.resolution);

    let tag = textureLoad(cell_in, vec2<i32>(id.xy), 0).r;

    var wale: u32;
    var course: u32;
    if tag != CELL_MISS {
        let cell = tag / 8u;
        wale = cell % params.wales;
        course = cell / params.wales;
    } else {
        // Gap fallback: which wale/course band this pixel's own position
        // sits in (knit_reference.py's own gx/gc).
        var gx = params.wales - 1u;
        for (var w = 0u; w < params.wales; w = w + 1u) {
            if px <= widths_cumsum[w] {
                gx = w;
                break;
            }
        }
        wale = gx;
        course = min(u32(floor(py * f32(params.courses))), params.courses - 1u);
    }

    let sid = hash01(wale, course, params.seed);
    let stype = f32(chart_is_knit[course * params.wales + wale]);

    textureStore(stitch_id_out, vec2<i32>(id.xy), vec4<f32>(sid, sid, sid, 1.0));
    textureStore(stitch_type_out, vec2<i32>(id.xy), vec4<f32>(stype, stype, stype, 1.0));
}
