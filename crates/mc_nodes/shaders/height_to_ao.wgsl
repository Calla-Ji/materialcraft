// SPDX-License-Identifier: MIT OR Apache-2.0
// height_to_ao.wgsl — Converters / Height to AO
//
// Inputs:  `input_tex` — the original height map, plus three max-filtered
//          ("dilated") copies of it at progressively larger radii
//          (fine/medium/broad — see `height_to_ao.rs`'s `dispatch`, which
//          produces them via `height_to_ao_dilate.wgsl` before this pass
//          runs).
// Output:  `output` — darker wherever a pixel sits notably below the
//          *tallest* neighbor within reach at *any* of the three scales,
//          unchanged (white) where it's at or above every scale's tallest
//          neighbor.
// Params:
//   intensity — multiplier on the occlusion strength
//
// Comparing a pixel against the *tallest* nearby height (a max filter),
// rather than an *average* of nearby heights (a blur) or a handful of
// discrete directional ray samples, avoids two different failure modes a
// simpler approach hit in practice (see PROGRESS.md): a blurred average
// dilutes a thin, sharp feature (e.g. a leaf shape's pointed tip) down to
// almost nothing once it's surrounded mostly by low background, leaving a
// visible gap in the occlusion right at that tip; a few discrete ray
// samples have gaps between the sampled directions that show up as a hard
// ring tracing a sharp edge instead of a soft falloff. A max filter has
// neither problem — a nearby tall neighbor registers at full strength no
// matter how little area it covers, every neighbor within reach is
// considered rather than just a handful of sampled directions, and
// comparing against three filter radii at once layers fine crevice detail
// with broader-scale shadowing, the same way combining several octaves of
// noise adds richness a single scale can't.

struct HeightToAoParams {
    resolution: u32,
    intensity: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: HeightToAoParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var dilated_fine: texture_2d<f32>;
@group(0) @binding(3) var dilated_medium: texture_2d<f32>;
@group(0) @binding(4) var dilated_broad: texture_2d<f32>;
@group(0) @binding(5) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let h = textureLoad(input_tex, coord, 0).r;
    let fine = textureLoad(dilated_fine, coord, 0).r;
    let medium = textureLoad(dilated_medium, coord, 0).r;
    let broad = textureLoad(dilated_broad, coord, 0).r;

    // The strongest-occluding scale wins, rather than averaging the three —
    // averaging would wash out a real crevice that only one scale's radius
    // happens to "see," since the other two scales correctly report ~0
    // there.
    let occlusion = max(fine - h, max(medium - h, broad - h));

    let ao = clamp(1.0 - max(occlusion, 0.0) * params.intensity, 0.0, 1.0);
    textureStore(output, coord, vec4<f32>(vec3<f32>(ao), 1.0));
}
