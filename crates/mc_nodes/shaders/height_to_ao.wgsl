// height_to_ao.wgsl — Converters / Height to AO
//
// Inputs:  `input_tex` — the original height map, plus three pre-blurred
//          copies of it at progressively larger radii (fine/medium/broad —
//          see `height_to_ao.rs`'s `dispatch`, which produces them via the
//          Blur node before this pass runs).
// Output:  `output` — darker wherever a pixel sits notably below its local
//          blurred-neighborhood average at *any* of the three scales
//          (a crevice relative to its surroundings), unchanged (white)
//          where it's at or above every scale's local average.
// Params:
//   intensity — multiplier on the occlusion strength
//
// Comparing a pixel to a *blurred* version of the height map (rather than a
// handful of discrete directional ray samples) means every pixel within
// reach contributes to the average smoothly, with no "gaps" between sample
// directions — so occlusion grades evenly around a contour of any shape,
// instead of only picking up the specific few directions/distances a ray
// happened to sample. Comparing against three different blur radii at once
// (instead of one) layers fine crevice detail with broader-scale shadowing,
// the same way combining several octaves of noise adds richness a single
// scale can't.

struct HeightToAoParams {
    resolution: u32,
    intensity: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: HeightToAoParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var blurred_fine: texture_2d<f32>;
@group(0) @binding(3) var blurred_medium: texture_2d<f32>;
@group(0) @binding(4) var blurred_broad: texture_2d<f32>;
@group(0) @binding(5) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let h = textureLoad(input_tex, coord, 0).r;
    let fine = textureLoad(blurred_fine, coord, 0).r;
    let medium = textureLoad(blurred_medium, coord, 0).r;
    let broad = textureLoad(blurred_broad, coord, 0).r;

    // The strongest-occluding scale wins, rather than averaging the three —
    // averaging would wash out a real crevice that only one scale's radius
    // happens to "see," since the other two scales correctly report ~0
    // there.
    let occlusion = max(fine - h, max(medium - h, broad - h));

    let ao = clamp(1.0 - max(occlusion, 0.0) * params.intensity, 0.0, 1.0);
    textureStore(output, coord, vec4<f32>(vec3<f32>(ao), 1.0));
}
