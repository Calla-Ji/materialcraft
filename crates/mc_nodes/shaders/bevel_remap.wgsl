// bevel_remap.wgsl — second pass of Filters / Spatial / Bevel: reshapes an
// already-blurred mask/height map's edge transition with a power curve,
// controlled by `exponent` (derived from the node's Convexity param on the
// Rust side — see `bevel.rs`).
//
// Inputs:  `input_tex` — the *blurred* mask (see `bevel.rs`'s `dispatch`,
//          which runs the existing Blur node first).
// Output:  `output` — the same values raised to `exponent`. Convexity = 0
//          maps to exponent = 1 (no change — a plain blurred chamfer);
//          positive convexity gives an exponent < 1, which rises quickly
//          near 0 and flattens out near 1 (a rounded, dome-like profile);
//          negative convexity gives an exponent > 1, which stays low
//          longer and rises sharply near 1 (a scooped, concave profile).
// Params:  exponent — the power curve's exponent.
//
// Blurring first (rather than computing a true distance-to-edge field) is
// a standard, cheap approximation: far from any edge the blur hasn't
// reached, so the plateau on either side stays exactly 0 or 1 regardless
// of `exponent` (pow(0, e) = 0, pow(1, e) = 1) — only the transition band
// near an edge actually reshapes.

struct BevelRemapParams {
    resolution: u32,
    exponent: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: BevelRemapParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let value = textureLoad(input_tex, vec2<i32>(id.xy), 0);
    let remapped = pow(clamp(value.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(params.exponent));
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(remapped, value.a));
}
