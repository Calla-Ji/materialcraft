// SPDX-License-Identifier: MIT OR Apache-2.0
// bevel_remap.wgsl — second pass of Filters / Spatial / Bevel: reshapes how
// much the box blur (see `bevel_box_blur.wgsl`) changed each pixel, with a
// power curve controlled by `exponent` (derived from the node's Convexity
// param on the Rust side — see `bevel.rs`), then adds that reshaped amount
// back onto the *original*, unblurred value.
//
// Inputs:  `original_tex` — the node's real input, unblurred.
//          `blurred_tex` — the same input after the box blur.
// Output:  `output` — `original + sign(delta) * abs(delta)^exponent`, where
//          `delta = blurred - original`. Convexity = 0 maps to exponent = 1
//          (no reshaping at all — delta passes through unchanged, giving
//          the box blur's own plain linear chamfer); positive convexity
//          gives an exponent > 1, which suppresses small deltas and keeps
//          the plateau flat for longer, with the transition compressed
//          near the true edge (a rounded, dome-like profile); negative
//          convexity gives an exponent < 1, which amplifies small deltas
//          and spreads the transition out further from the edge (a
//          shallow, scooped, concave profile).
//
// Shaping *delta* rather than the blurred value directly (an earlier
// version did this, and read as "just a blur" once reshaped — see
// PROGRESS.md) guarantees two things no amount of tuning the first version
// could: at Distance = 0, blurred equals original exactly, so delta is
// exactly 0 everywhere and `sign(0) * 0^exponent = 0` regardless of
// exponent — Distance = 0 is a true no-op, not an approximate one. And for
// an input that already has its own non-flat shading baked in (e.g. a
// pattern generator's own per-tile fade) rather than starting from a clean
// 0/1 mask, anywhere the blur didn't change anything stays exactly as it
// was, instead of the whole value range getting reshaped by a power curve
// that assumed a 0..1 mask to begin with.

struct BevelRemapParams {
    resolution: u32,
    exponent: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: BevelRemapParams;
@group(0) @binding(1) var original_tex: texture_2d<f32>;
@group(0) @binding(2) var blurred_tex: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;

fn shape_delta(d: f32, exponent: f32) -> f32 {
    return sign(d) * pow(abs(d), exponent);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let original = textureLoad(original_tex, coord, 0);
    let blurred = textureLoad(blurred_tex, coord, 0);
    let delta = blurred.rgb - original.rgb;
    let shaped = vec3<f32>(
        shape_delta(delta.r, params.exponent),
        shape_delta(delta.g, params.exponent),
        shape_delta(delta.b, params.exponent),
    );
    let result = clamp(original.rgb + shaped, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(output, coord, vec4<f32>(result, original.a));
}
