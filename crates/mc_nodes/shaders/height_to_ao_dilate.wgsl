// SPDX-License-Identifier: MIT OR Apache-2.0
// height_to_ao_dilate.wgsl — one separable pass of a *falloff-weighted* max
// filter: each output texel becomes the highest nearby height found,
// attenuated by a Gaussian falloff with distance (same curve blur.wgsl
// uses), rather than a flat average. Used by Height to AO instead of a
// blur: averaging dilutes a thin, sharp feature (e.g. a leaf shape's
// pointed tip) down to almost nothing once it's surrounded mostly by low
// background, so a blur-based comparison misses occlusion exactly where a
// shape narrows to a point. Taking a *max* instead has no such dilution — a
// tall neighbor right next door registers at (near) full strength no matter
// how little area it covers — but a max with no distance weighting at all
// instead jumps straight from "fully occluded" to "fully clear" right at
// the filter's radius, the same hard-edge problem a single-radius approach
// already had once before. Attenuating each sample by the same Gaussian
// curve blur.wgsl uses keeps the no-dilution property (full strength at
// distance 0) while still fading smoothly to ~0 by the radius, and —
// because a Gaussian is one of the few functions that separates exactly —
// composes correctly across this pass's two calls (horizontal then
// vertical, from `mc_nodes::height_to_ao::dispatch`) into a true
// radially-symmetric falloff, not a squarish approximation of one.
//
// Params:
//   direction — 0 = horizontal pass, 1 = vertical pass
//   radius_px — how far out to look, in pixels

struct DilateParams {
    resolution: u32,
    direction: u32,
    radius_px: f32,
    wrap: u32,
}

@group(0) @binding(0) var<uniform> params: DilateParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let size = i32(params.resolution);
    // Same adaptive step/tap-count shape as blur.wgsl, so cost stays
    // bounded (32 taps max) regardless of how large radius_px gets.
    let sigma = max(params.radius_px, 0.5);
    let step = max(sigma / 16.0, 1.0);
    let taps = clamp(i32(ceil(sigma * 2.0 / step)), 1, 32);

    var highest = textureLoad(input_tex, vec2<i32>(id.xy), 0).r;
    for (var i = 1; i <= taps; i = i + 1) {
        let dist = f32(i) * step;
        let falloff = exp(-(dist * dist) / (2.0 * sigma * sigma));
        let offset = i32(round(dist));

        var coord_pos = vec2<i32>(id.xy);
        var coord_neg = vec2<i32>(id.xy);
        let wrap = params.wrap != 0u;
        if params.direction == 0u {
            coord_pos.x = tile_coord(coord_pos.x + offset, size, wrap);
            coord_neg.x = tile_coord(coord_neg.x - offset, size, wrap);
        } else {
            coord_pos.y = tile_coord(coord_pos.y + offset, size, wrap);
            coord_neg.y = tile_coord(coord_neg.y - offset, size, wrap);
        }
        let neighbor = max(
            textureLoad(input_tex, coord_pos, 0).r,
            textureLoad(input_tex, coord_neg, 0).r,
        );
        highest = max(highest, neighbor * falloff);
    }
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(vec3<f32>(highest), 1.0));
}
