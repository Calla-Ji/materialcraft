// bevel_box_blur.wgsl — one separable pass of a *uniform* box blur, for
// Filters / Spatial / Bevel's first stage. Deliberately not the existing
// Blur node's Gaussian kernel: a Gaussian has no sharp start or end to its
// falloff, so no matter how the result is reshaped afterward, it still
// reads as a generic soft blur rather than a defined bevel ramp. A box
// blur of a hard step produces an exactly *linear* ramp over a width equal
// to the box's own diameter — a plain chamfer by construction — which the
// Convexity remap (`bevel_remap.wgsl`) then curves into a dome or scoop.
// Outside that ramp width, the plateau on either side is completely
// undisturbed, the same guarantee the old Gaussian version had.
//
// Params:
//   direction — 0 = horizontal pass, 1 = vertical pass
//   radius_px — half the box's width, in pixels (so the full ramp width is
//               2*radius_px, matching the node's Distance param reaching
//               that far from an edge)
//
// Subsamples at a resolution-independent stride (same reasoning as
// blur.wgsl's adaptive step) rather than averaging every single texel in
// the box, so cost stays bounded regardless of how large radius_px gets —
// since every sample in a box blur carries equal weight, subsampling is an
// unbiased estimate of the true average, not a biased approximation.

struct BoxBlurParams {
    resolution: u32,
    direction: u32,
    radius_px: f32,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> params: BoxBlurParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let size = i32(params.resolution);
    let radius = max(params.radius_px, 0.0);
    if radius < 0.5 {
        textureStore(output, vec2<i32>(id.xy), textureLoad(input_tex, vec2<i32>(id.xy), 0));
        return;
    }
    let step = max(radius / 32.0, 1.0);
    let taps = clamp(i32(round(radius / step)), 1, 32);

    var sum = textureLoad(input_tex, vec2<i32>(id.xy), 0);
    var count = 1.0;
    for (var i = 1; i <= taps; i = i + 1) {
        let dist = i32(round(f32(i) * step));
        var coord_pos = vec2<i32>(id.xy);
        var coord_neg = vec2<i32>(id.xy);
        if params.direction == 0u {
            coord_pos.x = wrap_coord(coord_pos.x + dist, size);
            coord_neg.x = wrap_coord(coord_neg.x - dist, size);
        } else {
            coord_pos.y = wrap_coord(coord_pos.y + dist, size);
            coord_neg.y = wrap_coord(coord_neg.y - dist, size);
        }
        sum = sum + textureLoad(input_tex, coord_pos, 0) + textureLoad(input_tex, coord_neg, 0);
        count = count + 2.0;
    }
    textureStore(output, vec2<i32>(id.xy), sum / count);
}
