// SPDX-License-Identifier: MIT OR Apache-2.0
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
// One thread per row (horizontal pass) or column (vertical pass), each
// doing an exact sliding-window sum along its line — not subsampling at a
// stride. An earlier version subsampled at a bounded number of taps to
// keep cost independent of radius; for a sharp edge that produced a
// faint, periodic staircase in the output ramp instead of a smooth line
// (confirmed live by the owner: the resulting height map read as "ragged"
// under lighting/displacement even at full mesh tessellation, since the
// artifact was baked into the texture itself, not a geometry problem).
// A sliding window is both *exact* (a true average of every texel in the
// box, not an estimate) and just as cheap: each step is one add and one
// subtract regardless of how large the radius gets, so total cost per
// line is O(resolution), not O(resolution * radius) — there's no
// bounded-tap tradeoff to make here.

struct BoxBlurParams {
    resolution: u32,
    direction: u32,
    radius_px: f32,
    wrap: u32,
}

@group(0) @binding(0) var<uniform> params: BoxBlurParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

fn load_along(line: i32, pos: i32, size: i32, horizontal: bool, wrap: bool) -> vec4<f32> {
    let wrapped = tile_coord(pos, size, wrap);
    if horizontal {
        return textureLoad(input_tex, vec2<i32>(wrapped, line), 0);
    }
    return textureLoad(input_tex, vec2<i32>(line, wrapped), 0);
}

fn store_along(line: i32, pos: i32, horizontal: bool, value: vec4<f32>) {
    if horizontal {
        textureStore(output, vec2<i32>(pos, line), value);
    } else {
        textureStore(output, vec2<i32>(line, pos), value);
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = i32(params.resolution);
    if i32(id.x) >= size {
        return;
    }
    let line = i32(id.x);
    let horizontal = params.direction == 0u;
    let wrap = params.wrap != 0u;

    let radius = i32(round(max(params.radius_px, 0.0)));
    if radius < 1 {
        for (var pos = 0; pos < size; pos = pos + 1) {
            store_along(line, pos, horizontal, load_along(line, pos, size, horizontal, wrap));
        }
        return;
    }

    let window = 2.0 * f32(radius) + 1.0;
    var sum = vec4<f32>(0.0);
    for (var dx = -radius; dx <= radius; dx = dx + 1) {
        sum = sum + load_along(line, dx, size, horizontal, wrap);
    }
    store_along(line, 0, horizontal, sum / window);

    for (var pos = 1; pos < size; pos = pos + 1) {
        sum = sum + load_along(line, pos + radius, size, horizontal, wrap);
        sum = sum - load_along(line, pos - radius - 1, size, horizontal, wrap);
        store_along(line, pos, horizontal, sum / window);
    }
}
