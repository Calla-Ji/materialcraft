// SPDX-License-Identifier: MIT OR Apache-2.0
// palette_dither.wgsl — Filters / Stylize / Palette Dither
//
// Inputs:  `image_tex` — the Color texture to dither down to a fixed
//          palette.
// Output:  `output` — for each pixel, the dither threshold nudges the color
//          slightly (scaled by `spread` and the palette's spacing) before
//          picking the nearest of up to 8 palette colors by `distance_mode`.
//          That nudge is what turns a smooth gradient into an interspersed
//          pattern of its two nearest palette colors instead of a hard,
//          banded edge.
// Params:  pattern/cells/seed — same meaning as Ordered Dither (minus Dot).
//          spread — dither strength. color_count/colors — the palette,
//          already sorted on the Rust side (stop position only decided
//          order, not blending). distance_mode — 0 RGB (plain Euclidean), 1
//          Perceptual (luminance-weighted).

struct PaletteDitherParams {
    resolution: u32,
    cells: u32,
    pattern: u32,
    color_count: u32,
    distance_mode: u32,
    spread: f32,
    seed: u32,
    _pad0: u32,
    colors: array<vec4<f32>, 8>,
}

@group(0) @binding(0) var<uniform> params: PaletteDitherParams;
@group(0) @binding(1) var image_tex: texture_2d<f32>;
@group(0) @binding(2) var blue_noise: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;

const PATTERN_BAYER_4X4: u32 = 1u;
const PATTERN_BAYER_8X8: u32 = 2u;
const PATTERN_BLUE_NOISE: u32 = 3u;
const BLUE_NOISE_SIZE: i32 = 64;
const DISTANCE_RGB: u32 = 0u;
const LUMINANCE_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

fn threshold_at(coords: vec2<i32>) -> f32 {
    let uv = (vec2<f32>(coords) + 0.5) / f32(params.resolution);
    let raw_cell = vec2<i32>(floor(uv * f32(params.cells)));

    if params.pattern == PATTERN_BLUE_NOISE {
        let offset = vec2<i32>(i32(params.seed) * 17, i32(params.seed) * 31);
        let texel = vec2<i32>(
            wrap_coord(raw_cell.x + offset.x, BLUE_NOISE_SIZE),
            wrap_coord(raw_cell.y + offset.y, BLUE_NOISE_SIZE),
        );
        return textureLoad(blue_noise, texel, 0).r;
    }

    var size: u32 = 2u;
    if params.pattern == PATTERN_BAYER_4X4 {
        size = 4u;
    } else if params.pattern == PATTERN_BAYER_8X8 {
        size = 8u;
    }
    return bayer_threshold(vec2<u32>(raw_cell), size);
}

fn color_distance(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let diff = a - b;
    if params.distance_mode == DISTANCE_RGB {
        return dot(diff, diff);
    }
    return dot(diff * diff, LUMINANCE_WEIGHTS);
}

fn nearest_palette_color(color: vec3<f32>) -> vec3<f32> {
    var best = params.colors[0].rgb;
    var best_dist = color_distance(color, best);
    for (var i = 1u; i < params.color_count; i = i + 1u) {
        let candidate = params.colors[i].rgb;
        let d = color_distance(color, candidate);
        if d < best_dist {
            best_dist = d;
            best = candidate;
        }
    }
    return best;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coords = vec2<i32>(id.xy);
    let original = textureLoad(image_tex, coords, 0);
    let t = threshold_at(coords);

    // Scaled by the palette's own spacing (1 / color_count) so the nudge is
    // meaningful regardless of how many colors are in the palette — the
    // same role `levels` plays in Ordered Dither's quantization step size.
    let step = 1.0 / f32(max(params.color_count, 2u));
    let noise = (t - 0.5) * params.spread * step;
    let perturbed = clamp(original.rgb + vec3<f32>(noise), vec3<f32>(0.0), vec3<f32>(1.0));

    let result = nearest_palette_color(perturbed);
    textureStore(output, coords, vec4<f32>(result, original.a));
}
