// SPDX-License-Identifier: MIT OR Apache-2.0
// ordered_dither.wgsl — Filters / Stylize / Ordered Dither
//
// Inputs:  `image_tex` — the texture to dither (Gray or Color — both are
//          just RGBA on the GPU, see `mc_nodes::defs::effective_output_type`;
//          for a Gray-sourced image, Per Channel and Luminance modes produce
//          identical results since R=G=B already, matching the brief's
//          "(Color input only)" note on `mode` without any special-casing).
// Output:  `output` — `image_tex` quantized to `levels` tones per channel
//          and dithered by `pattern`'s threshold, mixed in by `spread`
//          (0 = plain posterize, 1 = full dither) and shifted by `bias`.
// Params:  pattern/cells/seed — which threshold mask to dither with, same
//          meaning as the Dither Pattern node (minus its Dot option, which
//          doesn't apply here). levels — tone steps per channel (2 = pure
//          1-bit). spread — dither strength. bias — shifts the threshold
//          lighter/darker. mode — Per Channel dithers each of R/G/B
//          independently; Luminance quantizes brightness only and rescales
//          all 3 channels together, preserving hue.

struct OrderedDitherParams {
    resolution: u32,
    cells: u32,
    pattern: u32,
    levels: i32,
    spread: f32,
    bias: f32,
    mode: u32,
    seed: u32,
}

@group(0) @binding(0) var<uniform> params: OrderedDitherParams;
@group(0) @binding(1) var image_tex: texture_2d<f32>;
@group(0) @binding(2) var blue_noise: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;

const PATTERN_BAYER_4X4: u32 = 1u;
const PATTERN_BAYER_8X8: u32 = 2u;
const PATTERN_BLUE_NOISE: u32 = 3u;
const BLUE_NOISE_SIZE: i32 = 64;
const MODE_PER_CHANNEL: u32 = 0u;

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

// Posterizes `value` to one of `levels` evenly-spaced steps, nudged by the
// dither threshold `t` first: `spread` blends between a plain round (0) and
// the full dither-matrix range (1); `bias` shifts the effective threshold.
// Rounding *before* dividing back down is what makes a flat input produce a
// clean N-level split rather than a smoothly-ramped one.
fn dither_channel(value: f32, t: f32) -> f32 {
    let levels_f = f32(max(params.levels, 2));
    let scaled = value * (levels_f - 1.0);
    let noise = (t - 0.5 + params.bias) * params.spread;
    let quantized = round(scaled + noise);
    return clamp(quantized / (levels_f - 1.0), 0.0, 1.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coords = vec2<i32>(id.xy);
    let original = textureLoad(image_tex, coords, 0);
    let t = threshold_at(coords);

    var result_rgb: vec3<f32>;
    if params.mode == MODE_PER_CHANNEL {
        result_rgb = vec3<f32>(
            dither_channel(original.r, t),
            dither_channel(original.g, t),
            dither_channel(original.b, t),
        );
    } else {
        let luminance = dot(original.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        let quantized_luminance = dither_channel(luminance, t);
        if luminance < 0.0001 {
            result_rgb = vec3<f32>(quantized_luminance, quantized_luminance, quantized_luminance);
        } else {
            result_rgb = original.rgb * (quantized_luminance / luminance);
        }
    }

    textureStore(
        output,
        coords,
        vec4<f32>(clamp(result_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), original.a),
    );
}
