// SPDX-License-Identifier: MIT OR Apache-2.0
// hsl_adjust.wgsl — Filters / Adjustments / HSL Adjust
//
// Inputs:  `input_tex` — a Color texture.
// Output:  `output` — hue rotated, saturation scaled, lightness shifted;
//          alpha passed through unchanged.
// Params:
//   hue_shift   — fraction of a full turn to rotate hue by
//   saturation  — multiplier on saturation (0 = grayscale, 1 = unchanged)
//   lightness   — additive shift on lightness

struct HslAdjustParams {
    resolution: u32,
    hue_shift: f32,
    saturation: f32,
    lightness: f32,
}

@group(0) @binding(0) var<uniform> params: HslAdjustParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

fn rgb_to_hsl(c: vec3<f32>) -> vec3<f32> {
    let max_c = max(c.r, max(c.g, c.b));
    let min_c = min(c.r, min(c.g, c.b));
    let l = (max_c + min_c) * 0.5;
    let delta = max_c - min_c;

    if delta < 0.0001 {
        return vec3<f32>(0.0, 0.0, l);
    }

    let s = select(
        delta / (2.0 - max_c - min_c),
        delta / (max_c + min_c),
        l < 0.5,
    );

    var h: f32;
    if max_c == c.r {
        h = (c.g - c.b) / delta + select(0.0, 6.0, c.g < c.b);
    } else if max_c == c.g {
        h = (c.b - c.r) / delta + 2.0;
    } else {
        h = (c.r - c.g) / delta + 4.0;
    }
    h = h / 6.0;

    return vec3<f32>(h, s, l);
}

fn hue_to_rgb(p: f32, q: f32, t_in: f32) -> f32 {
    var t = t_in;
    if t < 0.0 {
        t = t + 1.0;
    }
    if t > 1.0 {
        t = t - 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 0.5 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    return p;
}

fn hsl_to_rgb(hsl: vec3<f32>) -> vec3<f32> {
    let h = hsl.x;
    let s = hsl.y;
    let l = hsl.z;

    if s < 0.0001 {
        return vec3<f32>(l);
    }

    let q = select(l + s - l * s, l * (1.0 + s), l < 0.5);
    let p = 2.0 * l - q;

    return vec3<f32>(
        hue_to_rgb(p, q, h + 1.0 / 3.0),
        hue_to_rgb(p, q, h),
        hue_to_rgb(p, q, h - 1.0 / 3.0),
    );
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let value = textureLoad(input_tex, vec2<i32>(id.xy), 0);
    var hsl = rgb_to_hsl(value.rgb);
    hsl.x = fract(hsl.x + params.hue_shift);
    hsl.y = clamp(hsl.y * params.saturation, 0.0, 1.0);
    hsl.z = clamp(hsl.z + params.lightness, 0.0, 1.0);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(hsl_to_rgb(hsl), value.a));
}
