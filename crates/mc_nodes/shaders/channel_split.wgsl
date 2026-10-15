// SPDX-License-Identifier: MIT OR Apache-2.0
// channel_split.wgsl — Filters / Glitch / Channel Split
//
// Inputs:  `image_tex` — the texture to split.
//          `mask_tex` — defaults to solid white (fully visible) when
//          nothing is connected; see `mc_nodes::channel_split::dispatch`.
// Output:  `output` — each of R/G/B resampled at its own UV-space offset,
//          wrap-addressed so it stays tileable; alpha passes through
//          unchanged. Mixed back toward the untouched image by
//          `mask * strength`.
// Params:  offset_r/offset_g/offset_b — per-channel UV-space sample offset.
//          strength — overall effect strength, in 0..1.

struct ChannelSplitParams {
    resolution: u32,
    strength: f32,
    offset_r: vec2<f32>,
    offset_g: vec2<f32>,
    offset_b: vec2<f32>,
    tiling_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: ChannelSplitParams;
@group(0) @binding(1) var image_tex: texture_2d<f32>;
@group(0) @binding(2) var mask_tex: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;

fn sample_channel(coords: vec2<i32>, offset_uv: vec2<f32>, size: i32, channel: u32) -> f32 {
    let offset_px = vec2<i32>(round(offset_uv * f32(size)));
    let sample_coord = vec2<i32>(
        tile_coord(coords.x + offset_px.x, size, wraps_x(params.tiling_mode)),
        tile_coord(coords.y + offset_px.y, size, wraps_y(params.tiling_mode)),
    );
    let texel = textureLoad(image_tex, sample_coord, 0);
    switch channel {
        case 0u: { return texel.r; }
        case 1u: { return texel.g; }
        default: { return texel.b; }
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coords = vec2<i32>(id.xy);
    let size = i32(params.resolution);

    let original = textureLoad(image_tex, coords, 0);
    let split_color = vec3<f32>(
        sample_channel(coords, params.offset_r, size, 0u),
        sample_channel(coords, params.offset_g, size, 1u),
        sample_channel(coords, params.offset_b, size, 2u),
    );

    let mask = textureLoad(mask_tex, coords, 0).r;
    let amount = clamp(mask * params.strength, 0.0, 1.0);
    let result = mix(original.rgb, split_color, amount);

    textureStore(output, coords, vec4<f32>(result, original.a));
}
