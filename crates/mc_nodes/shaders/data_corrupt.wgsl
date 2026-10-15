// SPDX-License-Identifier: MIT OR Apache-2.0
// data_corrupt.wgsl — Filters / Glitch / Data Corrupt
//
// Inputs:  `image_tex` — the texture to corrupt (Gray or Color — both are
//          just RGBA on the GPU, see `mc_nodes::defs::effective_output_type`).
//          `mask_tex` — defaults to solid white (fully visible) when nothing
//          is connected; see `mc_nodes::data_corrupt::dispatch`.
// Output:  `output` — divided into a `columns` x `rows` block grid, same as
//          Block Shift; a seeded hash per block decides whether it's
//          corrupted at all (`amount` is the fraction that are) and, for
//          Mix, which of the 3 underlying effects it gets.
// Params:  columns/rows — the block grid. amount — fraction of blocks
//          affected. effect — Posterize (crushes each channel to 3 levels),
//          Channel Swap (a random RGB channel permutation per block),
//          Invert, or Mix (randomly one of the three, per block). seed —
//          reseeds which blocks are affected and how.

struct DataCorruptParams {
    resolution: u32,
    columns: i32,
    rows: i32,
    effect: u32,
    amount: f32,
    seed: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: DataCorruptParams;
@group(0) @binding(1) var image_tex: texture_2d<f32>;
@group(0) @binding(2) var mask_tex: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;

const EFFECT_POSTERIZE: u32 = 0u;
const EFFECT_CHANNEL_SWAP: u32 = 1u;
const EFFECT_INVERT: u32 = 2u;
const EFFECT_MIX: u32 = 3u;
const POSTERIZE_LEVELS: f32 = 3.0;

fn apply_channel_swap(c: vec3<f32>, variant: u32) -> vec3<f32> {
    switch variant {
        case 0u: { return c.rgb; }
        case 1u: { return c.rbg; }
        case 2u: { return c.grb; }
        case 3u: { return c.gbr; }
        case 4u: { return c.brg; }
        default: { return c.bgr; }
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coords = vec2<i32>(id.xy);
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);
    let block_cell = vec2<i32>(floor(uv * vec2<f32>(f32(params.columns), f32(params.rows))));
    let original = textureLoad(image_tex, coords, 0);

    var corrupted = original.rgb;
    if hash2(block_cell, params.seed) < params.amount {
        var effect = params.effect;
        if effect == EFFECT_MIX {
            let pick = hash2(block_cell, params.seed + 4007u);
            if pick < 0.3333 {
                effect = EFFECT_POSTERIZE;
            } else if pick < 0.6667 {
                effect = EFFECT_CHANNEL_SWAP;
            } else {
                effect = EFFECT_INVERT;
            }
        }

        if effect == EFFECT_POSTERIZE {
            corrupted = round(original.rgb * (POSTERIZE_LEVELS - 1.0)) / (POSTERIZE_LEVELS - 1.0);
        } else if effect == EFFECT_CHANNEL_SWAP {
            let variant = u32(hash2(block_cell, params.seed + 5003u) * 6.0) % 6u;
            corrupted = apply_channel_swap(original.rgb, variant);
        } else {
            corrupted = 1.0 - original.rgb;
        }
    }

    let mask = textureLoad(mask_tex, coords, 0).r;
    let result = mix(original.rgb, corrupted, mask);
    textureStore(output, coords, vec4<f32>(result, original.a));
}
