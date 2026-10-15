// SPDX-License-Identifier: MIT OR Apache-2.0
// channel_pack.wgsl — Converters / Channel Packer
//
// Inputs:  `r_tex`, `g_tex`, `b_tex` — three grayscale inputs (each reads
//          its own red channel, same convention every other Gray-consuming
//          node already uses). Unconnected inputs default to white/mid-
//          grey/black respectively — see `mc_nodes::channel_pack::dispatch`.
// Output:  `output` — one texture with `r_tex`/`g_tex`/`b_tex` packed into
//          its R/G/B channels, alpha always 1. The Unreal Engine 5 export
//          preset's own ORM packing (R=AO, G=Roughness, B=Metallic) is
//          this node's main motivating use, but it's a generic 3-into-1
//          packer, not ORM-specific.
// Params:  none.

struct ChannelPackParams {
    resolution: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: ChannelPackParams;
@group(0) @binding(1) var r_tex: texture_2d<f32>;
@group(0) @binding(2) var g_tex: texture_2d<f32>;
@group(0) @binding(3) var b_tex: texture_2d<f32>;
@group(0) @binding(4) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let r = textureLoad(r_tex, coords, 0).r;
    let g = textureLoad(g_tex, coords, 0).r;
    let b = textureLoad(b_tex, coords, 0).r;

    textureStore(output, coords, vec4<f32>(r, g, b, 1.0));
}
