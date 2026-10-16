// SPDX-License-Identifier: MIT OR Apache-2.0
// display_encode.wgsl — encodes a linear-space source texture to sRGB gamma
// for on-screen display, optionally isolating a single channel first.
//
// Every node's output texture is stored linear (CLAUDE.md §8: sRGB encoding
// only happens at specific boundaries). egui's own renderer assumes any
// texture it's handed is *already* gamma-encoded and never decodes it on
// sample (confirmed by reading egui-wgpu's own shader) — so a node's linear
// bytes, registered with egui directly, display too dark/off. This pass
// produces the sRGB-encoded copy egui actually needs, using the exact same
// piecewise OETF `mc_export::linear_to_srgb_u8` uses, so the 2D View/
// thumbnails match an exported PNG pixel-for-pixel (modulo 8-bit rounding).
//
// `channel` (CLAUDE.md §9's "channel view (RGB / R / G / B / A)", built as
// part of the 2D View's own toolbar) picks a single channel to broadcast
// across R/G/B instead of passing all three through — the only way to
// actually judge e.g. a packed ORM texture's Roughness channel on its own,
// since the combined RGB view of three unrelated grayscale channels
// necessarily reads as an arbitrary, not-meant-to-look-sensible color.
// Alpha is broadcast raw (no sRGB) when isolated, same as every other
// place in this codebase that treats alpha as linear coverage, not a
// gamma-encoded color channel (`mc_export::apply_color_space` skips it too).
//
// `is_sdf` renders an Sdf-typed source's iso-band preview instead
// (docs/hardsurface_nodes/HARDSURFACE_NODES.md §2) — a raw float distance
// value must never be shown as plain grey (it would look identical to a
// completely different distance a short way off), so this shows repeating
// bands every 1/200 tile unit plus a highlighted zero contour, ignoring
// `channel` (an Sdf port is always single-channel; there's nothing to
// isolate). Not gamma-encoded — like the raw Alpha branch above, this is
// a synthetic visualization already in display space, not a decoded/
// re-encoded physical color.

struct DisplayEncodeParams {
    resolution: u32,
    channel: u32, // 0 = RGB, 1 = R, 2 = G, 3 = B, 4 = A
    is_sdf: u32, // 1 = render the Sdf iso-band preview instead
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: DisplayEncodeParams;
@group(0) @binding(1) var source_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

const SDF_BAND_SCALE: f32 = 200.0; // spec: frac(sdf * 200)
// A clearly-distinct accent for the zero contour, never produced by the
// plain grey band ramp itself, so it reads as "this is the edge" at a
// glance instead of blending into a coincidentally-similar band shade.
const SDF_ZERO_CONTOUR: vec3<f32> = vec3<f32>(0.25, 0.55, 1.0);

fn encode_sdf_preview(sdf: f32, resolution: u32) -> vec3<f32> {
    let banded_grey = vec3<f32>(fract(sdf * SDF_BAND_SCALE));
    // ~1.5-pixel-wide anti-aliased contour around sdf == 0, expressed in
    // tile units (the same resolution-independence convention `coverage()`'s
    // own 1-pixel AA ramp uses) so it reads as ~1-2 screen pixels wide at
    // any graph resolution, not a fixed fraction of the tile.
    let half_width = 1.5 / f32(resolution);
    let contour = 1.0 - smoothstep(0.0, half_width, abs(sdf));
    return mix(banded_grey, SDF_ZERO_CONTOUR, contour);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coords = vec2<i32>(id.xy);
    let src = textureLoad(source_tex, coords, 0);

    if params.is_sdf == 1u {
        let preview = encode_sdf_preview(src.r, params.resolution);
        textureStore(output, coords, vec4<f32>(preview, 1.0));
        return;
    }

    if params.channel == 4u {
        // Alpha, raw — no sRGB, and always fully opaque so the isolated
        // channel is actually visible regardless of the source's own alpha.
        textureStore(output, coords, vec4<f32>(vec3<f32>(src.a), 1.0));
        return;
    }
    if params.channel >= 1u && params.channel <= 3u {
        let value = select(select(src.b, src.g, params.channel == 2u), src.r, params.channel == 1u);
        let encoded = linear_to_srgb(value);
        textureStore(output, coords, vec4<f32>(vec3<f32>(encoded), 1.0));
        return;
    }

    let encoded = vec3<f32>(
        linear_to_srgb(src.r),
        linear_to_srgb(src.g),
        linear_to_srgb(src.b),
    );
    textureStore(output, coords, vec4<f32>(encoded, src.a));
}
