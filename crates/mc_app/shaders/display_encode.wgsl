// display_encode.wgsl — encodes a linear-space source texture to sRGB gamma
// for on-screen display.
//
// Every node's output texture is stored linear (CLAUDE.md §8: sRGB encoding
// only happens at specific boundaries). egui's own renderer assumes any
// texture it's handed is *already* gamma-encoded and never decodes it on
// sample (confirmed by reading egui-wgpu's own shader) — so a node's linear
// bytes, registered with egui directly, display too dark/off. This pass
// produces the sRGB-encoded copy egui actually needs, using the exact same
// piecewise OETF `mc_export::linear_to_srgb_u8` uses, so the 2D View/
// thumbnails match an exported PNG pixel-for-pixel (modulo 8-bit rounding).

struct DisplayEncodeParams {
    resolution: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coords = vec2<i32>(id.xy);
    let src = textureLoad(source_tex, coords, 0);
    let encoded = vec3<f32>(
        linear_to_srgb(src.r),
        linear_to_srgb(src.g),
        linear_to_srgb(src.b),
    );
    textureStore(output, coords, vec4<f32>(encoded, src.a));
}
