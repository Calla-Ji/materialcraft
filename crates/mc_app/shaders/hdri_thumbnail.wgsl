// SPDX-License-Identifier: MIT OR Apache-2.0
// hdri_thumbnail.wgsl — downsamples an HDRI equirectangular environment map
// (linear, unbounded HDR values) into a small sRGB-encoded preview for the
// Environment panel, so the artist can see which HDRI is active without
// switching the 3D viewport's shading mode or orbiting to find the sun.
//
// Unlike display_encode.wgsl (a plain linear-to-sRGB encode for already
// 0..1 node output textures), an HDRI can have values well above 1.0 (a sun
// disc, a bright window) — encoding those directly would just clip to
// white. Reinhard tonemapping (color / (1 + color)) compresses the full
// HDR range into 0..1 first, the same shape of curve (if not the exact
// formula) `viewport3d.wgsl` already uses for the main render, so the
// thumbnail's relative brightness reads consistently with the viewport.

struct ThumbnailParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: ThumbnailParams;
@group(0) @binding(1) var env_sampler: sampler;
@group(0) @binding(2) var source_tex: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba8unorm, write>;

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(f32(params.width), f32(params.height));
    let hdr = textureSampleLevel(source_tex, env_sampler, uv, 0.0).rgb;
    let tonemapped = hdr / (vec3<f32>(1.0) + hdr);
    let encoded = vec3<f32>(
        linear_to_srgb(tonemapped.r),
        linear_to_srgb(tonemapped.g),
        linear_to_srgb(tonemapped.b),
    );
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(encoded, 1.0));
}
