// blend.wgsl — Filters / Blending & Masking / Blend
//
// Inputs:  `base_tex`, `blend_tex` — the two layers to combine.
//          `mask_tex` — defaults to solid white (fully visible) when nothing
//          is connected; see `mc_nodes::blend::dispatch`.
// Output:  `output` — base combined with blend per `mode`, mixed in by
//          mask * opacity.
// Params (via the `params` uniform):
//   mode    — 0 Copy, 1 Add, 2 Subtract, 3 Multiply, 4 Screen, 5 Overlay,
//             6 Max, 7 Min, 8 Difference
//   opacity — overall strength of the blend, in 0..1

struct BlendParams {
    resolution: u32,
    mode: u32,
    opacity: f32,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> params: BlendParams;
@group(0) @binding(1) var base_tex: texture_2d<f32>;
@group(0) @binding(2) var blend_tex: texture_2d<f32>;
@group(0) @binding(3) var mask_tex: texture_2d<f32>;
@group(0) @binding(4) var output: texture_storage_2d<rgba8unorm, write>;

fn combine(base: vec3<f32>, blend: vec3<f32>) -> vec3<f32> {
    switch params.mode {
        case 0u: { return blend; }                                     // Copy
        case 1u: { return base + blend; }                               // Add
        case 2u: { return base - blend; }                               // Subtract
        case 3u: { return base * blend; }                               // Multiply
        case 4u: { return 1.0 - (1.0 - base) * (1.0 - blend); }         // Screen
        case 5u: {                                                      // Overlay
            return select(
                1.0 - 2.0 * (1.0 - base) * (1.0 - blend),
                2.0 * base * blend,
                base < vec3<f32>(0.5),
            );
        }
        case 6u: { return max(base, blend); }                           // Max
        case 7u: { return min(base, blend); }                           // Min
        default: { return abs(base - blend); }                          // Difference
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let base = textureLoad(base_tex, coords, 0);
    let blend = textureLoad(blend_tex, coords, 0);
    let mask = textureLoad(mask_tex, coords, 0).r;

    let combined = clamp(combine(base.rgb, blend.rgb), vec3<f32>(0.0), vec3<f32>(1.0));
    let amount = clamp(mask * params.opacity, 0.0, 1.0);
    let result = mix(base.rgb, combined, amount);

    textureStore(output, coords, vec4<f32>(result, base.a));
}
