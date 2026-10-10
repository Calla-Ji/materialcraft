// grayscale_conversion.wgsl — Filters / Adjustments / Grayscale Conversion
//
// Inputs:  `input_tex` — a Color texture.
// Output:  `output` — Rec. 709 luminance, broadcast to RGB (alpha = 1), the
//          same "Gray stored as RGB" convention every Gray-output node uses
//          (CLAUDE.md §5: the one explicit, intentional way to turn a Color
//          into a Gray — implicit Color->Gray connections are blocked in the
//          UI, see `mc_nodes::defs::ports_compatible`).

struct GrayscaleConversionParams {
    resolution: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: GrayscaleConversionParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let value = textureLoad(input_tex, vec2<i32>(id.xy), 0);
    let luminance = dot(value.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(vec3<f32>(luminance), 1.0));
}
