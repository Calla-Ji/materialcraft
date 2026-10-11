// mip_downsample.wgsl — one box-filter step of a channel texture's mip
// chain: reads mip L-1, writes mip L at half the resolution.
//
// Wrap-addressed (same as every node shader): these textures tile, so the
// downsampled result needs to tile too, or a material would visibly seam at
// a distance even though it's seamless up close.
//
// Normal maps need `is_normal` set: averaging 4 encoded normals directly
// (like any other channel) shrinks the resulting vector below unit length,
// which reads as a "flatter" surface at lower mips — decode, average,
// renormalize, re-encode instead.

struct DownsampleParams {
    dest_size: u32,
    is_normal: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: DownsampleParams;
@group(0) @binding(1) var source_tex: texture_2d<f32>;
@group(0) @binding(2) var dest_tex: texture_storage_2d<rgba8unorm, write>;

fn wrap(coord: i32, size: i32) -> i32 {
    return ((coord % size) + size) % size;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.dest_size || id.y >= params.dest_size) {
        return;
    }
    let src_size = i32(params.dest_size) * 2;
    let src_origin = vec2<i32>(id.xy) * 2;

    let c00 = textureLoad(
        source_tex,
        vec2<i32>(wrap(src_origin.x, src_size), wrap(src_origin.y, src_size)),
        0,
    );
    let c10 = textureLoad(
        source_tex,
        vec2<i32>(wrap(src_origin.x + 1, src_size), wrap(src_origin.y, src_size)),
        0,
    );
    let c01 = textureLoad(
        source_tex,
        vec2<i32>(wrap(src_origin.x, src_size), wrap(src_origin.y + 1, src_size)),
        0,
    );
    let c11 = textureLoad(
        source_tex,
        vec2<i32>(wrap(src_origin.x + 1, src_size), wrap(src_origin.y + 1, src_size)),
        0,
    );

    var result: vec4<f32>;
    if params.is_normal == 1u {
        let n = normalize(
            (c00.xyz * 2.0 - 1.0) + (c10.xyz * 2.0 - 1.0) +
            (c01.xyz * 2.0 - 1.0) + (c11.xyz * 2.0 - 1.0),
        );
        let a = (c00.a + c10.a + c01.a + c11.a) * 0.25;
        result = vec4<f32>(n * 0.5 + 0.5, a);
    } else {
        result = (c00 + c10 + c01 + c11) * 0.25;
    }

    textureStore(dest_tex, vec2<i32>(id.xy), result);
}
