// SPDX-License-Identifier: MIT OR Apache-2.0
// path_stamp.wgsl — Generators / Patterns / Path's Stamp distribution.
//
// Inputs:  `stamp_tex` — whatever's connected to the Path node's own
//          Stamp input (a Shape node today, a future Stitch node later),
//          sampled nearest-neighbor (no `textureSample`/sampler — compute
//          shaders can't use implicit-derivative sampling) at each
//          instance's own local coordinate.
//          `base_tex` — Preview Mask's own content before stamping (the
//          curve's outline if Show Curve is on, blank otherwise).
//          `instances` — one entry per placed copy (position + final
//          rotation, already including both "align to curve" and the
//          Stamp Rotation param — this shader doesn't know about either,
//          it just rotates by whatever angle it's given).
// Output:  `output` — `base_tex` with every instance's own stamp
//          composited on top via `max` (same "most prominent wins" rule
//          as every other strand/shape composite in this codebase).
// Params:  resolution, instance_count, stamp_size (tile units, both
//          axes), wrap (test each instance's 8 neighboring tile copies
//          too, for seamless tiling).

struct StampParams {
    resolution: u32,
    instance_count: u32,
    stamp_size: f32,
    wrap: u32,
}

struct StampInstance {
    pos: vec2<f32>,
    angle: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: StampParams;
@group(0) @binding(1) var<storage, read> instances: array<StampInstance>;
@group(0) @binding(2) var stamp_tex: texture_2d<f32>;
@group(0) @binding(3) var base_tex: texture_2d<f32>;
@group(0) @binding(4) var output: texture_storage_2d<rgba8unorm, write>;

const WRAP_OFFSETS = array<vec2<f32>, 9>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(-1.0, 0.0), vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, -1.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
);

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.resolution || gid.y >= params.resolution {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5, 0.5)) / f32(params.resolution);

    var best = textureLoad(base_tex, coord, 0).r;
    let half_size = params.stamp_size * 0.5;
    let offset_count = select(1u, 9u, params.wrap != 0u);
    let res_minus_one = i32(params.resolution) - 1;

    for (var i: u32 = 0u; i < params.instance_count; i = i + 1u) {
        let inst = instances[i];
        let c = cos(-inst.angle);
        let s = sin(-inst.angle);
        for (var k: u32 = 0u; k < offset_count; k = k + 1u) {
            let center = inst.pos + WRAP_OFFSETS[k];
            let d = uv - center;
            let local = vec2<f32>(d.x * c - d.y * s, d.x * s + d.y * c);
            if abs(local.x) <= half_size && abs(local.y) <= half_size {
                let stamp_uv = (local / params.stamp_size) + vec2<f32>(0.5, 0.5);
                let px = vec2<i32>(stamp_uv * f32(params.resolution));
                let clamped = clamp(px, vec2<i32>(0, 0), vec2<i32>(res_minus_one, res_minus_one));
                best = max(best, textureLoad(stamp_tex, clamped, 0).r);
            }
        }
    }

    textureStore(output, coord, vec4<f32>(best, best, best, 1.0));
}
