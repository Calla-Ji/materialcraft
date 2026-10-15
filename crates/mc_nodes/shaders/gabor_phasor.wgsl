// SPDX-License-Identifier: MIT OR Apache-2.0
// gabor_phasor.wgsl — Generators / Noise / Gabor Noise, `phasor` output pass.
//
// Input:  `phase_in` — Gabor Noise's own `phase` output (0..1, wrapped),
//         read back as a plain sampled texture rather than another storage
//         binding, since the main `gabor_noise.wgsl` pass already uses all
//         4 storage-texture bindings the default WebGPU limit allows.
// Output: `phasor_out` — `0.5 + 0.5 sin(2 pi (phase - 0.5))`, a one-to-one
//         pointwise function of `phase` alone (see `GABOR_STREAKS_SPEC.md`'s
//         Outputs table), so no need to redo the heavy neighbor-cell sum.

const PI: f32 = 3.14159265358979323846;

@group(0) @binding(0) var phase_in: texture_2d<f32>;
@group(0) @binding(1) var phasor_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(phase_in);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let phase01 = textureLoad(phase_in, coord, 0).r;
    let phase_rad = (phase01 - 0.5) * 2.0 * PI;
    let phasor = 0.5 + 0.5 * sin(phase_rad);
    textureStore(phasor_out, coord, vec4<f32>(phasor, phasor, phasor, 1.0));
}
