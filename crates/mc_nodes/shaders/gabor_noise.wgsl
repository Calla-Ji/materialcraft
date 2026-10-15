// SPDX-License-Identifier: MIT OR Apache-2.0
// gabor_noise.wgsl — Generators / Noise / Gabor Noise
//
// Inputs:  none (this is a generator node).
// Outputs: `value`, `phase`, `intensity` (broadcast to RGB, alpha = 1) and
//          `raw` (unit-variance float, unclamped — meant for math nodes
//          downstream, not direct display). `phasor` is NOT written here —
//          it's a pointwise function of `phase` alone, computed by the
//          separate `gabor_phasor.wgsl` pass instead, since a single
//          compute shader stage is capped at 4 storage texture bindings
//          (`wgpu::Limits::default()`, the WebGPU baseline) and this node
//          has 5 outputs total.
// Params (via the `params` uniform):
//   cells      — grid cells per tile; kernel radius r = 1/cells
//   frequency  — stripes (cycles) per tile along the angle
//   angle_rad  — direction the values change along, in radians
//   spread     — 0 = one direction, 1 = every direction (isotropic)
//   impulses   — kernels per cell
//   seed       — changes the random pattern
//   sigma_v/sigma_env — normalization std-devs, computed on the CPU (see
//     `gabor_noise::gabor_sigma`, ported from the owner's own reference
//     package's `noise_reference.py`) and passed in as a uniform, same as
//     that package's own GPU notes recommend.
//
// Every pixel sums the impulses of its own Gabor cell and its 8 neighbors
// (the cell size equals the kernel radius, so nothing farther away can
// reach it) — ported exactly from `noise_reference.py`'s `gabor_noise`.

const PI: f32 = 3.14159265358979323846;

struct GaborRasterParams {
    resolution: u32,
    cells: u32,
    impulses: u32,
    seed: u32,
    frequency: f32,
    angle_rad: f32,
    spread: f32,
    sigma_v: f32,
    sigma_env: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: GaborRasterParams;
@group(0) @binding(1) var value_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var phase_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var intensity_out: texture_storage_2d<rgba16float, write>;
// `raw` needs more than `rgba16float`'s ~10-bit mantissa: at its typical
// magnitude (unit variance, reaching ~3-4) that's only ~2e-3 resolution,
// too coarse for the reference package's own 1e-3 tolerance on this output
// (confirmed by measuring the actual quantization step, not assumed).
@group(0) @binding(4) var raw_out: texture_storage_2d<rgba32float, write>;

// Gaussian envelope, s = |d| / r, shifted/rescaled to be exactly 0 at s = 1
// (no seam at a kernel's own edge). Ported from `noise_reference.py`'s
// `envelope`.
fn envelope(s: f32) -> f32 {
    if s < 1.0 {
        let e_pi = exp(-PI);
        return (exp(-PI * s * s) - e_pi) / (1.0 - e_pi);
    }
    return 0.0;
}

fn wrap_cell(x: i32, n: i32) -> i32 {
    return ((x % n) + n) % n;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }

    let n = i32(params.cells);
    let nf = f32(params.cells);
    let r = 1.0 / nf;

    let x = (f32(id.x) + 0.5) / f32(params.resolution);
    let y = (f32(id.y) + 0.5) / f32(params.resolution);
    let ci = i32(floor(x * nf));
    let cj = i32(floor(y * nf));

    var re = 0.0;
    var im = 0.0;

    for (var dj: i32 = -1; dj <= 1; dj++) {
        for (var di: i32 = -1; di <= 1; di++) {
            let i_cell = ci + di;
            let j_cell = cj + dj;
            let cid = u32(wrap_cell(j_cell, n)) * params.cells + u32(wrap_cell(i_cell, n));
            for (var k: u32 = 0u; k < params.impulses; k++) {
                let base = k * 8u;
                let hx = hash01(cid, base + 0u, params.seed);
                let hy = hash01(cid, base + 1u, params.seed);
                let hw = hash01(cid, base + 2u, params.seed);
                let ho = hash01(cid, base + 3u, params.seed);

                let px = (f32(i_cell) + hx) / nf;
                let py = (f32(j_cell) + hy) / nf;
                let dx = x - px;
                let dy = y - py;

                let w = select(-1.0, 1.0, hw < 0.5);
                let om = params.angle_rad + params.spread * PI * (2.0 * ho - 1.0);
                let env = envelope(sqrt(dx * dx + dy * dy) / r);
                let ph = 2.0 * PI * params.frequency * (dx * cos(om) + dy * sin(om));

                re += w * env * cos(ph);
                im += w * env * sin(ph);
            }
        }
    }

    let value = clamp(0.5 + 0.5 * im / (3.0 * params.sigma_v), 0.0, 1.0);
    let phase = atan2(im, re);
    let phase01 = phase / (2.0 * PI) + 0.5;
    let intensity = clamp(sqrt(re * re + im * im) / (3.0 * params.sigma_env), 0.0, 1.0);
    let raw = im / params.sigma_v;

    let coord = vec2<i32>(id.xy);
    textureStore(value_out, coord, vec4<f32>(value, value, value, 1.0));
    textureStore(phase_out, coord, vec4<f32>(phase01, phase01, phase01, 1.0));
    textureStore(intensity_out, coord, vec4<f32>(intensity, intensity, intensity, 1.0));
    textureStore(raw_out, coord, vec4<f32>(raw, raw, raw, 1.0));
}
