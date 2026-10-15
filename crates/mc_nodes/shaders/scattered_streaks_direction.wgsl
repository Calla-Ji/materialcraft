// SPDX-License-Identifier: MIT OR Apache-2.0
// scattered_streaks_direction.wgsl — Scattered Streaks' `direction` output.
//
// A second, separate pass redoing the same per-pixel gather as
// `scattered_streaks.wgsl` (see that file's own doc comment for why it's
// split this way: 5 outputs total, but a single compute shader stage is
// capped at 4 storage-texture bindings). Unlike Gabor Noise's `phasor`
// split, `direction` isn't a cheap pointwise function of another output —
// it needs to know which streak actually won each pixel, so this redoes
// the full neighbor-cell search rather than reading back a result.
//
// Output: `direction` — a doubled-angle RG encoding of the winning
// streak's own angle, `(0.5 + 0.5 cos 2θ, 0.5 + 0.5 sin 2θ)` (same
// encoding as Stitch), defaulting to flat `(0.5, 0.5)` wherever no streak
// covers a pixel — NOT multiplied by `mask` the way `streak_id`/`along`
// are, matching `noise_reference.py`'s own `scattered_streaks` exactly
// (its `dirr`/`dirg` arrays start at 0.5 and are only overwritten at
// winning pixels, never zeroed afterward).

const PI: f32 = 3.14159265358979323846;

struct StreakRasterParams {
    resolution: u32,
    cells: u32,
    per_cell: u32,
    seed: u32,
    length: f32,
    length_random: f32,
    width: f32,
    width_random: f32,
    angle_rad: f32,
    angle_spread: f32,
    bend: f32,
    taper: f32,
    brightness_random: f32,
    profile: u32,
    blend: u32,
    search_radius: u32,
}

@group(0) @binding(0) var<uniform> params: StreakRasterParams;
@group(0) @binding(1) var angle_in: texture_2d<f32>;
@group(0) @binding(2) var direction_out: texture_storage_2d<rgba16float, write>;

fn wrap_cell(x: i32, n: i32) -> i32 {
    return ((x % n) + n) % n;
}

struct Gather {
    best_val: f32,
    best_angle: f32,
}

fn gather(x: f32, y: f32) -> Gather {
    var result: Gather;
    result.best_val = 0.0;
    result.best_angle = 0.0;

    let n = i32(params.cells);
    let nf = f32(params.cells);
    let ci = i32(floor(x * nf));
    let cj = i32(floor(y * nf));
    let r = i32(params.search_radius);
    let map_dims = textureDimensions(angle_in);

    for (var dj: i32 = -r; dj <= r; dj++) {
        for (var di: i32 = -r; di <= r; di++) {
            let i_cell = ci + di;
            let j_cell = cj + dj;
            let cid = u32(wrap_cell(j_cell, n)) * params.cells + u32(wrap_cell(i_cell, n));

            for (var k: u32 = 0u; k < params.per_cell; k++) {
                let sid = cid * params.per_cell + k;
                let h0 = hash01(sid, 0u, params.seed);
                let h1 = hash01(sid, 1u, params.seed);
                let h2 = hash01(sid, 2u, params.seed);
                let h3 = hash01(sid, 3u, params.seed);
                let h4 = hash01(sid, 4u, params.seed);
                let h5 = hash01(sid, 5u, params.seed);
                let h6 = hash01(sid, 6u, params.seed);

                let cx = (f32(i_cell) + h0) / nf;
                let cy = (f32(j_cell) + h1) / nf;
                var angle = params.angle_rad + params.angle_spread * PI * (2.0 * h2 - 1.0);

                let axi = i32(floor(fract(cx) * f32(map_dims.x))) % i32(map_dims.x);
                let ayi = i32(floor(fract(cy) * f32(map_dims.y))) % i32(map_dims.y);
                angle += textureLoad(angle_in, vec2<i32>(axi, ayi), 0).r;

                let length = params.length * (1.0 - params.length_random * h3);
                let width = params.width * (1.0 - params.width_random * h4);
                let half_len = length * 0.5;
                let half_w = width * 0.5;
                if (half_len <= 0.0 || half_w <= 0.0) {
                    continue;
                }
                let bend = params.bend * length * (2.0 * h5 - 1.0);
                let brightness = 1.0 - params.brightness_random * h6;

                let ux = cos(angle);
                let uy = sin(angle);
                let vx = -uy;
                let vy = ux;

                let px = x - cx;
                let py = y - cy;
                let a = px * ux + py * uy;
                let two_a_over_l = a / half_len;
                if (abs(two_a_over_l) > 1.0) {
                    continue;
                }
                let q = clamp(two_a_over_l, -1.0, 1.0);
                let b = px * vx + py * vy - bend * (1.0 - q * q);
                let t = abs(b) / half_w;

                var across = 0.0;
                if (params.profile == 0u) {
                    across = sqrt(clamp(1.0 - t * t, 0.0, 1.0));
                } else if (params.profile == 1u) {
                    if (t < 1.0) {
                        across = 0.5 + 0.5 * cos(PI * min(t, 1.0));
                    }
                } else {
                    across = clamp(0.5 - (abs(b) - half_w) * f32(params.resolution), 0.0, 1.0);
                }

                var fade = 1.0;
                if (params.taper > 0.0) {
                    fade = smoothstep(0.0, params.taper, 1.0 - abs(two_a_over_l));
                }
                let val = brightness * across * fade;
                if (val <= 0.0) {
                    continue;
                }

                if (val > result.best_val) {
                    result.best_val = val;
                    result.best_angle = angle;
                }
            }
        }
    }

    return result;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let x = (f32(id.x) + 0.5) / f32(params.resolution);
    let y = (f32(id.y) + 0.5) / f32(params.resolution);

    let g = gather(x, y);

    var dir_r = 0.5;
    var dir_g = 0.5;
    if (g.best_val > 0.0) {
        dir_r = 0.5 + 0.5 * cos(2.0 * g.best_angle);
        dir_g = 0.5 + 0.5 * sin(2.0 * g.best_angle);
    }

    textureStore(direction_out, vec2<i32>(id.xy), vec4<f32>(dir_r, dir_g, 0.0, 1.0));
}
