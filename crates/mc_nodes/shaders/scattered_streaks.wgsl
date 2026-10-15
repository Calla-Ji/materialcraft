// SPDX-License-Identifier: MIT OR Apache-2.0
// scattered_streaks.wgsl — Generators / Noise / Scattered Streaks.
//
// Input:   `angle_in` — an optional per-pixel angle field (radians), added
//          to every streak's own angle at its centre (nearest pixel,
//          wrapped). Defaults to a flat 0.0 (no extra rotation) when
//          nothing is connected upstream.
// Outputs: `height`, `mask`, `streak_id`, `along` (broadcast to RGB,
//          alpha = 1). `direction` is NOT written here — see
//          `scattered_streaks_direction.wgsl`'s own doc comment for why
//          (same 4-storage-texture-binding limit reason as Gabor Noise's
//          own `phasor` split).
// Params (via the `params` uniform): see `GABOR_STREAKS_SPEC.md`'s own
//   Scattered Streaks parameter table for each one's meaning.
//
// Rather than the spec's own suggested GPU approach (instanced quads +
// blending), this gathers: for each pixel, every streak that could reach it
// lives in a nearby cell of the same `cells x cells` grid `streak_list`
// itself partitions streaks into (`mc_nodes::scattered_streaks::
// search_radius_cells` sizes the search window on the CPU from the worst-
// case streak reach for the current params), so a pixel only needs to
// recompute (via the same `hash01`-seeded formulas as the CPU reference)
// the handful of streaks in its own neighborhood, not the whole list.

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
@group(0) @binding(2) var height_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var mask_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var streak_id_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var along_out: texture_storage_2d<rgba16float, write>;

fn wrap_cell(x: i32, n: i32) -> i32 {
    return ((x % n) + n) % n;
}

// The single per-pixel "winner" state this gather tracks: the streak with
// the largest individual coverage `val` at this pixel wins `streak_id`/
// `along`/`direction` — this matches the Python reference's own "max"
// blend mode exactly (there, `height` IS the true per-pixel max, so
// whichever streak set it is unambiguously the single largest contributor
// regardless of processing order). For "add" blend, the Python reference
// instead picks per-pixel winners via a sequential, streak-processing-order-
// dependent rule (each streak claims any pixel where its own value is at
// least half its own peak, overwriting any earlier claim) — not meaningful
// to replicate exactly on a GPU with no inherent per-pixel processing
// order, so "largest individual contributor wins" is used for both modes;
// only `height` itself (a true order-independent sum, clamped to 1) is
// exact for "add". `GABOR_STREAKS_SPEC.md`'s own tolerance for this step
// already allows `mask`/`streak_id` to differ on up to 0.1% of pixels for
// exactly this kind of winner-identity edge case.
struct Gather {
    sum_val: f32,
    best_val: f32,
    best_rand: f32,
    best_along: f32,
    best_angle: f32,
}

fn gather(x: f32, y: f32) -> Gather {
    var result: Gather;
    result.sum_val = 0.0;
    result.best_val = 0.0;
    result.best_rand = 0.0;
    result.best_along = 0.0;
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
                let h7 = hash01(sid, 7u, params.seed);

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

                result.sum_val += val;
                if (val > result.best_val) {
                    result.best_val = val;
                    result.best_rand = h7;
                    result.best_along = 0.5 + a / length;
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

    var height = g.best_val;
    if (params.blend == 1u) {
        height = min(g.sum_val, 1.0);
    }
    let mask = select(0.0, 1.0, height > 0.0);
    let streak_id = g.best_rand * mask;
    let along = g.best_along * mask;

    let coord = vec2<i32>(id.xy);
    textureStore(height_out, coord, vec4<f32>(height, height, height, 1.0));
    textureStore(mask_out, coord, vec4<f32>(mask, mask, mask, 1.0));
    textureStore(streak_id_out, coord, vec4<f32>(streak_id, streak_id, streak_id, 1.0));
    textureStore(along_out, coord, vec4<f32>(along, along, along, 1.0));
}
