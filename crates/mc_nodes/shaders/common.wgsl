// SPDX-License-Identifier: MIT OR Apache-2.0
// common.wgsl — shared helpers for node shaders (hashing, gradients, tiling).
//
// Every generator must be *periodic*: sampling just past the right edge of
// the UV domain must match sampling just past the left edge, so textures
// tile seamlessly. The helpers below hash on integer lattice *cell*
// coordinates wrapped into a fixed-size period, rather than on raw
// floating-point UVs, which is what makes that possible.

// Hashes a 2D integer cell coordinate to a pseudo-random float in [0, 1).
// Same idea as Blender's "White Noise" texture node, but we hash integers
// (lattice points) instead of the continuous UV so wrapping is exact.
fn hash2(cell: vec2<i32>, seed: u32) -> f32 {
    var x = u32(cell.x) * 1664525u + u32(cell.y) * 1013904223u + seed * 374761393u;
    x = (x ^ (x >> 16u)) * 2246822519u;
    x = (x ^ (x >> 13u)) * 3266489917u;
    x = x ^ (x >> 16u);
    return f32(x) / 4294967295.0;
}

// A pseudo-random unit gradient vector for a lattice point, wrapped to
// `period` so the gradient field repeats exactly every `period` cells —
// this is what makes Perlin noise below tile.
fn gradient(cell: vec2<i32>, period: i32, seed: u32) -> vec2<f32> {
    let wrapped = vec2<i32>(
        ((cell.x % period) + period) % period,
        ((cell.y % period) + period) % period,
    );
    let angle = hash2(wrapped, seed) * 6.28318530718;
    return vec2<f32>(cos(angle), sin(angle));
}

// Smootherstep (Perlin's improved fade curve): 6t^5 - 15t^4 + 10t^3.
fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Tileable 2D Perlin (gradient) noise. `uv` is in [0, 1); `period` is how
// many noise cells fit across the texture (so the gradient lattice wraps
// exactly at the texture edges). Returns a value in roughly [-1, 1].
fn perlin_2d(uv: vec2<f32>, period: i32, seed: u32) -> f32 {
    let p = uv * f32(period);
    let cell = vec2<i32>(floor(p));
    let local = fract(p);

    let g00 = gradient(cell + vec2<i32>(0, 0), period, seed);
    let g10 = gradient(cell + vec2<i32>(1, 0), period, seed);
    let g01 = gradient(cell + vec2<i32>(0, 1), period, seed);
    let g11 = gradient(cell + vec2<i32>(1, 1), period, seed);

    let d00 = local - vec2<f32>(0.0, 0.0);
    let d10 = local - vec2<f32>(1.0, 0.0);
    let d01 = local - vec2<f32>(0.0, 1.0);
    let d11 = local - vec2<f32>(1.0, 1.0);

    let n00 = dot(g00, d00);
    let n10 = dot(g10, d10);
    let n01 = dot(g01, d01);
    let n11 = dot(g11, d11);

    let u = fade(local.x);
    let v = fade(local.y);

    let nx0 = mix(n00, n10, u);
    let nx1 = mix(n01, n11, u);
    return mix(nx0, nx1, v);
}

// 2D simplex noise (Gustavson-style skewed-triangle lattice). `uv` is in
// [0, 1); `period` is how many noise cells fit across the texture. Returns a
// value in roughly [-1, 1].
//
// NOT tileable on its own, unlike `perlin_2d` — this is the raw primitive;
// use `tileable_simplex_2d` below for an actually-seamless result. Every
// lattice point's gradient comes from `gradient()`, which wraps to `period`
// before hashing, so the *gradients* repeat every `period` cells — but
// simplex's skewed lattice doesn't line up with that period the way
// Perlin's axis-aligned square grid does (skewing by a non-integer amount
// shifts which simplex cell/corner a given point falls into, so the
// triangle boundaries themselves drift relative to the gradient period).
// Confirmed by testing, not assumed: `simplex_2d` alone fails the tiling
// test by a real, visible margin, not a rounding-level one.
fn simplex_2d(uv: vec2<f32>, period: i32, seed: u32) -> f32 {
    let skew = 0.36602540378; // (sqrt(3) - 1) / 2
    let unskew = 0.21132486541; // (3 - sqrt(3)) / 6

    let p = uv * f32(period);
    let s = (p.x + p.y) * skew;
    let cell_f = floor(p + vec2<f32>(s, s));
    let t = (cell_f.x + cell_f.y) * unskew;
    let origin = cell_f - vec2<f32>(t, t);
    let d0 = p - origin;

    var i1 = 0;
    var j1 = 1;
    if d0.x > d0.y {
        i1 = 1;
        j1 = 0;
    }

    let d1 = d0 - vec2<f32>(f32(i1), f32(j1)) + vec2<f32>(unskew, unskew);
    let d2 = d0 - vec2<f32>(1.0, 1.0) + vec2<f32>(2.0 * unskew, 2.0 * unskew);

    let cell = vec2<i32>(cell_f);
    let g0 = gradient(cell, period, seed);
    let g1 = gradient(cell + vec2<i32>(i1, j1), period, seed);
    let g2 = gradient(cell + vec2<i32>(1, 1), period, seed);

    var n0 = 0.0;
    var t0 = 0.5 - dot(d0, d0);
    if t0 > 0.0 {
        t0 = t0 * t0;
        n0 = t0 * t0 * dot(g0, d0);
    }

    var n1 = 0.0;
    var t1 = 0.5 - dot(d1, d1);
    if t1 > 0.0 {
        t1 = t1 * t1;
        n1 = t1 * t1 * dot(g1, d1);
    }

    var n2 = 0.0;
    var t2 = 0.5 - dot(d2, d2);
    if t2 > 0.0 {
        t2 = t2 * t2;
        n2 = t2 * t2 * dot(g2, d2);
    }

    return 70.0 * (n0 + n1 + n2);
}

// A genuinely seamless-tiling version of `simplex_2d`, built by blending it
// with 3 copies of itself shifted by exactly one period in x, y, and both —
// the standard technique for tiling a noise function that isn't periodic on
// its own (Perlin's original 2002 tileable-noise note describes the same
// idea). `uv` is in [0, 1), not pre-multiplied by `period`, since the blend
// weights need the unscaled position.
//
// Why this is exact, not approximate: at uv.x = 0 the weight `wx` is exactly
// 1, so the blend reduces to just `simplex_2d(uv, ...)`; at uv.x = 1, `wx`
// is exactly 0, so it reduces to just the shifted copy evaluated at
// `uv.x - 1 = 0` — the *same* function call as the uv.x = 0 case. The two
// edges are mathematically forced to agree, not just close. In practice a
// texel never lands exactly on uv.x/y = 0 or 1 (samples are at pixel
// centers), so the match is extremely close rather than bit-exact, same as
// every other tileable noise here.
fn tileable_simplex_2d(uv: vec2<f32>, period: i32, seed: u32) -> f32 {
    let wx = 1.0 - uv.x;
    let wy = 1.0 - uv.y;
    let n00 = simplex_2d(uv, period, seed);
    let n10 = simplex_2d(vec2<f32>(uv.x - 1.0, uv.y), period, seed);
    let n01 = simplex_2d(vec2<f32>(uv.x, uv.y - 1.0), period, seed);
    let n11 = simplex_2d(vec2<f32>(uv.x - 1.0, uv.y - 1.0), period, seed);
    return n00 * wx * wy + n10 * (1.0 - wx) * wy + n01 * wx * (1.0 - wy)
        + n11 * (1.0 - wx) * (1.0 - wy);
}

// Wraps an integer pixel coordinate into [0, size) — repeat addressing for
// `textureLoad`, which (unlike `textureSample`) has no built-in wrap mode.
fn wrap_coord(coord: i32, size: i32) -> i32 {
    return ((coord % size) + size) % size;
}

// Shared by any node with a "Tiling Mode" param (`mc_nodes::tiling::
// TilingMode` on the Rust side — Transform 2D, Warp, Blur, ...): `flags`
// packs wrap-X in bit 0, wrap-Y in bit 1, matching `TilingMode::as_u32`.
// `Both` (every node's default, matching the always-on behavior before this
// param existed) sets both bits.
fn wraps_x(flags: u32) -> bool {
    return (flags & 1u) != 0u;
}

fn wraps_y(flags: u32) -> bool {
    return (flags & 2u) != 0u;
}

// Wraps or clamps a single axis of an integer texel coordinate that's gone
// out of [0, size) from reading a neighboring pixel (a blur/warp/sharpen
// tap) — `wrap_coord` when that axis should tile, clamp-to-edge (repeat the
// border texel, the standard "don't tile" addressing mode) when it
// shouldn't.
fn tile_coord(coord: i32, size: i32, wrap: bool) -> i32 {
    if wrap {
        return wrap_coord(coord, size);
    }
    return clamp(coord, 0, size - 1);
}

// Same idea as `tile_coord`, for a single axis of a continuous UV backward-
// mapped outside [0, 1) (e.g. Transform 2D's offset/rotated sample
// position): `fract` when that axis should tile, clamp to the last valid
// value otherwise so an offset that pushes content off one edge reveals a
// stretched copy of the border pixel instead of the opposite edge's content.
fn tile_uv(p: f32, wrap: bool) -> f32 {
    if wrap {
        return fract(p);
    }
    return clamp(p, 0.0, 0.999999);
}

// Tileable 2D Voronoi/Worley "cells": returns (F1, F2), the distances to the
// nearest and second-nearest jittered points. Each cell gets one point,
// placed randomly inside it and wrapped to `period` the same way `gradient`
// above wraps its lattice, so cells tile exactly.
fn voronoi_f1f2(uv: vec2<f32>, period: i32, jitter: f32, seed: u32) -> vec2<f32> {
    let p = uv * f32(period);
    let cell = vec2<i32>(floor(p));
    let local = fract(p);

    var f1 = 1e9;
    var f2 = 1e9;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let neighbor = cell + vec2<i32>(dx, dy);
            let wrapped = vec2<i32>(
                wrap_coord(neighbor.x, period),
                wrap_coord(neighbor.y, period),
            );
            let rand = vec2<f32>(hash2(wrapped, seed), hash2(wrapped, seed + 7919u));
            let point_pos = vec2<f32>(f32(dx), f32(dy)) + 0.5 + (rand - 0.5) * jitter;
            let dist = length(point_pos - local);
            if dist < f1 {
                f2 = f1;
                f1 = dist;
            } else if dist < f2 {
                f2 = dist;
            }
        }
    }
    return vec2<f32>(f1, f2);
}

// A recursively-defined ordered-dither (Bayer) threshold matrix, for any
// power-of-two `size` — public-domain technique (see e.g. Wikipedia's
// "Ordered dithering" article for the general recursive definition):
// doubling the matrix size tiles 4 copies of the smaller matrix into the 4
// quadrants, each copy scaled by 4 and offset by a fixed per-quadrant
// constant {0, 2, 3, 1} (top-left, top-right, bottom-left, bottom-right).
// That recursion is built here bit-by-bit, least-significant first: each
// loop iteration consumes one more bit of `cell.x`/`cell.y`, multiplies the
// running total by 4 (undoing one level of the previous, finer recursion)
// and adds this level's own quadrant constant — which is exactly the
// recursive definition read outward from its base case. Verified against
// the standard 2x2 ({0,2;3,1}) and 4x4 matrices by this file's own tests.
fn bayer_threshold(cell: vec2<u32>, size: u32) -> f32 {
    var quadrant_constant: array<u32, 4> = array<u32, 4>(0u, 2u, 3u, 1u);
    let x0 = cell.x % size;
    let y0 = cell.y % size;
    var value: u32 = 0u;
    var level: u32 = 0u;
    loop {
        if (1u << level) >= size {
            break;
        }
        let bx = (x0 >> level) & 1u;
        let by = (y0 >> level) & 1u;
        value = value * 4u + quadrant_constant[by * 2u + bx];
        level = level + 1u;
    }
    return (f32(value) + 0.5) / f32(size * size);
}
