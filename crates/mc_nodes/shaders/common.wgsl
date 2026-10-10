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

// Wraps an integer pixel coordinate into [0, size) — repeat addressing for
// `textureLoad`, which (unlike `textureSample`) has no built-in wrap mode.
fn wrap_coord(coord: i32, size: i32) -> i32 {
    return ((coord % size) + size) % size;
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
