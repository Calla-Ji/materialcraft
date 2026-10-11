// SPDX-License-Identifier: MIT OR Apache-2.0
// dither_pattern.wgsl — Generators / Patterns / Dither Pattern
//
// Output:  `output` — the raw ordered-dither threshold mask itself (Gray),
//          for reuse in Blend, Threshold, and similar nodes.
// Params:  pattern — Bayer 2x2/4x4/8x8 (procedural, see `bayer_threshold` in
//          common.wgsl), Blue Noise (sampled from the bundled CC0 mask, see
//          ASSETS.md), or Dot (a fixed-radius circle per cell).
//          cells_x/cells_y — how many pattern tiles span the texture on
//          each axis; each rounded up to a multiple of the pattern's own
//          tile size (2/4/8/64/1) on the Rust side so it always divides
//          evenly and wraps seamlessly.
//          rotation — an integer (a, b) pair encoding a classic rational
//          halftone screen angle (see `dither_pattern.rs`'s doc comment on
//          `ROTATION_NAMES` for why this has to be a small integer pair
//          rather than a free angle to keep the pattern tileable). No
//          effect on Blue Noise, which has no grid of its own to rotate.
//          dot_size — fraction of a cell's radius the Dot pattern's circle
//          fills. Unused by every other pattern.
//          seed — offsets the Blue Noise sampling position; no effect on
//          any other pattern.

struct DitherPatternParams {
    resolution: u32,
    cells_x: i32,
    cells_y: i32,
    pattern: u32,
    rotation_a: i32,
    rotation_b: i32,
    dot_size: f32,
    seed: u32,
}

@group(0) @binding(0) var<uniform> params: DitherPatternParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var blue_noise: texture_2d<f32>;

const PATTERN_BAYER_2X2: u32 = 0u;
const PATTERN_BAYER_4X4: u32 = 1u;
const PATTERN_BAYER_8X8: u32 = 2u;
const PATTERN_BLUE_NOISE: u32 = 3u;
const PATTERN_DOT: u32 = 4u;
const BLUE_NOISE_SIZE: i32 = 64;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);

    var value: f32;
    if params.pattern == PATTERN_BLUE_NOISE {
        let raw_cell = vec2<i32>(floor(uv * vec2<f32>(f32(params.cells_x), f32(params.cells_y))));
        // Offsets the sample position by a seed-derived amount (not a hash
        // of the cell itself, since the mask is already a fixed, pre-
        // generated pattern — reseeding just slides which part of it lands
        // where) before wrapping into the mask's native 64x64 size.
        let offset = vec2<i32>(i32(params.seed) * 17, i32(params.seed) * 31);
        let texel = vec2<i32>(
            wrap_coord(raw_cell.x + offset.x, BLUE_NOISE_SIZE),
            wrap_coord(raw_cell.y + offset.y, BLUE_NOISE_SIZE),
        );
        value = textureLoad(blue_noise, texel, 0).r;
    } else {
        // A continuous (pre-floor) cell-space position, so Dot's circle can
        // use the fractional part within a cell — rotated by the integer
        // screen-angle matrix `[[a,-b],[b,a]]`, which (see the Rust-side
        // doc comment) stays exactly periodic in `uv` for any integer a, b
        // as long as `cells_x`/`cells_y` are already multiples of the
        // pattern's tile size.
        let p = uv * vec2<f32>(f32(params.cells_x), f32(params.cells_y));
        let a = f32(params.rotation_a);
        let b = f32(params.rotation_b);
        let p_rotated = vec2<f32>(a * p.x - b * p.y, b * p.x + a * p.y);
        let cell_i = vec2<i32>(floor(p_rotated));

        if params.pattern == PATTERN_DOT {
            let local = fract(p_rotated) - 0.5;
            let dist = length(local);
            let edge = clamp(params.dot_size, 0.0, 1.0) * 0.5;
            value = 1.0 - smoothstep(edge - 0.03, edge + 0.03, dist);
        } else {
            var size: u32 = 2u;
            if params.pattern == PATTERN_BAYER_4X4 {
                size = 4u;
            } else if params.pattern == PATTERN_BAYER_8X8 {
                size = 8u;
            }
            let wrapped = vec2<u32>(
                u32(wrap_coord(cell_i.x, i32(size))),
                u32(wrap_coord(cell_i.y, i32(size))),
            );
            value = bayer_threshold(wrapped, size);
        }
    }

    textureStore(output, vec2<i32>(id.xy), vec4<f32>(vec3<f32>(value), 1.0));
}
