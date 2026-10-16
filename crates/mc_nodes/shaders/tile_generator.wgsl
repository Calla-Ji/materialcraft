// SPDX-License-Identifier: MIT OR Apache-2.0
// tile_generator.wgsl — Generators / Patterns / Pattern Generator
//
// Inputs:  `shape_tex` — an optional custom tile shape (e.g. the Shape
//          node's output): when connected, this is sampled (rotation- and
//          size-jittered the same as the built-in shape) instead of the
//          built-in rounded rect, so `gap`/`bevel` have no effect — the
//          shape's own silhouette and softness are whatever was authored
//          upstream. `has_shape` says which case this dispatch is; if
//          nothing's connected, `mc_nodes::tile_generator::dispatch` still
//          binds *something* here (never actually read).
// Output:  `output` — a brick/grid pattern: a height-like value, bright on
//          each tile's raised face (softened at the edges by `bevel`), 0 in
//          the gap between tiles.
// Params:
//   rows, columns      — grid density
//   offset             — stagger alternate rows by this fraction of a
//                        column width (0 = plain grid, 0.5 = classic brick).
//                        NOTE: for the stagger to tile seamlessly top-to-
//                        bottom, `rows` must be even — an odd row count
//                        flips row parity across the vertical wrap, same
//                        constraint every brick generator has. Ignored
//                        entirely outside `distribution` = Both: staggering
//                        alternate rows only makes sense across a full
//                        grid, and would otherwise knock every other tile
//                        sideways out of a single centered column/row.
//   gap                — mortar gap width, as a fraction of a tile. Only
//                        affects the built-in shape (see `shape_tex` above).
//   bevel              — how much of the tile's edge fades from full
//                        brightness to the gap. Only affects the built-in
//                        shape.
//   size_jitter        — per-tile random shrink, 0..1
//   rotation_jitter     — per-tile random rotation, 0..1 (small angles —
//                        this is a per-tile *cosmetic* tilt, not a precise
//                        physically-fitted brick edge)
//   luminance_jitter   — per-tile random brightness variation, 0..1
//   seed               — changes the per-tile randomness
//   distribution       — which grid axes actually get a tile: 0 = Both (the
//                        standard full rows x columns grid), 1 = Horizontal
//                        Only ("turn this into vertical stripes": columns
//                        repeat as usual, rows don't exist at all — every
//                        pixel shows the full, uncropped tile height, so a
//                        deliberately tall connected Shape (e.g. a big
//                        Scale Y stretch) comes through at its own real
//                        height instead of being squeezed into a small
//                        centered window), 2 = Vertical Only (the exact
//                        mirror: rows repeat, every pixel shows the full
//                        uncropped tile width). Column indices are always
//                        wrapped for their per-tile hash regardless of this
//                        value, independent of it — that's what keeps the
//                        staggered brick seam matching edge to edge, not a
//                        user-facing choice. Rotation Jitter is skipped
//                        outside Both: it only has one consistent geometric
//                        meaning when both axes repeat at the same scale,
//                        and Horizontal/Vertical Only deliberately mix one
//                        repeating (scaled) axis with one unscaled, full-
//                        tile axis.
//   shape_size         — only used when `shape_tex` is connected: how large
//                        each stamped shape is, as a fraction of the WHOLE
//                        TILE (same units Shape's own Size param uses) —
//                        not a fraction of each cell, so it stays the same
//                        absolute size as Tile Count (rows/columns) changes
//                        instead of silently shrinking as the grid gets
//                        finer. Internally this is still a zoom into the
//                        connected Shape's own canvas, applied to the
//                        REPEATING axis only (both axes, in Both mode; no
//                        cross-cell blending — a value larger than the
//                        current cell pitch just clips hard at the cell
//                        edge rather than spilling into neighbors) — the
//                        uncropped axis in Horizontal/Vertical Only ignores
//                        it entirely, by design (see `distribution` above).
//                        Built-in shape uses `gap` for this instead.

struct TileGeneratorParams {
    resolution: u32,
    rows: u32,
    columns: u32,
    seed: u32,
    offset: f32,
    gap: f32,
    bevel: f32,
    size_jitter: f32,
    rotation_jitter: f32,
    luminance_jitter: f32,
    distribution: u32,
    has_shape: u32,
    shape_size: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
}

@group(0) @binding(0) var<uniform> params: TileGeneratorParams;
@group(0) @binding(1) var shape_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);

    let rows_f = f32(params.rows);
    let columns_f = f32(params.columns);
    let columns_i = i32(params.columns);
    let is_horizontal_only = params.distribution == 1u;
    let is_vertical_only = params.distribution == 2u;
    let is_both = !is_horizontal_only && !is_vertical_only;

    // Horizontal Only means "turn this into vertical stripes": columns is
    // the one repeating axis (density/zoom controlled by Tile Count/Shape
    // Size, below), and the row axis isn't cropped to some strip thickness
    // at all — it always shows the FULL, uncropped tile height, so whatever
    // the connected Shape's own vertical extent is (including a
    // deliberately tall Scale Y stretch) comes through unchanged, not
    // squeezed into a `shape_size`-tall window. Vertical Only is the exact
    // mirror (rows repeat, columns show the full width uncropped).
    // `row_idx`/`col_idx` still need *some* fixed value on the restricted
    // axis purely for per-tile hashing (Size/Rotation/Luminance Jitter) —
    // which one doesn't matter since there's only ever one visible
    // row/column there.
    var row_idx: i32;
    var local_y: f32;
    if is_horizontal_only {
        local_y = uv.y - 0.5;
        row_idx = 0;
    } else {
        row_idx = i32(floor(uv.y * rows_f));
        local_y = fract(uv.y * rows_f) - 0.5;
    }

    // Stagger only has a coherent meaning across a full grid — applying it
    // to alternating rows of a single centered column would knock every
    // other tile sideways out of that column, so it's skipped entirely
    // outside Both (regardless of the Offset value).
    var x = uv.x * columns_f;
    if is_both && (row_idx % 2) != 0 {
        x = x + params.offset;
    }

    var col_idx: i32;
    var local_x: f32;
    if is_vertical_only {
        local_x = uv.x - 0.5;
        col_idx = 0;
    } else {
        col_idx = wrap_coord(i32(floor(x)), columns_i);
        local_x = fract(x) - 0.5;
    }

    let cell = vec2<i32>(col_idx, row_idx);

    let h = hash2(cell, params.seed);
    let h2 = hash2(cell, params.seed + 101u);
    let h3 = hash2(cell, params.seed + 211u);

    let size_scale = 1.0 - params.size_jitter * h;
    // Rotation only has one consistent geometric meaning across a uniform
    // grid where both axes repeat at the same scale (Both). Horizontal/
    // Vertical Only mix one repeating (scaled) axis with one unscaled,
    // full-tile axis — rotating would mix two different units — so it's
    // skipped there (angle 0), the same "small cosmetic tilt, not a
    // precise fit" scope Rotation Jitter already documents for itself.
    let angle = select(0.0, (h2 - 0.5) * params.rotation_jitter * 0.6, is_both);
    let luminance = 1.0 - params.luminance_jitter * h3;

    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let rotated = vec2<f32>(
        local_x * cos_a - local_y * sin_a,
        local_x * sin_a + local_y * cos_a,
    );

    var value = 0.0;
    if params.has_shape != 0u {
        let size = i32(params.resolution);
        // `shape_size` is tile-absolute (see the param doc comment
        // above) — scale by `columns_f` (== `rows_f` here, since
        // `effective_grid` on the Rust side always forces a square
        // grid once a Shape is connected) to turn that absolute target
        // into the right per-cell zoom for the current Tile Count. Only
        // the repeating axis is scaled by it; Horizontal/Vertical
        // Only's other axis samples the Shape's own canvas 1:1,
        // uncropped, over the whole tile.
        let effective_scale =
            max(size_scale * max(params.shape_size * columns_f, 0.0001), 0.0001);
        var sample_pos: vec2<f32>;
        if is_horizontal_only {
            let sx = clamp(rotated.x / effective_scale + 0.5, 0.0, 1.0);
            sample_pos = vec2<f32>(sx, uv.y);
        } else if is_vertical_only {
            let sy = clamp(rotated.y / effective_scale + 0.5, 0.0, 1.0);
            sample_pos = vec2<f32>(uv.x, sy);
        } else {
            sample_pos = clamp(
                rotated / effective_scale + vec2<f32>(0.5, 0.5),
                vec2<f32>(0.0, 0.0),
                vec2<f32>(1.0, 1.0),
            );
        }
        let coord = vec2<i32>(
            clamp(i32(sample_pos.x * f32(size)), 0, size - 1),
            clamp(i32(sample_pos.y * f32(size)), 0, size - 1),
        );
        value = textureLoad(shape_tex, coord, 0).r * luminance;
    } else {
        let half_size = vec2<f32>(0.5 - params.gap) * size_scale;
        let d = abs(rotated) - half_size;
        let outside = length(max(d, vec2<f32>(0.0)));
        let inside = min(max(d.x, d.y), 0.0);
        let sdf = outside + inside;

        if sdf < 0.0 {
            let bevel_amount = clamp(-sdf / max(params.bevel, 0.0001), 0.0, 1.0);
            value = luminance * mix(0.5, 1.0, bevel_amount);
        }
    }
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
