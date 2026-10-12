// SPDX-License-Identifier: MIT OR Apache-2.0
// scanline_jitter.wgsl — Filters / Glitch / Scanline Jitter
//
// Inputs:  `image_tex` — the texture to glitch (Gray or Color — both are
//          just RGBA on the GPU, see `mc_nodes::defs::effective_output_type`).
//          `mask_tex` — defaults to solid white (fully visible) when nothing
//          is connected; see `mc_nodes::scanline_jitter::dispatch`.
// Output:  `output` — divided into `lines` bands along `orientation`; a
//          seeded hash per band decides whether it jitters at all (`amount`
//          is the fraction that do) and, if so, by how much perpendicular to
//          the band (up to `max_offset`, in UV units). Every jittered sample
//          is wrap-addressed, so the source content stays tileable even
//          though the jitter pattern itself is intentionally discontinuous
//          band-to-band (that discontinuity *is* the classic "bad video
//          signal" look).
// Params:  lines — how many bands span the texture. amount — fraction of
//          bands that jitter. max_offset — largest possible jitter, in UV
//          units. orientation — whether bands run horizontally or
//          vertically (and so which axis they jitter along). seed —
//          reseeds which bands jitter and by how much.

struct ScanlineJitterParams {
    resolution: u32,
    lines: i32,
    orientation: u32,
    amount: f32,
    max_offset: f32,
    seed: u32,
    tiling_mode: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: ScanlineJitterParams;
@group(0) @binding(1) var image_tex: texture_2d<f32>;
@group(0) @binding(2) var mask_tex: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba8unorm, write>;

const ORIENTATION_HORIZONTAL: u32 = 0u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coords = vec2<i32>(id.xy);
    let size = i32(params.resolution);
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);

    // Horizontal bands run along x, stacked along y, so the band index
    // comes from `uv.y`; vertical bands are the transpose of that.
    var band: i32;
    if params.orientation == ORIENTATION_HORIZONTAL {
        band = i32(floor(uv.y * f32(params.lines)));
    } else {
        band = i32(floor(uv.x * f32(params.lines)));
    }
    let band_cell = vec2<i32>(band, 0);

    var offset_px = 0i;
    if hash2(band_cell, params.seed) < params.amount {
        let h = hash2(band_cell, params.seed + 3001u) * 2.0 - 1.0;
        offset_px = i32(round(h * params.max_offset * f32(params.resolution)));
    }

    var sample_coord: vec2<i32>;
    if params.orientation == ORIENTATION_HORIZONTAL {
        sample_coord = vec2<i32>(
            tile_coord(coords.x - offset_px, size, wraps_x(params.tiling_mode)),
            coords.y,
        );
    } else {
        sample_coord = vec2<i32>(
            coords.x,
            tile_coord(coords.y - offset_px, size, wraps_y(params.tiling_mode)),
        );
    }

    let jittered = textureLoad(image_tex, sample_coord, 0);
    let original = textureLoad(image_tex, coords, 0);
    let mask = textureLoad(mask_tex, coords, 0).r;

    textureStore(output, coords, mix(original, jittered, mask));
}
