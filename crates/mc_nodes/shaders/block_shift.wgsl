// SPDX-License-Identifier: MIT OR Apache-2.0
// block_shift.wgsl — Filters / Glitch / Block Shift
//
// Inputs:  `image_tex` — the texture to glitch (Gray or Color — both are
//          just RGBA on the GPU, see `mc_nodes::defs::effective_output_type`).
//          `mask_tex` — defaults to solid white (fully visible) when nothing
//          is connected; see `mc_nodes::block_shift::dispatch`.
// Output:  `output` — divided into a `columns` x `rows` block grid; a seeded
//          hash per block decides whether it shifts at all (`amount` is the
//          fraction that do) and, if so, by how much along `direction`
//          (up to `max_offset`, in UV units). Every shifted sample is
//          wrap-addressed, so the source content stays tileable even though
//          the shift pattern itself is intentionally blocky/discontinuous
//          (that discontinuity *is* the effect).
// Params:  columns/rows — the block grid. amount — fraction of blocks that
//          shift. max_offset — largest possible shift, in UV units.
//          direction — which axis/axes a shifting block moves along.
//          seed — reseeds which blocks shift and by how much.

struct BlockShiftParams {
    resolution: u32,
    columns: i32,
    rows: i32,
    direction: u32,
    amount: f32,
    max_offset: f32,
    seed: u32,
    tiling_mode: u32,
}

@group(0) @binding(0) var<uniform> params: BlockShiftParams;
@group(0) @binding(1) var image_tex: texture_2d<f32>;
@group(0) @binding(2) var mask_tex: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;

const DIRECTION_HORIZONTAL: u32 = 0u;
const DIRECTION_VERTICAL: u32 = 1u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coords = vec2<i32>(id.xy);
    let size = i32(params.resolution);
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);

    let block_cell = vec2<i32>(floor(uv * vec2<f32>(f32(params.columns), f32(params.rows))));
    let does_shift = hash2(block_cell, params.seed) < params.amount;

    var offset_uv = vec2<f32>(0.0, 0.0);
    if does_shift {
        let hx = hash2(block_cell, params.seed + 1009u) * 2.0 - 1.0;
        let hy = hash2(block_cell, params.seed + 2003u) * 2.0 - 1.0;
        if params.direction == DIRECTION_HORIZONTAL {
            offset_uv = vec2<f32>(hx * params.max_offset, 0.0);
        } else if params.direction == DIRECTION_VERTICAL {
            offset_uv = vec2<f32>(0.0, hy * params.max_offset);
        } else {
            offset_uv = vec2<f32>(hx, hy) * params.max_offset;
        }
    }

    let offset_px = vec2<i32>(round(offset_uv * f32(params.resolution)));
    let sample_coord = vec2<i32>(
        tile_coord(coords.x - offset_px.x, size, wraps_x(params.tiling_mode)),
        tile_coord(coords.y - offset_px.y, size, wraps_y(params.tiling_mode)),
    );
    let shifted = textureLoad(image_tex, sample_coord, 0);
    let original = textureLoad(image_tex, coords, 0);
    let mask = textureLoad(mask_tex, coords, 0).r;

    textureStore(output, coords, mix(original, shifted, mask));
}
