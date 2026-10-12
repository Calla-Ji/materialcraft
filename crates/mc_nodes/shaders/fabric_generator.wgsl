// SPDX-License-Identifier: MIT OR Apache-2.0
// fabric_generator.wgsl — Generators / Patterns / Fabric Generator
//
// Inputs:  none.
// Output:  `output` — a height-like value tracing a woven or knit fabric's
//          thread structure, 0 in the valleys between threads up to 1 on a
//          thread's own ridge. Chain into Height to Normal the same way
//          Pattern Generator's output is meant to be used.
// Params:
//   weave_type         — 0 Plain Weave, 1 Twill, 2 Rib Knit, 3 Stockinette
//                         (Jersey Knit)
//   rows, columns       — thread/stitch density
//   thread_width        — how much of each thread's own cell it fills, 0..1
//   thread_roundness    — 0 a flat ribbon, 1 a round cross-section
//   thread_group_size   — Plain Weave only: threads per group acting as one
//                         woven unit (1 = plain weave, 2+ = basket weave)
//   twill_count         — Twill only: the diagonal stripe's repeat period
//   rib_width           — Rib Knit only: columns per rib stripe
//   size_jitter,
//   luminance_jitter,
//   seed                — per-thread/stitch randomness, same convention as
//                         Pattern Generator
//
// Seamless tiling needs rows/columns to divide evenly by whatever period
// each weave type's own over/under or stagger pattern repeats on — same
// constraint, and same reason, as Pattern Generator's "rows must be even"
// note (see `mc_nodes::tile_generator`): Plain Weave/Basket Weave need rows
// AND columns even; Twill needs both a multiple of `twill_count`; Rib Knit
// needs columns a multiple of `rib_width * 2`; Stockinette needs rows even
// (it reuses Pattern Generator's own running-bond stagger). The defaults
// below satisfy every one of these at once.

struct FabricGeneratorParams {
    resolution: u32,
    rows: u32,
    columns: u32,
    seed: u32,
    weave_type: u32,
    thread_width: f32,
    thread_roundness: f32,
    thread_group_size: u32,
    twill_count: u32,
    rib_width: u32,
    size_jitter: f32,
    luminance_jitter: f32,
}

@group(0) @binding(0) var<uniform> params: FabricGeneratorParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

const WEAVE_PLAIN: u32 = 0u;
const WEAVE_TWILL: u32 = 1u;
const WEAVE_RIB_KNIT: u32 = 2u;
const WEAVE_STOCKINETTE: u32 = 3u;

// A single thread's cross-section: `t` is the signed distance from the
// thread's centerline, `half_width` the thread's own half-width, both in
// cell-fraction space. 0 outside the thread, rising to 1 at its center —
// `roundness` blends from a flat-topped ribbon (0) to a circular
// cross-section (1, the classic embroidery-render look).
fn thread_profile(t: f32, half_width: f32, roundness: f32) -> f32 {
    if half_width <= 0.0001 {
        return 0.0;
    }
    let d = clamp(abs(t) / half_width, 0.0, 1.0);
    let flat = 1.0 - d;
    let round_profile = sqrt(max(1.0 - d * d, 0.0));
    return mix(flat, round_profile, roundness);
}

fn wrap_i(v: i32, n: i32) -> i32 {
    return ((v % n) + n) % n;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);
    let columns = max(params.columns, 1u);
    let rows = max(params.rows, 1u);

    let col = i32(floor(uv.x * f32(columns)));
    let row = i32(floor(uv.y * f32(rows)));
    let xp = fract(uv.x * f32(columns)) - 0.5;
    let yp = fract(uv.y * f32(rows)) - 0.5;

    // Per-crossing jitter: a touch of per-thread width/brightness variation
    // so the weave doesn't read as a mechanically perfect grid.
    let cell = vec2<i32>(col, row);
    let h1 = hash2(cell, params.seed);
    let h2 = hash2(cell, params.seed + 101u);
    let width_scale = 1.0 - params.size_jitter * h1;
    let luminance = 1.0 - params.luminance_jitter * h2;

    let half_w = clamp(params.thread_width, 0.0, 1.0) * 0.5 * width_scale;

    var value = 0.0;
    if params.weave_type == WEAVE_RIB_KNIT {
        let rib_width = max(i32(params.rib_width), 1);
        let raised = wrap_i(col, rib_width * 2) < rib_width;
        var h = thread_profile(xp, half_w, params.thread_roundness);
        // A gentle per-row undulation hints at a knit column's stacked
        // loops without simulating each individual one.
        h = h * (0.9 + 0.1 * cos(yp * 6.28318530718));
        value = select(h * 0.45, h, raised);
    } else if params.weave_type == WEAVE_STOCKINETTE {
        // Reuses Pattern Generator's own running-bond stagger (alternate
        // rows offset by half a column) so loops interlock the way real
        // stockinette stitches do, instead of lining up in a plain grid.
        var x = uv.x * f32(columns);
        if wrap_i(row, 2) != 0 {
            x = x + 0.5;
        }
        let loop_col = i32(floor(x));
        let lx = fract(x) - 0.5;
        let ly = yp;
        let loop_cell = vec2<i32>(wrap_i(loop_col, i32(columns)), row);
        let lh1 = hash2(loop_cell, params.seed);
        let lh2 = hash2(loop_cell, params.seed + 101u);
        let loop_width_scale = 1.0 - params.size_jitter * lh1;
        // A squashed rhombus — the classic simplified "V" stitch silhouette
        // stockinette knitting is known for.
        let d = abs(lx) * 2.2 + abs(ly) * 1.3;
        let edge = clamp(params.thread_width, 0.05, 1.0) * loop_width_scale;
        value = clamp(1.0 - d / edge, 0.0, 1.0) * (1.0 - params.luminance_jitter * lh2);
        textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
        return;
    } else {
        // Plain Weave and Twill both cross a vertical and a horizontal
        // thread at every cell; they only differ in which thread is "on
        // top" at a given crossing.
        let v_h = thread_profile(xp, half_w, params.thread_roundness);
        let h_h = thread_profile(yp, half_w, params.thread_roundness);

        var vertical_on_top: bool;
        if params.weave_type == WEAVE_TWILL {
            let n = max(i32(params.twill_count), 2);
            vertical_on_top = wrap_i(col - row, n) < n / 2;
        } else {
            let group = max(i32(params.thread_group_size), 1);
            vertical_on_top = wrap_i(col / group + row / group, 2) == 0;
        }

        var top_h = h_h;
        var bottom_h = v_h;
        if vertical_on_top {
            top_h = v_h;
            bottom_h = h_h;
        }
        // The "under" thread is visibly recessed relative to the "over"
        // thread at each crossing — the ridge pattern that makes this read
        // as woven rather than a flat grid.
        value = max(top_h, bottom_h * 0.55);
    }

    value = clamp(value * luminance, 0.0, 1.0);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
