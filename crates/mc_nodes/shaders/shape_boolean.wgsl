// SPDX-License-Identifier: MIT OR Apache-2.0
// shape_boolean.wgsl — Converters / Shape Boolean
// (docs/hardsurface_nodes/HARDSURFACE_NODES.md §3.3)
//
// Inputs:  `a_tex`, `b_tex` — two Sdf textures (R channel, tile units,
//          negative inside).
// Output:  `sdf` — the combined field (raw; the `Exact` redistance pass, if
//          any, happens on the CPU after this dispatch — see
//          `mc_nodes::shape_boolean::dispatch`'s own doc comment).
//          `a_only`/`b_only`/`overlap` — binary (0/1) masks from the
//          *inputs'* own sign, independent of `op`/`mode`/`exact`.
//          `blend` — binary mask of pixels the chosen blend mode actually
//          changed vs. the hard (`min`/`max`) result, for welds/coloring —
//          the Dete Node Pack's "blend with intersection mask" for free.
// Params:
//   op          — 0 union, 1 subtract (a minus b), 2 intersect
//   mode        — 0 hard, 1 smooth, 2 round, 3 chamfer, 4 groove (groove
//                 ignores `op` entirely, same as the reference)
//   radius      — blend size (smooth/round/chamfer) or groove width, tile
//                 units
//   groove_depth — groove mode only: how far the groove cuts into `a`
//
// `_union`/`shape_boolean` ported from the public formulas in
// docs/hardsurface_nodes/HARDSURFACE_NODES.md §3.3 (iq's smooth minimum;
// hg_sdf's fOpUnionRound/fOpUnionChamfer/fOpGroove, MIT-licensed, see
// THIRD_PARTY.md) — reimplemented from the formulas, not copied shader
// code.

struct ShapeBooleanParams {
    resolution: u32,
    op: u32,
    mode: u32,
    radius: f32,
    groove_depth: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: ShapeBooleanParams;
@group(0) @binding(1) var a_tex: texture_2d<f32>;
@group(0) @binding(2) var b_tex: texture_2d<f32>;
@group(0) @binding(3) var sdf_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var a_only_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var b_only_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var overlap_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var blend_out: texture_storage_2d<rgba16float, write>;

// iq smooth minimum / hg_sdf fOpUnionRound/fOpUnionChamfer — `mode` controls
// which; `hard` (and `radius <= 0`) is a plain `min`.
fn op_union(a: f32, b: f32, mode: u32, r: f32) -> f32 {
    if mode == 0u || r <= 0.0 {
        return min(a, b);
    }
    if mode == 1u {
        // smooth: k = 4r (iq's own convention)
        let k = 4.0 * r;
        let h = max(k - abs(a - b), 0.0) / k;
        return min(a, b) - h * h * k * 0.25;
    }
    if mode == 2u {
        // round: hg_sdf fOpUnionRound
        let ux = max(r - a, 0.0);
        let uy = max(r - b, 0.0);
        return max(r, min(a, b)) - length(vec2<f32>(ux, uy));
    }
    // chamfer: hg_sdf fOpUnionChamfer
    return min(min(a, b), (a - r + b) * sqrt(0.5));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coords = vec2<i32>(id.xy);
    let a = textureLoad(a_tex, coords, 0).r;
    let b = textureLoad(b_tex, coords, 0).r;

    var out: f32;
    var hard: f32;
    if params.mode == 4u {
        // groove: cuts a groove of width `radius` along b's outline into a;
        // `op` doesn't apply here, same as the reference.
        out = max(a, min(a + params.groove_depth, params.radius - abs(b)));
        hard = a;
    } else if params.op == 0u {
        out = op_union(a, b, params.mode, params.radius);
        hard = min(a, b);
    } else if params.op == 2u {
        out = -op_union(-a, -b, params.mode, params.radius);
        hard = max(a, b);
    } else {
        // subtract: a minus b
        out = -op_union(-a, b, params.mode, params.radius);
        hard = max(a, -b);
    }

    let ia = a < 0.0;
    let ib = b < 0.0;
    let a_only = f32(ia && !ib);
    let b_only = f32(ib && !ia);
    let overlap = f32(ia && ib);
    let blend = f32(abs(out - hard) > 1e-7);

    textureStore(sdf_out, coords, vec4<f32>(out, out, out, 1.0));
    textureStore(a_only_out, coords, vec4<f32>(a_only, a_only, a_only, 1.0));
    textureStore(b_only_out, coords, vec4<f32>(b_only, b_only, b_only, 1.0));
    textureStore(overlap_out, coords, vec4<f32>(overlap, overlap, overlap, 1.0));
    textureStore(blend_out, coords, vec4<f32>(blend, blend, blend, 1.0));
}
