// SPDX-License-Identifier: MIT OR Apache-2.0
// bevel_profile.wgsl — Converters / Bevel Profile
// (docs/hardsurface_nodes/HARDSURFACE_NODES.md §3.4)
//
// Input:  `sdf_tex` — an Sdf texture (R channel, tile units, negative inside).
// Output: `height` — base + depth * p * coverage(s), the beveled height map.
//         `edge`   — AA coverage gated to the bevel band (0 <= u < 1), not
//                    binary.
//         `top`    — binary mask of the flat top (u >= 1).
//         `u`      — the bevel coordinate itself (0 at the edge, 1 on the
//                    flat top).
// Params:
//   preset — 0 chamfer, 1 round, 2 cove, 3 ogee, 4 step, 5 lip, 6 flat,
//            7 custom
//   inward — nonzero bevels inward (the normal case); zero flips the sdf's
//            sign first, bevelling outward (a moat/raised surround) instead.
//   width  — how far the bevel reaches from the zero contour, tile units.
//   wall   — 0..1, a vertical drop before the bevel starts (a crisp AA step).
//   depth, base — final height = base + depth * p * coverage.
//   custom_y0..custom_y3 — `custom` preset only: y-values of a monotone
//            cubic through the fixed x-positions [0, 0.2, 0.6, 1.0] — the
//            same "fixed x, adjustable y" shape the Curve node's own
//            v0..v4 params already use, just 4 points instead of 5.
//
// All seven named presets plus the custom pchip ported from the public
// formulas in docs/hardsurface_nodes/HARDSURFACE_NODES.md §3.4 (a standard,
// public construction — Fritsch-Carlson monotone cubic interpolation,
// public math) — reimplemented from the formulas, not copied code.

struct BevelProfileParams {
    resolution: u32,
    preset: u32,
    inward: u32,
    width: f32,
    wall: f32,
    depth: f32,
    base: f32,
    custom_y0: f32,
    custom_y1: f32,
    custom_y2: f32,
    custom_y3: f32,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> params: BevelProfileParams;
@group(0) @binding(1) var sdf_tex: texture_2d<f32>;
@group(0) @binding(2) var height_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var edge_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var top_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var u_out: texture_storage_2d<rgba16float, write>;

fn coverage(s: f32, res: f32) -> f32 {
    return clamp(0.5 - s * res, 0.0, 1.0);
}

// Fritsch-Carlson monotone cubic through 4 fixed points — ported from
// `_pchip` (hs_reference.py), specialized to this fixed N=4 case (WGSL has
// no growable arrays, and every caller here has exactly 4 control points:
// the `lip` preset's own fixed curve, or the `custom` preset's 4 adjustable
// y-values).
fn pchip4(ys: vec4<f32>, xs: vec4<f32>, x_in: f32) -> f32 {
    let xs_a = array<f32, 4>(xs.x, xs.y, xs.z, xs.w);
    let ys_a = array<f32, 4>(ys.x, ys.y, ys.z, ys.w);
    let h = vec3<f32>(xs.y - xs.x, xs.z - xs.y, xs.w - xs.z);
    let dlt = vec3<f32>((ys.y - ys.x) / h.x, (ys.z - ys.y) / h.y, (ys.w - ys.z) / h.z);

    var m0 = dlt.x;
    var m3 = dlt.z;
    var m1 = 0.0;
    if dlt.x * dlt.y > 0.0 {
        let w1 = 2.0 * h.y + h.x;
        let w2 = h.y + 2.0 * h.x;
        m1 = (w1 + w2) / (w1 / dlt.x + w2 / dlt.y);
    }
    var m2 = 0.0;
    if dlt.y * dlt.z > 0.0 {
        let w1 = 2.0 * h.z + h.y;
        let w2 = h.z + 2.0 * h.y;
        m2 = (w1 + w2) / (w1 / dlt.y + w2 / dlt.z);
    }
    let m_a = array<f32, 4>(m0, m1, m2, m3);

    let x = clamp(x_in, xs.x, xs.w);
    var k = 0u;
    if x >= xs.z {
        k = 2u;
    } else if x >= xs.y {
        k = 1u;
    }
    let x0 = xs_a[k];
    let x1 = xs_a[k + 1u];
    let y0 = ys_a[k];
    let y1 = ys_a[k + 1u];
    let mk0 = m_a[k];
    let mk1 = m_a[k + 1u];
    let hk = x1 - x0;
    let t = (x - x0) / hk;
    let t2 = t * t;
    let t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * y0 + (t3 - 2.0 * t2 + t) * hk * mk0
        + (-2.0 * t3 + 3.0 * t2) * y1 + (t3 - t2) * hk * mk1;
}

fn profile(u: f32, preset: u32, custom_y: vec4<f32>) -> f32 {
    if preset == 0u {
        return u; // chamfer
    } else if preset == 1u {
        return sqrt(clamp(1.0 - (1.0 - u) * (1.0 - u), 0.0, 1.0)); // round
    } else if preset == 2u {
        return 1.0 - sqrt(clamp(1.0 - u * u, 0.0, 1.0)); // cove
    } else if preset == 3u {
        return u * u * (3.0 - 2.0 * u); // ogee
    } else if preset == 4u {
        // step: two chamfers with a ledge at 0.5 (breakpoints 0.35, 0.65)
        if u < 0.35 {
            return u / 0.35 * 0.5;
        } else if u < 0.65 {
            return 0.5;
        }
        return 0.5 + (u - 0.65) / 0.35 * 0.5;
    } else if preset == 5u {
        // lip: pchip([0,0.25,0.5,1.0], [0,1.0,0.75,0.75], u) — raised rim
        return pchip4(vec4<f32>(0.0, 1.0, 0.75, 0.75), vec4<f32>(0.0, 0.25, 0.5, 1.0), u);
    } else if preset == 6u {
        return 1.0; // flat: pure wall, no bevel
    }
    // custom: pchip through [0, 0.2, 0.6, 1.0] with user y-values
    return pchip4(custom_y, vec4<f32>(0.0, 0.2, 0.6, 1.0), u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coords = vec2<i32>(id.xy);
    let sdf = textureLoad(sdf_tex, coords, 0).r;
    let s = select(-sdf, sdf, params.inward != 0u);

    let res_f = f32(params.resolution);
    let u = clamp(-s / max(params.width, 1e-9), 0.0, 1.0);
    let custom_y = vec4<f32>(params.custom_y0, params.custom_y1, params.custom_y2, params.custom_y3);
    let p = params.wall + (1.0 - params.wall) * profile(u, params.preset, custom_y);
    let cov = coverage(s, res_f);
    let height = params.base + params.depth * p * cov;
    let edge = cov * f32(u < 1.0);
    let top = f32(u >= 1.0);

    textureStore(height_out, coords, vec4<f32>(height, height, height, 1.0));
    textureStore(edge_out, coords, vec4<f32>(edge, edge, edge, 1.0));
    textureStore(top_out, coords, vec4<f32>(top, top, top, 1.0));
    textureStore(u_out, coords, vec4<f32>(u, u, u, 1.0));
}
