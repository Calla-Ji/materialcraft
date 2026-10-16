// SPDX-License-Identifier: MIT OR Apache-2.0
// panel_layout.wgsl — Generators / Patterns / Panel Layout
// (docs/hardsurface_nodes/HARDSURFACE_NODES.md §3.5)
//
// `rects` (the guillotine subdivision's own leaves, CPU-computed —
// `mc_nodes::panel_layout::panel_rects`) is a dynamic-length storage
// buffer, same pattern `tube_raster.wgsl` already uses for a per-pixel
// loop over CPU-built geometry. For each pixel, every rect is tested in
// order and overwritten on a match (not early-exited) — mirrors the
// reference's own sequential `np.where` semantics exactly; rects never
// truly overlap once the torus is partitioned, so in practice only the
// genuine owner ever matches.
//
// Output: `sdf`     — iq rounded box per panel, inset by gap/2, negative
//                     inside (Rgba32Float — an Sdf socket).
//         `panel_id`    — hash01(index, 11, seed), random 0..1 per panel.
//         `panel_index` — the winning rect's own index, as a plain float
//                         (no dedicated int texture format in this
//                         codebase).
//         `uv_u`/`uv_v` — 0..1 local coordinate inside the owning panel.
//         `seam`        — 1 - coverage(sdf, res), an AA mask of the gaps.

struct PanelLayoutParams {
    resolution: u32,
    rect_count: u32,
    seed: u32,
    gap: f32,
    corner_radius: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct RectData {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    index: f32,
}

@group(0) @binding(0) var<uniform> params: PanelLayoutParams;
@group(0) @binding(1) var<storage, read> rects: array<RectData>;
@group(0) @binding(2) var sdf_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var panel_id_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var panel_index_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var uv_u_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var uv_v_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var seam_out: texture_storage_2d<rgba16float, write>;

fn coverage(s: f32, res: f32) -> f32 {
    return clamp(0.5 - s * res, 0.0, 1.0);
}

// iq rounded box SDF, centre at origin, half sizes hx/hy, corner radius r.
fn sd_round_box(px: f32, py: f32, hx: f32, hy: f32, r_in: f32) -> f32 {
    let r = min(r_in, min(hx, hy));
    let qx = abs(px) - hx + r;
    let qy = abs(py) - hy + r;
    return length(vec2<f32>(max(qx, 0.0), max(qy, 0.0))) + min(max(qx, qy), 0.0) - r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let res_f = f32(params.resolution);
    let uv = (vec2<f32>(id.xy) + 0.5) / res_f;

    var sdf: f32 = 1e20;
    var idx: f32 = -1.0;
    var uv_u: f32 = 0.0;
    var uv_v: f32 = 0.0;

    for (var i: u32 = 0u; i < params.rect_count; i = i + 1u) {
        let r = rects[i];
        let w = r.x1 - r.x0;
        let h = r.y1 - r.y0;
        let lx = fract(uv.x - r.x0);
        let ly = fract(uv.y - r.y0);
        if (lx < w && ly < h) {
            sdf = sd_round_box(
                lx - w * 0.5, ly - h * 0.5,
                w * 0.5 - params.gap * 0.5, h * 0.5 - params.gap * 0.5,
                params.corner_radius,
            );
            idx = r.index;
            uv_u = lx / w;
            uv_v = ly / h;
        }
    }

    let panel_id = hash01(u32(max(idx, 0.0)), 11u, params.seed);
    let seam = 1.0 - coverage(sdf, res_f);

    let coords = vec2<i32>(id.xy);
    textureStore(sdf_out, coords, vec4<f32>(sdf, sdf, sdf, 1.0));
    textureStore(panel_id_out, coords, vec4<f32>(panel_id, panel_id, panel_id, 1.0));
    textureStore(panel_index_out, coords, vec4<f32>(idx, idx, idx, 1.0));
    textureStore(uv_u_out, coords, vec4<f32>(uv_u, uv_u, uv_u, 1.0));
    textureStore(uv_v_out, coords, vec4<f32>(uv_v, uv_v, uv_v, 1.0));
    textureStore(seam_out, coords, vec4<f32>(seam, seam, seam, 1.0));
}
