// SPDX-License-Identifier: MIT OR Apache-2.0
// thread_shape.wgsl — Generators / Patterns / Thread
//
// Rebuilt from real yarn geometry (THREAD_NODE_SPEC.md, the owner's textile
// reference book, and thread_reference.py — see thread_shape.rs's module
// doc comment for the rebuild history and ground-truth verification). Every
// ply is a round TUBE of constant radius `r` — NOT a 1D brightness profile
// — whose centerline follows a helix around its parent thread's axis. The
// visible surface at each pixel is a genuine top-surface-height occlusion
// test across every strand stacked at that pixel's thread row: the
// physically highest tube wins outright (a z-buffer, in effect), so a ply's
// own width never changes and one ply crossing in front of another looks
// like one solid tube physically occluding another, not two profiles
// blending or pinching.
//
// Coordinates: the tile is [0,1)^2. The thread runs along +X (horizontal).
// Image Y points DOWN. Z is height, pointing toward the viewer. All lengths
// are in tile units (resolution-independent).
//
// Inputs:  none.
// Outputs: height    — primary output; z_max / r0, clamped to [0,1]. Chain
//                       into Fabric Generator's Thread input, same as every
//                       earlier version of this node.
//          ply_id     — (winning strand's flat index + 1) / strand count.
//                       Lets an artist color each ply differently via
//                       Gradient Map. 0 in the background.
//          fiber      — the winning strand's own groove `ridge` value
//                       (THREAD_NODE_SPEC.md §4). Use as roughness/AO
//                       detail. 0 in the background or with Fiber Twist
//                       "None".
//          mask       — 1 wherever any strand is hit, else 0.
//
// `threads` rows stack in Y (an integer, so this tiles vertically the same
// way `plies`' turn count being an integer makes it tile horizontally).
// Each pixel's own row is `floor(uv.y * threads)` — evaluating only that
// row's own strands (rather than sweeping every row against every pixel,
// the way thread_reference.py's row loop does for simplicity) is exactly
// equivalent, since a pixel's local Y relative to a *different* row's
// center can never fall inside that row's own +-row_height/2 band — the
// rows tile the full image height with no overlap.

struct ThreadShapeParams {
    resolution: u32,
    threads: u32,
    plies: u32,
    s_ply: i32,
    k_p: u32,
    fibers_per_ply: u32,
    has_cable: u32,
    cable_groups: u32,
    s_cable: i32,
    k_c: u32,
    seed: u32,
    fiber_sign: i32,
    r0: f32,
    a: f32,
    r: f32,
    fiber_turns: f32,
    fiber_depth: f32,
    big_a: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: ThreadShapeParams;
@group(0) @binding(1) var height_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var ply_id_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var fiber_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var mask_out: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265358979;
const TWO_PI: f32 = 6.28318530717958;

// Top-surface height of a tilted round tube of radius `r`, cut by the plane
// x=const (THREAD_NODE_SPEC.md §3.5). `qy` is this pixel's Y offset from
// the tube's own centerline at this x; `(ty, tz)` is the tube's own
// (unnormalized) direction — dcenter_y/dx, dcenter_z/dx. A tube around a
// line with direction d=(1,ty,tz) (normalized) is every point whose
// distance to that line is r; in the x=const plane that's a quadratic in
// qz (an ellipse, wider than 2r when the tube tilts — a tilted cylinder's
// cross-section is an ellipse, not a circle). Returns (qz, hit) — hit is
// 0.0 where this pixel misses the tube entirely (negative discriminant).
fn tube_top(qy: f32, ty: f32, tz: f32, r: f32) -> vec2<f32> {
    let n = sqrt(1.0 + ty * ty + tz * tz);
    let dy = ty / n;
    let dz = tz / n;
    let a_coef = 1.0 - dz * dz; // always > 0: dy^2+dz^2 < 1 for a unit vector (1,dy,dz)
    let b_coef = -2.0 * dy * dz * qy;
    let c_coef = (1.0 - dy * dy) * qy * qy - r * r;
    let disc = b_coef * b_coef - 4.0 * a_coef * c_coef;
    if disc < 0.0 {
        return vec2<f32>(0.0, 0.0);
    }
    let qz = (-b_coef + sqrt(disc)) / (2.0 * a_coef);
    return vec2<f32>(qz, 1.0);
}

struct StrandCentre {
    cy: f32,
    cz: f32,
    ty: f32,
    tz: f32,
}

// Strand `k`'s (0..total_strands) centerline position and direction at
// along-thread position `x`. Flat-indexes (group, ply) when cabled:
// g = k / plies, j = k % plies (THREAD_NODE_SPEC.md §3.4) — group and ply
// motion add together (a helix of helices), each with its own analytic
// derivative computed directly rather than the reference script's finite
// difference (exact, and avoids an extra eval per strand).
fn strand_centre(k: u32, x: f32, phase0: f32) -> StrandCentre {
    let n = f32(params.plies);
    let theta_rate = f32(params.s_ply) * TWO_PI * f32(params.k_p);

    if params.has_cable != 0u {
        let g = k / params.plies;
        let j = k % params.plies;
        let g_count = f32(params.cable_groups);

        let theta_g_rate = f32(params.s_cable) * TWO_PI * f32(params.k_c);
        let theta_g = theta_g_rate * x + TWO_PI * f32(g) / g_count + phase0;
        let gy = params.big_a * cos(theta_g);
        let gz = params.big_a * sin(theta_g);
        let dgy = -params.big_a * sin(theta_g) * theta_g_rate;
        let dgz = params.big_a * cos(theta_g) * theta_g_rate;

        let theta = theta_rate * x + TWO_PI * f32(j) / n + phase0 * 0.37;
        let py = params.a * cos(theta);
        let pz = params.a * sin(theta);
        let dpy = -params.a * sin(theta) * theta_rate;
        let dpz = params.a * cos(theta) * theta_rate;

        return StrandCentre(gy + py, gz + pz, dgy + dpy, dgz + dpz);
    }

    let theta = theta_rate * x + TWO_PI * f32(k) / n + phase0;
    let cy = params.a * cos(theta);
    let cz = params.a * sin(theta);
    let dcy = -params.a * sin(theta) * theta_rate;
    let dcz = params.a * cos(theta) * theta_rate;
    return StrandCentre(cy, cz, dcy, dcz);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);
    let x = uv.x;

    let threads_f = f32(params.threads);
    let row = u32(floor(uv.y * threads_f));
    let row_h = 1.0 / threads_f;
    let row_centre = (f32(row) + 0.5) * row_h;
    let y_local = uv.y - row_centre;

    // Per-row random twist-phase offset, purely so stacked rows don't all
    // start their twist at the same point (thread_shape.rs's module doc
    // comment explains why this uses the project's existing `hash2` rather
    // than trying to match thread_reference.py's numpy RNG bit-for-bit).
    let phase0 = hash2(vec2<i32>(i32(row), 0), params.seed) * TWO_PI;

    let total_strands = select(params.plies, params.cable_groups * params.plies, params.has_cable != 0u);

    var best_z = -1.0e9;
    var best_k = 0u;
    var best_ridge = 0.0;
    var hit = false;

    for (var k = 0u; k < total_strands; k = k + 1u) {
        let sc = strand_centre(k, x, phase0);
        let qy = y_local - sc.cy;
        let tt = tube_top(qy, sc.ty, sc.tz, params.r);
        if tt.y < 0.5 {
            continue;
        }
        let qz = tt.x;
        var z = sc.cz + qz;

        // Fiber grooves: the angle around the strand's own center in a
        // WORLD-aligned frame (THREAD_NODE_SPEC.md §4 — deliberately not a
        // frame that rotates with the strand's own twist, since that twist
        // is already folded in by measuring phi in world space).
        var ridge = 1.0;
        if params.fiber_sign != 0 {
            let phi = atan2(qz, qy);
            let psi = f32(params.fibers_per_ply) * (phi - f32(params.fiber_sign) * TWO_PI * params.fiber_turns * x);
            let f = fract(psi / TWO_PI);
            ridge = sqrt(clamp(1.0 - (2.0 * f - 1.0) * (2.0 * f - 1.0), 0.0, 1.0));
            z = z - (1.0 - ridge) * params.fiber_depth * params.r;
        }

        if z > best_z {
            best_z = z;
            best_k = k;
            best_ridge = ridge;
            hit = true;
        }
    }

    var height = 0.0;
    var ply_id_val = 0.0;
    var fiber_val = 0.0;
    var mask_val = 0.0;
    if hit {
        height = clamp(best_z / params.r0, 0.0, 1.0);
        ply_id_val = (f32(best_k) + 1.0) / f32(total_strands);
        fiber_val = best_ridge;
        mask_val = 1.0;
    }

    textureStore(height_out, vec2<i32>(id.xy), vec4<f32>(height, height, height, 1.0));
    textureStore(ply_id_out, vec2<i32>(id.xy), vec4<f32>(ply_id_val, ply_id_val, ply_id_val, 1.0));
    textureStore(fiber_out, vec2<i32>(id.xy), vec4<f32>(fiber_val, fiber_val, fiber_val, 1.0));
    textureStore(mask_out, vec2<i32>(id.xy), vec4<f32>(mask_val, mask_val, mask_val, 1.0));
}
