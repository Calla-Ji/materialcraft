// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-mesh preparation for the GPU vertex-pulling tessellation/displacement
//! viewport (replaces `adaptive_mesh.rs`'s CPU-built quadtree — see that
//! module's doc comment history for why a CPU-built mesh whose size scales
//! with Tessellation hit a hard wgpu buffer-size ceiling and, separately,
//! froze the UI thread on every density/resolution change).
//!
//! This is a direct, mechanical Rust port of an independently-authored
//! Python reference (`viewport_reference.py`, kept in the private notes
//! repo alongside the full spec) — not derived from, or a reimplementation
//! of, any commercial tool's own code. The technique itself (decode a
//! sub-triangle lattice index in the vertex shader, evaluate position/
//! height/normal procedurally, draw instanced with no CPU-built vertex
//! buffer) is a well-known real-time graphics technique independent of
//! wgpu's lack of a hardware tessellation stage
//! (<https://github.com/gpuweb/gpuweb/issues/445>,
//! <https://github.com/gfx-rs/wgpu/issues/222>).
//!
//! [`prepare`] runs once per base mesh (and again only when the graph's
//! output resolution changes, since only `lam` depends on `tex_size`) and
//! derives everything the vertex shader needs to evaluate an arbitrarily
//! fine lattice point without ever rebuilding geometry on the CPU:
//! - **weld**: vertices at the same position (within 1e-6) share an id, so
//!   a vertex-shader evaluation of a shared edge/corner from either
//!   adjacent base triangle lands on the exact same input data.
//! - **ndisp**: the displacement normal (averaged distinct authored
//!   normals per welded group) — using this instead of a single face
//!   normal is what keeps a cube's corners from tearing open under
//!   displacement.
//! - **tan_u/tan_v**: per-vertex surface tangent frame (for shading
//!   normals), *not* welded — two faces meeting at a UV seam legitimately
//!   have different tangent frames there.
//! - **lam/cuv/ccount/twin/twin_flag/nb**: seam-aware mip level-of-detail
//!   and twin-UV bookkeeping, so both sides of a UV seam evaluate to
//!   bit-identical positions (the standard crack-free fix for this whole
//!   class of GPU tessellation technique — verified in the reference pack
//!   via a bit-exact crack test on real GPU output, not just visually).
//!
//! Kept in `f64` throughout, matching the Python reference's own double
//! precision (and the golden fixtures' 1e-9 tolerance, which assumes it) —
//! `f32` only enters once this data is packed into GPU buffers downstream.

// `prepare`/`make_rounded_cube` are used by `viewport3d.rs`'s real render
// path now; `make_plane`/`make_sphere`/`make_cube`/`decode`/`build_mips`
// remain test/golden-validation-only (the reference pack's own non-cube/
// non-rounded-cube test meshes, and pure functions this module's tests
// check against `testdata/tess_golden/` directly) — narrower
// `#[allow(dead_code)]`s on those below instead of a blanket one.

use glam::{DVec2, DVec3};
use std::collections::HashMap;

/// A base mesh, before (`pos`/`nrm`/`uv`/`tri`) and after (everything else)
/// [`prepare`]. Field shapes/semantics match `viewport_reference.py`'s
/// `Mesh` dataclass exactly — see that file (or the module doc above) for
/// what each one means.
pub struct Mesh {
    pub pos: Vec<DVec3>,
    pub nrm: Vec<DVec3>,
    pub uv: Vec<DVec2>,
    /// Vertex indices, counter-clockwise seen from outside.
    pub tri: Vec<[u32; 3]>,

    pub weld: Vec<u32>,
    pub ndisp: Vec<DVec3>,
    pub tan_u: Vec<DVec3>,
    pub tan_v: Vec<DVec3>,
    pub lam: Vec<f64>,
    /// Up to 4 distinct UVs of the welded group, sorted — used at seam
    /// corners. Unused slots (beyond `ccount`) are zero.
    pub cuv: Vec<[DVec2; 4]>,
    pub ccount: Vec<u32>,
    /// Per base triangle, per edge (0: v0->v1, 1: v1->v2, 2: v2->v0): the
    /// other side's uv at the same two welded vertices, `[ax, ay, bx, by]`.
    /// Only meaningful where `twin_flag` is set.
    pub twin: Vec<[[f64; 4]; 3]>,
    pub twin_flag: Vec<[bool; 3]>,
    /// Neighbour triangle per edge, -1 = border (no neighbour).
    pub nb: Vec<[i32; 3]>,
}

impl Mesh {
    fn new(pos: Vec<DVec3>, nrm: Vec<DVec3>, uv: Vec<DVec2>, tri: Vec<[u32; 3]>) -> Self {
        let v = pos.len();
        let t = tri.len();
        Mesh {
            pos,
            nrm,
            uv,
            tri,
            weld: vec![0; v],
            ndisp: vec![DVec3::ZERO; v],
            tan_u: vec![DVec3::ZERO; v],
            tan_v: vec![DVec3::ZERO; v],
            lam: vec![0.0; v],
            cuv: vec![[DVec2::ZERO; 4]; v],
            ccount: vec![0; v],
            twin: vec![[[0.0; 4]; 3]; t],
            twin_flag: vec![[false; 3]; t],
            nb: vec![[-1; 3]; t],
        }
    }
}

/// XZ plane, y up, centred, `res` x `res` quads. u along +x, v along -z (so
/// `cross(dP/du, dP/dv) = +y`).
#[allow(dead_code)] // test/golden-validation only — see module doc comment
pub fn make_plane(res: u32, size: f64, tiling: f64) -> Mesh {
    let mut pos = Vec::new();
    let mut uv = Vec::new();
    for j in 0..=res {
        for i in 0..=res {
            let u = i as f64 / res as f64;
            let v = j as f64 / res as f64;
            pos.push(DVec3::new((u - 0.5) * size, 0.0, (0.5 - v) * size));
            uv.push(DVec2::new(u * tiling, v * tiling));
        }
    }
    let nrm = vec![DVec3::new(0.0, 1.0, 0.0); pos.len()];
    let mut tri = Vec::new();
    for j in 0..res {
        for i in 0..res {
            let a = j * (res + 1) + i;
            let (b, c, d) = (a + 1, a + res + 1, a + res + 2);
            tri.push([a, b, d]);
            tri.push([a, d, c]);
        }
    }
    Mesh::new(pos, nrm, uv, tri)
}

/// UV sphere, y up. u around (seam at u = 0 / `tiling.0`), v from top (0) to
/// bottom. Pole rows are single triangles (no degenerate triangles).
#[allow(dead_code)] // test/golden-validation only — see module doc comment
pub fn make_sphere(seg: u32, rings: u32, radius: f64, tiling: (f64, f64)) -> Mesh {
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    let mut uv = Vec::new();
    for r in 0..=rings {
        let th = std::f64::consts::PI * r as f64 / rings as f64;
        for s in 0..=seg {
            let ph = 2.0 * std::f64::consts::PI * s as f64 / seg as f64;
            let n = DVec3::new(th.sin() * ph.cos(), th.cos(), -th.sin() * ph.sin());
            pos.push(n * radius);
            nrm.push(n);
            uv.push(DVec2::new(
                s as f64 / seg as f64 * tiling.0,
                r as f64 / rings as f64 * tiling.1,
            ));
        }
    }
    let w = seg + 1;
    let mut tri: Vec<[u32; 3]> = Vec::new();
    for r in 0..rings {
        for s in 0..seg {
            let a = r * w + s;
            let (b, c, d) = (a + 1, a + w, a + w + 1);
            if r != rings - 1 {
                tri.push([a, c, d]);
            }
            if r != 0 {
                tri.push([a, d, b]);
            }
        }
    }
    // Orient every triangle counter-clockwise seen from outside.
    for t in tri.iter_mut() {
        let [a, b, c] = *t;
        let (pa, pb, pc) = (pos[a as usize], pos[b as usize], pos[c as usize]);
        if (pb - pa).cross(pc - pa).dot(pa + pb + pc) < 0.0 {
            *t = [a, c, b];
        }
    }
    Mesh::new(pos, nrm, uv, tri)
}

/// Cube, 6 faces of `res` x `res` quads each, every face its own uv 0..1
/// and face normal (so all 12 edges are UV seams).
#[allow(dead_code)] // test/golden-validation only — see module doc comment
pub fn make_cube(res: u32, size: f64) -> Mesh {
    let faces: [([f64; 3], [f64; 3], [f64; 3]); 6] = [
        ([-1.0, -1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, -1.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, -1.0, 1.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, -1.0, -1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ];
    let h = size / 2.0;
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    let mut uv = Vec::new();
    let mut tri: Vec<[u32; 3]> = Vec::new();
    for (o, ua, va) in faces {
        let o = DVec3::from(o);
        let ua = DVec3::from(ua);
        let va = DVec3::from(va);
        let n = ua.cross(va);
        let base = pos.len() as u32;
        for j in 0..=res {
            for i in 0..=res {
                let p = h
                    * (o + 2.0 * ua * (i as f64 / res as f64) + 2.0 * va * (j as f64 / res as f64));
                pos.push(p);
                nrm.push(n);
                uv.push(DVec2::new(i as f64 / res as f64, j as f64 / res as f64));
            }
        }
        for j in 0..res {
            for i in 0..res {
                let a = base + j * (res + 1) + i;
                let (b, c, d) = (a + 1, a + res + 1, a + res + 2);
                tri.push([a, b, d]);
                tri.push([a, d, c]);
            }
        }
    }
    Mesh::new(pos, nrm, uv, tri)
}

/// 2x2 box-average mip chain down to 1x1 (`size` x `size`, `size` a power of
/// two). Matches `viewport_reference.py::build_mips`: each level is derived
/// from the *previous f32-rounded* level (not re-derived from full
/// precision each time) via an f64 average rounded back to f32 — this
/// chaining is what matches `tex_*.json`'s golden values closely enough to
/// clear its 1e-7 tolerance.
#[allow(dead_code)] // test/golden-validation only — see module doc comment
pub fn build_mips(h0: &[f32], size: u32) -> Vec<Vec<f32>> {
    assert!(
        size.is_power_of_two(),
        "build_mips: size must be a power of two"
    );
    assert_eq!(
        h0.len(),
        (size * size) as usize,
        "build_mips: size mismatch"
    );
    let mut mips = vec![h0.to_vec()];
    let mut s = size;
    while s > 1 {
        let prev = mips.last().expect("just pushed");
        let half = s / 2;
        let mut next = vec![0.0f32; (half * half) as usize];
        for y in 0..half {
            for x in 0..half {
                let get = |xx: u32, yy: u32| prev[(yy * s + xx) as usize] as f64;
                let sum = get(2 * x, 2 * y)
                    + get(2 * x + 1, 2 * y)
                    + get(2 * x, 2 * y + 1)
                    + get(2 * x + 1, 2 * y + 1);
                next[(y * half + x) as usize] = (sum * 0.25) as f32;
            }
        }
        mips.push(next);
        s = half;
    }
    mips
}

/// `vertex_index -> (a0, a1, a2, corner)` for a lattice of `N` segments per
/// base-triangle edge — see `displace.wgsl`'s own `decode` (this is the
/// same thing, ported for use in Rust-side tests/tooling). `corner` is
/// which of the sub-triangle's 3 winding positions this vertex is (used
/// for wireframe rendering), not a lattice coordinate itself.
pub fn decode(vid: u32, n: u32) -> ([u32; 3], u32) {
    let t = vid / 3;
    let corner = vid % 3;
    let mut r = (t as f64).sqrt() as u32;
    while (r + 1) * (r + 1) <= t {
        r += 1;
    }
    while r * r > t {
        r -= 1;
    }
    let k = t - r * r;
    let (i, j) = if k.is_multiple_of(2) {
        let a = k / 2;
        match corner {
            0 => (r, a),
            1 => (r + 1, a),
            _ => (r + 1, a + 1),
        }
    } else {
        let a = (k - 1) / 2;
        match corner {
            0 => (r, a),
            1 => (r + 1, a + 1),
            _ => (r, a + 1),
        }
    };
    ([n - i, i - j, j], corner)
}

/// A cube with filleted edges/corners, matching `viewport3d.rs`'s own
/// `generate_rounded_cube` exactly (same per-vertex "pull outward from an
/// inner box" construction, `radius: 0.0` degenerating cleanly to a sharp
/// cube) — this is the *base* mesh fed through [`prepare`] for the
/// vertex-pulling tessellation path, so `segments` here is a fixed,
/// moderate constant chosen for a smooth-looking fillet (see
/// `viewport3d.rs`'s `BASE_FILLET_SEGMENTS`), independent of the user's
/// own Tessellation slider — unlike a flat cube's faces, a curved fillet
/// doesn't gain any more curvature from subdividing a lattice point
/// further inside one of its base triangles (that's a pure barycentric/
/// affine blend of 3 corner positions, i.e. still flat), so the fillet's
/// own visual smoothness has to already be "baked in" at the base-mesh
/// level — raising Tessellation only adds *height-detail* density within
/// each already-curved base triangle, not more curvature.
pub fn make_rounded_cube(segments: u32, radius: f64, size: f64) -> Mesh {
    let radius = radius.clamp(0.0, 1.0);
    let inner = 1.0 - radius;
    let h = size / 2.0;
    let faces: [([f64; 3], [f64; 3], [f64; 3]); 6] = [
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ];
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    let mut uv = Vec::new();
    let mut tri: Vec<[u32; 3]> = Vec::new();
    for (normal, right, up) in faces {
        let normal = DVec3::from(normal);
        let right = DVec3::from(right);
        let up = DVec3::from(up);
        let base = pos.len() as u32;
        for j in 0..=segments {
            let v = j as f64 / segments as f64 * 2.0 - 1.0;
            for i in 0..=segments {
                let u = i as f64 / segments as f64 * 2.0 - 1.0;
                let p = normal + right * u + up * v;
                let clamped = p.clamp(DVec3::splat(-inner), DVec3::splat(inner));
                let delta = p - clamped;
                let n = if delta.length_squared() > 1e-12 {
                    delta.normalize()
                } else {
                    normal
                };
                pos.push(h * (clamped + n * radius));
                nrm.push(n);
                uv.push(DVec2::new(
                    i as f64 / segments as f64,
                    1.0 - j as f64 / segments as f64,
                ));
            }
        }
        for j in 0..segments {
            for i in 0..segments {
                let a = base + j * (segments + 1) + i;
                let (b, c, d) = (a + 1, a + segments + 1, a + segments + 2);
                tri.push([a, b, d]);
                tri.push([a, d, c]);
            }
        }
    }
    Mesh::new(pos, nrm, uv, tri)
}

fn allclose(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

fn round9(x: f64) -> i64 {
    (x * 1e9).round() as i64
}

/// One triangle's occurrence of a shared edge: (triangle, edge index, p, q).
type EdgeOccurrence = (usize, usize, usize, usize);

/// Derive every per-vertex/per-triangle field the shader needs. Runs once
/// per mesh (and when `tex_size` changes — only `lam` depends on it). See
/// the module doc comment for what each derived field is for.
pub fn prepare(mut m: Mesh, tex_size: u32) -> Mesh {
    let v = m.pos.len();
    let t = m.tri.len();

    // Weld by position (quantised to 1e-6); ids in order of first
    // appearance, matching a Python dict's insertion-order `keys`.
    let mut keys: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut weld = vec![0u32; v];
    for (i, &p) in m.pos.iter().enumerate() {
        let key = (
            (p.x * 1e6).round() as i64,
            (p.y * 1e6).round() as i64,
            (p.z * 1e6).round() as i64,
        );
        let next_id = keys.len() as u32;
        weld[i] = *keys.entry(key).or_insert(next_id);
    }

    // Canonical position per welded group (first vertex) -> bitwise
    // identical duplicates.
    let mut first: HashMap<u32, usize> = HashMap::new();
    for (i, &w) in weld.iter().enumerate() {
        first.entry(w).or_insert(i);
    }
    m.pos = (0..v).map(|i| m.pos[first[&weld[i]]]).collect();

    let mut grp: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &w) in weld.iter().enumerate() {
        grp.entry(w).or_default().push(i);
    }

    // Displacement normal: average of the distinct authored normals over
    // the welded group (unweighted, distinct normals only).
    let mut ndisp = vec![DVec3::ZERO; v];
    for ids in grp.values() {
        let mut ns: Vec<DVec3> = Vec::new();
        for &i in ids {
            let n = m.nrm[i].normalize();
            let dup = ns.iter().any(|q| {
                allclose(n.x, q.x, 1e-9, 1e-5)
                    && allclose(n.y, q.y, 1e-9, 1e-5)
                    && allclose(n.z, q.z, 1e-9, 1e-5)
            });
            if !dup {
                ns.push(n);
            }
        }
        let sum = ns.iter().fold(DVec3::ZERO, |acc, &n| acc + n);
        let avg = sum / sum.length();
        for &i in ids {
            ndisp[i] = avg;
        }
    }

    // Tangents dP/du, dP/dv per triangle, averaged per vertex (not welded
    // — uv frames differ across seams).
    let mut tan_u = vec![DVec3::ZERO; v];
    let mut tan_v = vec![DVec3::ZERO; v];
    for tr in &m.tri {
        let [a, b, c] = [tr[0] as usize, tr[1] as usize, tr[2] as usize];
        let e1 = m.pos[b] - m.pos[a];
        let e2 = m.pos[c] - m.pos[a];
        let d1 = m.uv[b] - m.uv[a];
        let d2 = m.uv[c] - m.uv[a];
        let det = d1.x * d2.y - d1.y * d2.x;
        if det.abs() < 1e-14 {
            continue;
        }
        let tu = (e1 * d2.y - e2 * d1.y) / det;
        let tv = (e2 * d1.x - e1 * d2.x) / det;
        for &i in &[a, b, c] {
            tan_u[i] += tu;
            tan_v[i] += tv;
        }
    }
    let mut cnt = vec![0u32; v];
    for tr in &m.tri {
        for &idx in tr {
            cnt[idx as usize] += 1;
        }
    }
    for i in 0..v {
        let c = cnt[i].max(1) as f64;
        tan_u[i] /= c;
        tan_v[i] /= c;
    }

    // lam: log2(tex_size * mean uv length of incident edges), welded (mean
    // over the group).
    let mut acc = vec![0.0f64; v];
    let mut na = vec![0.0f64; v];
    for tr in &m.tri {
        let [a, b, c] = [tr[0] as usize, tr[1] as usize, tr[2] as usize];
        for &(p, q) in &[(a, b), (b, c), (c, a)] {
            let l = (m.uv[p] - m.uv[q]).length() * tex_size as f64;
            acc[p] += l;
            na[p] += 1.0;
            acc[q] += l;
            na[q] += 1.0;
        }
    }
    let mut lam = vec![0.0f64; v];
    for ids in grp.values() {
        let acc_sum: f64 = ids.iter().map(|&i| acc[i]).sum();
        let na_sum: f64 = ids.iter().map(|&i| na[i]).sum();
        let val = (acc_sum / na_sum.max(1.0)).max(1e-9).log2();
        for &i in ids {
            lam[i] = val;
        }
    }

    // Seam-corner uv sets: distinct uvs of the group, sorted, at most 4.
    let mut cuv = vec![[DVec2::ZERO; 4]; v];
    let mut ccount = vec![0u32; v];
    for ids in grp.values() {
        let mut seen: Vec<(i64, i64)> = Vec::new();
        for &i in ids {
            let key = (round9(m.uv[i].x), round9(m.uv[i].y));
            if !seen.contains(&key) {
                seen.push(key);
            }
        }
        seen.sort_unstable();
        let us: Vec<DVec2> = seen
            .iter()
            .take(4)
            .map(|&key| {
                let i = *ids
                    .iter()
                    .find(|&&i| (round9(m.uv[i].x), round9(m.uv[i].y)) == key)
                    .expect("key came from this group's own uvs");
                m.uv[i]
            })
            .collect();
        for &i in ids {
            ccount[i] = us.len() as u32;
            for (k, &q) in us.iter().enumerate() {
                cuv[i][k] = q;
            }
        }
    }

    // Edges: neighbour + twin uvs (seam edges have the same welded
    // endpoints but different uvs).
    let mut edge_map: HashMap<(u32, u32), Vec<EdgeOccurrence>> = HashMap::new();
    for (ti, tr) in m.tri.iter().enumerate() {
        let [a, b, c] = [tr[0] as usize, tr[1] as usize, tr[2] as usize];
        for (e, &(p, q)) in [(a, b), (b, c), (c, a)].iter().enumerate() {
            let key = (weld[p].min(weld[q]), weld[p].max(weld[q]));
            edge_map.entry(key).or_default().push((ti, e, p, q));
        }
    }
    let mut nb = vec![[-1i32; 3]; t];
    let mut twin = vec![[[0.0f64; 4]; 3]; t];
    let mut twin_flag = vec![[false; 3]; t];
    for lst in edge_map.values() {
        if lst.len() != 2 {
            continue;
        }
        let (t0, e0, p0, q0) = lst[0];
        let (t1, e1, p1, q1) = lst[1];
        nb[t0][e0] = t1 as i32;
        nb[t1][e1] = t0 as i32;

        let mut u_at: HashMap<u32, DVec2> = HashMap::new();
        u_at.insert(weld[p1], m.uv[p1]);
        u_at.insert(weld[q1], m.uv[q1]);
        let tw0 = [
            u_at[&weld[p0]].x,
            u_at[&weld[p0]].y,
            u_at[&weld[q0]].x,
            u_at[&weld[q0]].y,
        ];

        let mut u_at0: HashMap<u32, DVec2> = HashMap::new();
        u_at0.insert(weld[p0], m.uv[p0]);
        u_at0.insert(weld[q0], m.uv[q0]);
        let tw1 = [
            u_at0[&weld[p1]].x,
            u_at0[&weld[p1]].y,
            u_at0[&weld[q1]].x,
            u_at0[&weld[q1]].y,
        ];

        let seam = !(allclose(tw0[0], m.uv[p0].x, 1e-8, 1e-5)
            && allclose(tw0[1], m.uv[p0].y, 1e-8, 1e-5)
            && allclose(tw0[2], m.uv[q0].x, 1e-8, 1e-5)
            && allclose(tw0[3], m.uv[q0].y, 1e-8, 1e-5));
        if seam {
            twin[t0][e0] = tw0;
            twin[t1][e1] = tw1;
            twin_flag[t0][e0] = true;
            twin_flag[t1][e1] = true;
        }
    }

    m.weld = weld;
    m.ndisp = ndisp;
    m.tan_u = tan_u;
    m.tan_v = tan_v;
    m.lam = lam;
    m.cuv = cuv;
    m.ccount = ccount;
    m.twin = twin;
    m.twin_flag = twin_flag;
    m.nb = nb;
    m
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // indices drive several parallel arrays/golden fields at once
mod tests {
    use super::*;

    /// Loads one of `testdata/tess_golden/mesh_*.json` (generated by the
    /// Python reference's own `make_goldens.py`) and checks every field of
    /// a freshly-`prepare()`d mesh against it: `weld`/`ccount`/`twin_flag`/
    /// `nb` exactly, every float within 1e-9 — the tolerances the spec
    /// itself calls for, since both sides run in double precision.
    fn check_against_golden(name: &str, m: Mesh) {
        let path = format!(
            "{}/testdata/tess_golden/mesh_{name}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
        let golden: serde_json::Value = serde_json::from_str(&text).unwrap();

        let v = m.pos.len();
        let t = m.tri.len();

        let arr = |field: &str| golden[field].as_array().unwrap().clone();
        let f = |x: &serde_json::Value| x.as_f64().unwrap();
        let i = |x: &serde_json::Value| x.as_i64().unwrap();

        assert_eq!(arr("pos").len(), v, "{name}: vertex count");
        assert_eq!(arr("tri").len(), t, "{name}: triangle count");

        for field in ["pos", "nrm", "uv", "ndisp", "tan_u", "tan_v"] {
            let g = arr(field);
            for idx in 0..v {
                let row = g[idx].as_array().unwrap();
                let got: Vec<f64> = match field {
                    "pos" => vec![m.pos[idx].x, m.pos[idx].y, m.pos[idx].z],
                    "nrm" => vec![m.nrm[idx].x, m.nrm[idx].y, m.nrm[idx].z],
                    "uv" => vec![m.uv[idx].x, m.uv[idx].y],
                    "ndisp" => vec![m.ndisp[idx].x, m.ndisp[idx].y, m.ndisp[idx].z],
                    "tan_u" => vec![m.tan_u[idx].x, m.tan_u[idx].y, m.tan_u[idx].z],
                    "tan_v" => vec![m.tan_v[idx].x, m.tan_v[idx].y, m.tan_v[idx].z],
                    _ => unreachable!(),
                };
                for (k, gv) in row.iter().enumerate() {
                    let gv = f(gv);
                    let d = (got[k] - gv).abs();
                    assert!(
                        d < 1e-9,
                        "{name}: {field}[{idx}][{k}] = {} vs golden {gv} (diff {d})",
                        got[k]
                    );
                }
            }
        }

        for idx in 0..v {
            assert_eq!(
                i(&arr("weld")[idx]) as u32,
                m.weld[idx],
                "{name}: weld[{idx}]"
            );
            assert_eq!(
                i(&arr("ccount")[idx]) as u32,
                m.ccount[idx],
                "{name}: ccount[{idx}]"
            );
            let glam = f(&arr("lam")[idx]);
            assert!((glam - m.lam[idx]).abs() < 1e-9, "{name}: lam[{idx}]");
            let gcuv = arr("cuv")[idx].as_array().unwrap().clone();
            for k in 0..4 {
                let row = gcuv[k].as_array().unwrap();
                assert!(
                    (f(&row[0]) - m.cuv[idx][k].x).abs() < 1e-9,
                    "{name}: cuv[{idx}][{k}].x"
                );
                assert!(
                    (f(&row[1]) - m.cuv[idx][k].y).abs() < 1e-9,
                    "{name}: cuv[{idx}][{k}].y"
                );
            }
        }

        for idx in 0..t {
            let gnb = arr("nb")[idx].as_array().unwrap().clone();
            let gtf = arr("twin_flag")[idx].as_array().unwrap().clone();
            let gtw = arr("twin")[idx].as_array().unwrap().clone();
            for e in 0..3 {
                assert_eq!(i(&gnb[e]) as i32, m.nb[idx][e], "{name}: nb[{idx}][{e}]");
                assert_eq!(
                    i(&gtf[e]) != 0,
                    m.twin_flag[idx][e],
                    "{name}: twin_flag[{idx}][{e}]"
                );
                let row = gtw[e].as_array().unwrap();
                for k in 0..4 {
                    let d = (f(&row[k]) - m.twin[idx][e][k]).abs();
                    assert!(d < 1e-9, "{name}: twin[{idx}][{e}][{k}] diff {d}");
                }
            }
        }
    }

    #[test]
    fn prepare_matches_golden_plane4() {
        check_against_golden("plane4", prepare(make_plane(4, 2.0, 1.0), 64));
    }

    #[test]
    fn prepare_matches_golden_sphere12x6() {
        check_against_golden(
            "sphere12x6",
            prepare(make_sphere(12, 6, 1.0, (2.0, 1.0)), 64),
        );
    }

    #[test]
    fn prepare_matches_golden_cube2() {
        check_against_golden("cube2", prepare(make_cube(2, 2.0), 64));
    }

    /// No external golden exists for this one (it's original geometry, not
    /// from the reference pack) — checks the structural invariants `prepare`
    /// itself relies on instead: a closed mesh (no border edges) at both a
    /// sharp (`radius=0`) and genuinely filleted corner.
    #[test]
    fn rounded_cube_is_closed_sharp_and_filleted() {
        for radius in [0.0, 0.25] {
            let m = prepare(make_rounded_cube(6, radius, 2.0), 64);
            for (t, edges) in m.nb.iter().enumerate() {
                for (e, &n) in edges.iter().enumerate() {
                    assert!(
                        n >= 0,
                        "radius={radius}: triangle {t} edge {e} has no neighbour (border edge)"
                    );
                }
            }
        }
    }

    /// At `radius=0` every vertex should land exactly on the unit cube's
    /// own face (the fillet's "pull outward" never actually triggers) —
    /// the degenerate case `MeshKind::Cube` relies on.
    #[test]
    fn rounded_cube_radius_zero_is_a_sharp_cube() {
        let m = make_rounded_cube(4, 0.0, 2.0);
        for p in &m.pos {
            let extreme = p.x.abs().max(p.y.abs()).max(p.z.abs());
            assert!(
                (extreme - 1.0).abs() < 1e-9,
                "expected a vertex on the unit cube's surface, got {p:?}"
            );
        }
    }

    fn golden(name: &str) -> serde_json::Value {
        let path = format!(
            "{}/testdata/tess_golden/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
        serde_json::from_str(&text).unwrap()
    }

    #[test]
    fn decode_matches_golden_exactly() {
        let g = golden("decode");
        for &n in &[1u32, 2, 3, 4, 7] {
            let rows = g[n.to_string()].as_array().unwrap();
            for vid in 0..3 * n * n {
                let ([a0, a1, a2], corner) = decode(vid, n);
                let row = rows[vid as usize].as_array().unwrap();
                let want = [
                    row[0].as_i64().unwrap() as u32,
                    row[1].as_i64().unwrap() as u32,
                    row[2].as_i64().unwrap() as u32,
                    row[3].as_i64().unwrap() as u32,
                ];
                assert_eq!([a0, a1, a2, corner], want, "N={n} vid={vid}");
            }
        }
    }

    fn check_mips_against_golden(name: &str) {
        let g = golden(&format!("tex_{name}"));
        let mips_json = g["mips"].as_array().unwrap();
        let level0 = mips_json[0].as_array().unwrap();
        let size = level0.len() as u32;
        let h0: Vec<f32> = level0
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
            })
            .collect();

        let mips = build_mips(&h0, size);
        assert_eq!(mips.len(), mips_json.len(), "{name}: mip count");
        for (level, want_rows) in mips_json.iter().enumerate() {
            let want_rows = want_rows.as_array().unwrap();
            let s = want_rows.len();
            for y in 0..s {
                let want_row = want_rows[y].as_array().unwrap();
                for x in 0..s {
                    let want = want_row[x].as_f64().unwrap() as f32;
                    let got = mips[level][y * s + x];
                    assert!(
                        (got - want).abs() < 1e-7,
                        "{name}: mip[{level}][{y}][{x}] = {got} vs golden {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn build_mips_matches_golden_random64() {
        check_mips_against_golden("random64");
    }

    #[test]
    fn build_mips_matches_golden_halfflat64() {
        check_mips_against_golden("halfflat64");
    }
}
