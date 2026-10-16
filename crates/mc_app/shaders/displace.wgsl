// displace.wgsl - MaterialCreator viewport: vertex-shader tessellation + displacement.
// Port of viewport_reference.py (the Python file is the ground truth).
// Entry points: vs_main / fs_main (render), cs_eval / cs_normal (tests only).
// Clean room: no Adobe shaders, no GPL code.

struct Uniforms {
    view_proj : mat4x4f,
    cam_pos   : vec4f,
    light_dir : vec4f,     // xyz = direction TOWARD the light (normalised)
    disp      : vec4f,     // x height_scale, y height_level, z tex_size (mip 0), w mip_count
    mode      : vec4u,     // x N (uniform: N; adaptive: max N), y adaptive 0/1, z seam_fix 0/1,
                           // w debug bits: 1 = wireframe, 2 = faceted geometric normal (no per-pixel normal)
    color_lo  : vec4f,     // albedo at height 0 (preview colouring only)
    color_hi  : vec4f,     // albedo at height 1
};

struct BaseVertex {
    pos   : vec4f,         // xyz position (welded duplicates bitwise identical), w = lam
    ndisp : vec4f,         // xyz displacement normal (welded average), w unused
    tan_u : vec4f,         // xyz dP/du, w = uv.x
    tan_v : vec4f,         // xyz dP/dv, w = uv.y
    cuv01 : vec4f,         // seam-corner uv set (sorted), entries 0 and 1
    cuv23 : vec4f,         // entries 2 and 3
    info  : vec4u,         // x = uv-set count (1 = not a seam corner), y = weld id
};

struct BaseTri {
    v   : vec4u,           // xyz vertex indices, w = N of this triangle (adaptive mode)
    en  : vec4u,           // xyz effective N per edge (adaptive), w = seam bits (bit e = edge e has a twin)
    tw0 : vec4f,           // twin uv of edge 0 (v0 -> v1): xy at v0, zw at v1
    tw1 : vec4f,           // edge 1 (v1 -> v2)
    tw2 : vec4f,           // edge 2 (v2 -> v0)
};

@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var<storage, read> verts : array<BaseVertex>;
@group(0) @binding(2) var<storage, read> tris : array<BaseTri>;
@group(0) @binding(3) var<storage, read> inst_tri : array<u32>;   // instance -> base triangle
@group(0) @binding(4) var height_tex : texture_2d<f32>;            // r32float, full mip chain (2x2 box)

// ---------------------------------------------------------------- height sampling

fn bilinear(level : i32, uv : vec2f) -> f32 {
    let s = i32(textureDimensions(height_tex, level).x);
    let sf = f32(s);
    let x = uv.x * sf - 0.5;
    let y = uv.y * sf - 0.5;
    let xf = floor(x);
    let yf = floor(y);
    let fx = x - xf;
    let fy = y - yf;
    let x0 = ((i32(xf) % s) + s) % s;
    let y0 = ((i32(yf) % s) + s) % s;
    let x1 = (x0 + 1) % s;
    let y1 = (y0 + 1) % s;
    let h00 = textureLoad(height_tex, vec2i(x0, y0), level).r;
    let h10 = textureLoad(height_tex, vec2i(x1, y0), level).r;
    let h01 = textureLoad(height_tex, vec2i(x0, y1), level).r;
    let h11 = textureLoad(height_tex, vec2i(x1, y1), level).r;
    let a = h00 * (1.0 - fx) + h10 * fx;
    let b = h01 * (1.0 - fx) + h11 * fx;
    return a * (1.0 - fy) + b * fy;
}

fn sample_height(uv : vec2f, lod_in : f32) -> f32 {
    let lod = clamp(lod_in, 0.0, U.disp.w - 1.0);
    let l0 = i32(floor(lod));
    let f = lod - f32(l0);
    let h0 = bilinear(l0, uv);
    if (f <= 0.0 || f32(l0 + 1) >= U.disp.w) {
        return h0;
    }
    return h0 * (1.0 - f) + bilinear(l0 + 1, uv) * f;
}

// ---------------------------------------------------------------- lattice decode

struct Lattice { a : vec3u, corner : u32 };

fn decode(vid : u32, N : u32) -> Lattice {
    let t = vid / 3u;
    let c = vid % 3u;
    var r = u32(sqrt(f32(t)));
    if ((r + 1u) * (r + 1u) <= t) { r = r + 1u; }      // exact integer sqrt
    if (r * r > t) { r = r - 1u; }
    let k = t - r * r;
    var i : u32;
    var j : u32;
    if ((k & 1u) == 0u) {
        let a = k / 2u;                                  // up: (r,a) (r+1,a) (r+1,a+1)
        if (c == 0u) { i = r; j = a; } else if (c == 1u) { i = r + 1u; j = a; } else { i = r + 1u; j = a + 1u; }
    } else {
        let a = (k - 1u) / 2u;                           // down: (r,a) (r+1,a+1) (r,a+1)
        if (c == 0u) { i = r; j = a; } else if (c == 1u) { i = r + 1u; j = a + 1u; } else { i = r; j = a + 1u; }
    }
    return Lattice(vec3u(N - i, i - j, j), c);
}

// ---------------------------------------------------------------- point evaluation

struct Eval {
    pos : vec3f, nd : vec3f, uv : vec2f, tu : vec3f, tv : vec3f, h : f32, lod : f32,
};

fn vuv(idx : u32) -> vec2f { return vec2f(verts[idx].tan_u.w, verts[idx].tan_v.w); }

fn mix3(a : vec3f, b : vec3f, t : f32) -> vec3f { return a + (b - a) * t; }
fn mix2(a : vec2f, b : vec2f, t : f32) -> vec2f { return a + (b - a) * t; }
fn mix1(a : f32, b : f32, t : f32) -> f32 { return a + (b - a) * t; }

fn corner_uv(idx : u32, q : u32) -> vec2f {
    let v = verts[idx];
    if (q == 0u) { return v.cuv01.xy; }
    if (q == 1u) { return v.cuv01.zw; }
    if (q == 2u) { return v.cuv23.xy; }
    return v.cuv23.zw;
}

fn finish(base : vec3f, nd_raw : vec3f, uv : vec2f, tu : vec3f, tv : vec3f, h : f32, lod : f32) -> Eval {
    let nd = normalize(nd_raw);
    let p = base + nd * (h - U.disp.y) * U.disp.x;
    return Eval(p, nd, uv, tu, tv, h, lod);
}

// base vertex: lod uses the GLOBAL N (U.mode.x) so every triangle around the vertex agrees
fn eval_corner(idx : u32) -> Eval {
    let v = verts[idx];
    let lod = v.pos.w - log2(f32(U.mode.x));
    var h : f32;
    if (U.mode.z == 1u && v.info.x > 1u) {
        var acc = 0.0;
        for (var q = 0u; q < v.info.x; q = q + 1u) {
            acc = acc + sample_height(corner_uv(idx, q), lod);
        }
        h = acc / f32(v.info.x);
    } else {
        h = sample_height(vuv(idx), lod);
    }
    return finish(v.pos.xyz, v.ndisp.xyz, vuv(idx), v.tan_u.xyz, v.tan_v.xyz, h, lod);
}

fn twin_of(tri : BaseTri, e : u32) -> vec4f {
    if (e == 0u) { return tri.tw0; }
    if (e == 1u) { return tri.tw1; }
    return tri.tw2;
}

// point k of N along edge e (A -> B); canonical order by weld id, adaptive snapping to Ne
fn eval_edge(tri : BaseTri, e : u32, A_in : u32, B_in : u32, k_in : u32, N : u32, Ne : u32) -> Eval {
    var A = A_in;
    var B = B_in;
    var k = k_in;
    var tw = twin_of(tri, e);
    let seam = U.mode.z == 1u && ((tri.en.w >> e) & 1u) == 1u;
    if (verts[A].info.y > verts[B].info.y) {
        A = B_in; B = A_in; k = N - k_in;
        tw = vec4f(tw.zw, tw.xy);
    }
    if (Ne < N) {
        let step = N / Ne;
        k = (k / step) * step;
    }
    if (k == 0u) { return eval_corner(A); }
    if (k == N) { return eval_corner(B); }
    let t = f32(k) / f32(N);
    let va = verts[A];
    let vb = verts[B];
    let uv = mix2(vuv(A), vuv(B), t);
    let lod = mix1(va.pos.w, vb.pos.w, t) - log2(f32(Ne));
    var h = sample_height(uv, lod);
    if (seam) {
        let tuv = mix2(tw.xy, tw.zw, t);
        h = 0.5 * (h + sample_height(tuv, lod));
    }
    return finish(mix3(va.pos.xyz, vb.pos.xyz, t), mix3(va.ndisp.xyz, vb.ndisp.xyz, t), uv,
                  mix3(va.tan_u.xyz, vb.tan_u.xyz, t), mix3(va.tan_v.xyz, vb.tan_v.xyz, t), h, lod);
}

fn eval_point(tri : BaseTri, a : vec3u, N : u32, eN : vec3u) -> Eval {
    let z = u32(a.x == 0u) + u32(a.y == 0u) + u32(a.z == 0u);
    if (z == 2u) {
        if (a.x == N) { return eval_corner(tri.v.x); }
        if (a.y == N) { return eval_corner(tri.v.y); }
        return eval_corner(tri.v.z);
    }
    if (z == 1u) {
        if (a.z == 0u) { return eval_edge(tri, 0u, tri.v.x, tri.v.y, a.y, N, eN.x); }
        if (a.x == 0u) { return eval_edge(tri, 1u, tri.v.y, tri.v.z, a.z, N, eN.y); }
        return eval_edge(tri, 2u, tri.v.z, tri.v.x, a.x, N, eN.z);
    }
    let w = vec3f(a) / f32(N);
    let v0 = verts[tri.v.x];
    let v1 = verts[tri.v.y];
    let v2 = verts[tri.v.z];
    let base = v0.pos.xyz * w.x + v1.pos.xyz * w.y + v2.pos.xyz * w.z;
    let uv = vuv(tri.v.x) * w.x + vuv(tri.v.y) * w.y + vuv(tri.v.z) * w.z;
    let nd = v0.ndisp.xyz * w.x + v1.ndisp.xyz * w.y + v2.ndisp.xyz * w.z;
    let lam = v0.pos.w * w.x + v1.pos.w * w.y + v2.pos.w * w.z;
    let lod = lam - log2(f32(N));
    let h = sample_height(uv, lod);
    let tu = v0.tan_u.xyz * w.x + v1.tan_u.xyz * w.y + v2.tan_u.xyz * w.z;
    let tv = v0.tan_v.xyz * w.x + v1.tan_v.xyz * w.y + v2.tan_v.xyz * w.z;
    return finish(base, nd, uv, tu, tv, h, lod);
}

fn tri_N(tri : BaseTri) -> u32 { return select(U.mode.x, tri.v.w, U.mode.y == 1u); }
fn tri_eN(tri : BaseTri, N : u32) -> vec3u { return select(vec3u(N), tri.en.xyz, U.mode.y == 1u); }

// ---------------------------------------------------------------- per-pixel normal

fn shading_normal(uv : vec2f, nd_in : vec3f, tu : vec3f, tv : vec3f, lod : f32) -> vec3f {
    let d = exp2(lod) / U.disp.z;
    let hu = (sample_height(uv + vec2f(d, 0.0), lod) - sample_height(uv - vec2f(d, 0.0), lod)) / (2.0 * d);
    let hv = (sample_height(uv + vec2f(0.0, d), lod) - sample_height(uv - vec2f(0.0, d), lod)) / (2.0 * d);
    let n = normalize(nd_in);
    let su = tu + n * U.disp.x * hu;
    let sv = tv + n * U.disp.x * hv;
    let c = normalize(cross(su, sv));
    return select(-c, c, dot(c, n) >= 0.0);
}

// ---------------------------------------------------------------- render

struct VSOut {
    @builtin(position) clip : vec4f,
    @location(0) world : vec3f,
    @location(1) nd : vec3f,
    @location(2) tu : vec3f,
    @location(3) tv : vec3f,
    @location(4) uv : vec2f,
    @location(5) bary : vec3f,
};

@vertex
fn vs_main(@builtin(vertex_index) vid : u32, @builtin(instance_index) inst : u32) -> VSOut {
    let tri = tris[inst_tri[inst]];
    let N = tri_N(tri);
    let L = decode(vid, N);
    let ev = eval_point(tri, L.a, N, tri_eN(tri, N));
    var o : VSOut;
    o.clip = U.view_proj * vec4f(ev.pos, 1.0);
    o.world = ev.pos;
    o.nd = ev.nd;
    o.tu = ev.tu;
    o.tv = ev.tv;
    o.uv = ev.uv;
    o.bary = vec3f(f32(L.corner == 0u), f32(L.corner == 1u), f32(L.corner == 2u));
    return o;
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4f {
    // shading mip from screen-space uv derivatives (computed before any branch)
    let tex_uv = in.uv * U.disp.z;
    let lod = max(0.0, log2(max(length(dpdx(tex_uv)), length(dpdy(tex_uv)))));
    let wire_w = fwidth(in.bary);
    let facet = normalize(cross(dpdy(in.world), dpdx(in.world)));
    var n = shading_normal(in.uv, in.nd, in.tu, in.tv, lod);
    if ((U.mode.w & 2u) != 0u) {
        n = select(-facet, facet, dot(facet, in.nd) >= 0.0);
    }
    let h = sample_height(in.uv, lod);
    let albedo = mix(U.color_lo.rgb, U.color_hi.rgb, clamp(h, 0.0, 1.0));
    let l = normalize(U.light_dir.xyz);
    let v = normalize(U.cam_pos.xyz - in.world);
    let hv = normalize(l + v);
    let diff = max(dot(n, l), 0.0);
    let spec = pow(max(dot(n, hv), 0.0), 48.0) * 0.25 * f32(diff > 0.0);
    let amb = mix(vec3f(0.10, 0.09, 0.08), vec3f(0.22, 0.25, 0.30), n.y * 0.5 + 0.5);
    var col = albedo * (amb + vec3f(1.0, 0.96, 0.9) * diff) + vec3f(spec);
    if ((U.mode.w & 1u) != 0u) {
        let e = min(min(in.bary.x / max(wire_w.x, 1e-6), in.bary.y / max(wire_w.y, 1e-6)), in.bary.z / max(wire_w.z, 1e-6));
        col = mix(vec3f(0.05, 0.35, 0.95), col, smoothstep(0.0, 1.0, e));
    }
    col = col * 0.8;                                    // exposure (preview only)
    return vec4f(pow(clamp(col, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2)), 1.0);
}

// ---------------------------------------------------------------- tests (gpu_check.py)

struct Query { tri : u32, vid : u32, pad0 : u32, pad1 : u32 };
@group(0) @binding(5) var<storage, read> queries : array<Query>;
@group(0) @binding(6) var<storage, read_write> results : array<vec4f>;   // 2 per query: pos.xyz,h  uv.xy,lod,kind

@compute @workgroup_size(64)
fn cs_eval(@builtin(global_invocation_id) g : vec3u) {
    if (g.x >= arrayLength(&queries)) { return; }
    let q = queries[g.x];
    let tri = tris[q.tri];
    let N = tri_N(tri);
    let L = decode(q.vid, N);
    let ev = eval_point(tri, L.a, N, tri_eN(tri, N));
    results[2u * g.x] = vec4f(ev.pos, ev.h);
    results[2u * g.x + 1u] = vec4f(ev.uv, ev.lod, f32(L.a.x) + 100.0 * f32(L.a.y) + 10000.0 * f32(L.a.z));
}

struct NQuery { uv_lod : vec4f, nd : vec4f, tu : vec4f, tv : vec4f };
@group(0) @binding(7) var<storage, read> nqueries : array<NQuery>;

@compute @workgroup_size(64)
fn cs_normal(@builtin(global_invocation_id) g : vec3u) {
    if (g.x >= arrayLength(&nqueries)) { return; }
    let q = nqueries[g.x];
    results[g.x] = vec4f(shading_normal(q.uv_lod.xy, q.nd.xyz, q.tu.xyz, q.tv.xyz, q.uv_lod.z), 0.0);
}
