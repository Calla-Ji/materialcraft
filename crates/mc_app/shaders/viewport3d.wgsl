// SPDX-License-Identifier: MIT OR Apache-2.0
// viewport3d.wgsl — the 3D preview's render pipeline (CLAUDE.md §9's "3D
// View": GGX metallic-roughness PBR, a simple procedural sky instead of a
// bundled HDRI, height used as real per-vertex displacement, and normal
// mapping from a real per-vertex tangent basis — both on every mesh, not
// just the Plane).
//
// Height drives actual vertex displacement again, not a per-pixel parallax
// shift — an earlier version of this file tried parallax occlusion mapping
// instead (to get past vertex displacement's own tessellation ceiling on
// sharp edges), but confirmed by hands-on testing, parallax's own defining
// trade-off — the shaded surface shifts with view angle, since that's
// literally the mechanism it uses to fake depth — reads as the texture
// visibly swimming/crawling as the camera orbits. That's not a tunable bug;
// it's what the technique is. Real vertex displacement doesn't have that
// problem (it's actual geometry, so it looks solid from every angle) — the
// tessellation ceiling that originally motivated moving away from it is
// instead addressed on the mesh-generation side now: `viewport3d.rs`'s
// `adaptive_mesh` module builds a content-aware quadtree mesh that
// concentrates vertex density wherever the Height map actually has
// detail (a text edge, a thread line), instead of spreading a fixed vertex
// budget uniformly over the whole face.
//
// Unlike the node shaders in mc_nodes, this is a traditional vertex+fragment
// render pipeline, not a compute shader, and it's the only thing in mc_app
// that reads the graph's own Output textures directly (sampled, not
// textureLoad, since the mesh's UVs don't line up 1:1 with texels).
//
// Real alpha transparency: the Output: Alpha channel drives the fragment's
// own alpha, and the pipeline blends with it (`viewport3d.rs`'s
// `BlendState::ALPHA_BLENDING`) rather than just overwriting the color
// target — so a low-alpha area genuinely shows the viewport's background
// through it. `viewport3d.rs` picks between two otherwise-identical
// pipelines depending on whether the graph actually has an Alpha node
// wired: with no back-face culling (`cull_mode: None`), several faces of a
// closed mesh can land on the same pixel, and depth *write* is what makes
// that resolve correctly for ordinary opaque rendering — but once alpha
// blending is doing real work, keeping depth write on means whichever
// partially-transparent face is drawn first still blocks anything behind
// it at the depth test, regardless of how transparent it actually was, so
// the background shows through inconsistently depending on draw order
// rather than the real alpha value. Turning depth write off once an Alpha
// node is present fixes that. Known remaining simplification: without
// depth write, two *different* transparent layers of the same mesh aren't
// sorted against each other (only against the opaque background behind
// both), so a concave shape's near and far transparent surfaces may not
// composite in the "correct" back-to-front order — acceptable for a
// material preview, not a full compositor.

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    // Direction the light travels (from the light, toward the scene).
    light_dir: vec4<f32>,
    // Solid-angle-weighted average color of the environment map — used as a
    // flat ambient/irradiance fill instead of a raw per-pixel sample, which
    // would paint a sharp, obviously-not-diffuse image of the sky onto the
    // surface (see `fs_main`'s `ambient_diffuse`).
    ambient_color: vec4<f32>,
    height_scale: f32,
    // > 0.5 bypasses PBR shading entirely and outputs Base Color directly
    // (still sRGB-encoded) — a ground-truth view for comparing against the
    // 2D View and an exported PNG, since none of exposure/tonemapping/
    // lighting can introduce a mismatch when none of it runs.
    shading_mode: f32,
    // Multiplies linear color before tonemapping (lit mode only).
    exposure: f32,
    // Multiplies just the environment's contribution (ambient fill +
    // reflections) — independent of `exposure` above, so the HDRI can be
    // dimmed/brightened without touching direct light.
    environment_intensity: f32,
    // 0 = sharp mirror reflection of the environment; 1 = heavily blurred.
    // See `blurred_sky_color` below.
    environment_blur: f32,
    // Only meaningful for `vs_main_tess` (Cube/Rounded Cube with a Height
    // node connected) — ignored by the plain-vertex-buffer `vs_main` path.
    // Exact integers stored as `f32` and cast back with `u32(...)`/`i32(...)`
    // in-shader: safe up to 2^24, far past any N or texture size this app
    // offers (see `tess_gpu::budget_n`).
    tess_n_global: f32,
    tess_tex_size: f32,
    tess_mip_count: f32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Direction UV's U axis runs across the surface — computed per mesh in
    // `viewport3d.rs`'s `compute_tangents`, from the actual UV layout rather
    // than guessed in the shader (see `fs_main`).
    @location(3) tangent: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var base_color_tex: texture_2d<f32>;
@group(1) @binding(1) var normal_tex: texture_2d<f32>;
@group(1) @binding(2) var roughness_tex: texture_2d<f32>;
@group(1) @binding(3) var metallic_tex: texture_2d<f32>;
@group(1) @binding(4) var ao_tex: texture_2d<f32>;
@group(1) @binding(5) var height_tex: texture_2d<f32>;
@group(1) @binding(6) var tex_sampler: sampler;
@group(1) @binding(7) var environment_tex: texture_2d<f32>;
@group(1) @binding(8) var alpha_tex: texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Every generated mesh (and any mesh a user imports later) carries a
    // real per-vertex normal, so displacing along it is meaningful
    // everywhere — not just on a flat plane. Vertex shaders can't use
    // implicit-derivative texture sampling, so this needs an explicit LOD;
    // mip 0 (full sharpness) is correct here now — `adaptive_mesh`'s own
    // content-aware vertex density is what keeps a hard height edge from
    // faceting, not a blurred sample.
    let height = textureSampleLevel(height_tex, tex_sampler, in.uv, 0.0).r;
    let position = in.position + in.normal * (height - 0.5) * uniforms.height_scale;

    out.world_pos = position;
    // Not re-deriving the normal from the displaced surface (that would need
    // finite differences) — using the flat mesh normal is a simplification,
    // fine for a quick preview.
    out.world_normal = in.normal;
    out.world_tangent = in.tangent;
    out.uv = in.uv;
    out.clip_position = uniforms.view_proj * vec4<f32>(position, 1.0);
    return out;
}

// ---------------------------------------------------------------------------
// vs_main_tess: GPU vertex-pulling tessellation + displacement, used instead
// of vs_main for Cube/Rounded Cube when a Height node is connected —
// replaces `mc_app::adaptive_mesh`'s CPU-built quadtree mesh (see that
// module's own doc-comment history: a CPU mesh whose size scaled with the
// Tessellation slider hit a hard wgpu per-buffer size ceiling, and
// rebuilding/re-uploading it on every Tessellation or Resolution change
// froze the UI thread long enough to cause real, confusing symptoms). This
// is a direct, mechanical port of an independently-authored reference
// (`crates/mc_app/src/tess_mesh.rs`'s own module doc comment has the full
// citation and golden-test validation story — not derived from, or a
// reimplementation of, any commercial tool's own code).
//
// One draw per frame, no vertex buffer at all:
// `draw(vertex_count = 3*N*N, instance_count = base_triangle_count)`. This
// vertex shader decodes `(instance_index, vertex_index)` into a lattice
// point of its base triangle and evaluates position/normal/uv/tangent
// procedurally from the storage buffers below (built once per base mesh,
// and again only when the Height content actually changes — the same
// `current_mesh_key` dirty-check the old adaptive mesh already used).
//
// Uniform tessellation only for now (no per-triangle adaptive level): every
// base triangle subdivides into the same N x N lattice, N =
// `uniforms.tess_n_global` — already reduced on the Rust side
// (`tess_gpu::budget_n`) so a large Tessellation value on a mesh with many
// base triangles can't ask for more sub-triangles than stays renderable.
// Seam handling (so two adjacent base triangles evaluate a shared edge to
// bit-identical-enough positions — see `tess_mesh::prepare`'s own doc
// comment) is always on; there's no naive/debug toggle in the real
// renderer, unlike the reference pack's own test harness.

struct TessVertex {
    pos: vec4<f32>,
    ndisp: vec4<f32>,
    tan_u: vec4<f32>,
    tan_v: vec4<f32>,
    cuv01: vec4<f32>,
    cuv23: vec4<f32>,
    info: vec4<u32>,
}

struct TessTri {
    v: vec4<u32>,
    en: vec4<u32>,
    tw0: vec4<f32>,
    tw1: vec4<f32>,
    tw2: vec4<f32>,
}

@group(2) @binding(0) var<storage, read> tess_verts: array<TessVertex>;
@group(2) @binding(1) var<storage, read> tess_tris: array<TessTri>;
@group(2) @binding(2) var<storage, read> tess_inst_tri: array<u32>;

fn tess_bilinear(level: i32, uv: vec2<f32>) -> f32 {
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
    let h00 = textureLoad(height_tex, vec2<i32>(x0, y0), level).r;
    let h10 = textureLoad(height_tex, vec2<i32>(x1, y0), level).r;
    let h01 = textureLoad(height_tex, vec2<i32>(x0, y1), level).r;
    let h11 = textureLoad(height_tex, vec2<i32>(x1, y1), level).r;
    let a = h00 * (1.0 - fx) + h10 * fx;
    let b = h01 * (1.0 - fx) + h11 * fx;
    return a * (1.0 - fy) + b * fy;
}

fn tess_sample_height(uv: vec2<f32>, lod_in: f32) -> f32 {
    let lod = clamp(lod_in, 0.0, uniforms.tess_mip_count - 1.0);
    let l0 = i32(floor(lod));
    let f = lod - f32(l0);
    let h0 = tess_bilinear(l0, uv);
    if f <= 0.0 || f32(l0 + 1) >= uniforms.tess_mip_count {
        return h0;
    }
    return h0 * (1.0 - f) + tess_bilinear(l0 + 1, uv) * f;
}

struct TessLattice { a: vec3<u32>, corner: u32 }

fn tess_decode(vid: u32, n: u32) -> TessLattice {
    let t = vid / 3u;
    let c = vid % 3u;
    var r = u32(sqrt(f32(t)));
    if (r + 1u) * (r + 1u) <= t { r = r + 1u; }
    if r * r > t { r = r - 1u; }
    let k = t - r * r;
    var i: u32;
    var j: u32;
    if (k & 1u) == 0u {
        let a = k / 2u;
        if c == 0u { i = r; j = a; } else if c == 1u { i = r + 1u; j = a; } else { i = r + 1u; j = a + 1u; }
    } else {
        let a = (k - 1u) / 2u;
        if c == 0u { i = r; j = a; } else if c == 1u { i = r + 1u; j = a + 1u; } else { i = r; j = a + 1u; }
    }
    return TessLattice(vec3<u32>(n - i, i - j, j), c);
}

struct TessEval { pos: vec3<f32>, nd: vec3<f32>, uv: vec2<f32>, tu: vec3<f32>, tv: vec3<f32> }

fn tess_vuv(idx: u32) -> vec2<f32> { return vec2<f32>(tess_verts[idx].tan_u.w, tess_verts[idx].tan_v.w); }

fn tess_mix1(a: f32, b: f32, t: f32) -> f32 { return a + (b - a) * t; }
fn tess_mix2(a: vec2<f32>, b: vec2<f32>, t: f32) -> vec2<f32> { return a + (b - a) * t; }
fn tess_mix3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> { return a + (b - a) * t; }

fn tess_corner_uv(idx: u32, q: u32) -> vec2<f32> {
    let v = tess_verts[idx];
    if q == 0u { return v.cuv01.xy; }
    if q == 1u { return v.cuv01.zw; }
    if q == 2u { return v.cuv23.xy; }
    return v.cuv23.zw;
}

fn tess_finish(base: vec3<f32>, nd_raw: vec3<f32>, uv: vec2<f32>, tu: vec3<f32>, tv: vec3<f32>, h: f32) -> TessEval {
    let nd = normalize(nd_raw);
    // height_level is a fixed 0.5 ("mid-gray is neutral"), matching
    // `vs_main`'s own existing displacement formula above exactly — the app
    // has never exposed this as its own artist-facing control.
    let p = base + nd * (h - 0.5) * uniforms.height_scale;
    return TessEval(p, nd, uv, tu, tv);
}

// Base vertex: lod uses the GLOBAL N so every triangle around the vertex
// agrees on this vertex's own height, whatever N each of them individually
// subdivides into.
fn tess_eval_corner(idx: u32) -> TessEval {
    let v = tess_verts[idx];
    let lod = v.pos.w - log2(uniforms.tess_n_global);
    var h: f32;
    if v.info.x > 1u {
        var acc = 0.0;
        for (var q = 0u; q < v.info.x; q = q + 1u) {
            acc = acc + tess_sample_height(tess_corner_uv(idx, q), lod);
        }
        h = acc / f32(v.info.x);
    } else {
        h = tess_sample_height(tess_vuv(idx), lod);
    }
    return tess_finish(v.pos.xyz, v.ndisp.xyz, tess_vuv(idx), v.tan_u.xyz, v.tan_v.xyz, h);
}

fn tess_twin_of(tri: TessTri, e: u32) -> vec4<f32> {
    if e == 0u { return tri.tw0; }
    if e == 1u { return tri.tw1; }
    return tri.tw2;
}

// Point k of n along edge e (A -> B): canonical order by weld id, so both
// triangles sharing this edge evaluate it the same way regardless of which
// one's own winding visits it first.
fn tess_eval_edge(tri: TessTri, e: u32, a_in: u32, b_in: u32, k_in: u32, n: u32, ne: u32) -> TessEval {
    var a = a_in;
    var b = b_in;
    var k = k_in;
    var tw = tess_twin_of(tri, e);
    let seam = ((tri.en.w >> e) & 1u) == 1u;
    if tess_verts[a].info.y > tess_verts[b].info.y {
        a = b_in;
        b = a_in;
        k = n - k_in;
        tw = vec4<f32>(tw.zw, tw.xy);
    }
    if ne < n {
        let step = n / ne;
        k = (k / step) * step;
    }
    if k == 0u { return tess_eval_corner(a); }
    if k == n { return tess_eval_corner(b); }
    let t = f32(k) / f32(n);
    let va = tess_verts[a];
    let vb = tess_verts[b];
    let uv = tess_mix2(tess_vuv(a), tess_vuv(b), t);
    let lod = tess_mix1(va.pos.w, vb.pos.w, t) - log2(f32(ne));
    var h = tess_sample_height(uv, lod);
    if seam {
        let tuv = tess_mix2(tw.xy, tw.zw, t);
        h = 0.5 * (h + tess_sample_height(tuv, lod));
    }
    return tess_finish(
        tess_mix3(va.pos.xyz, vb.pos.xyz, t),
        tess_mix3(va.ndisp.xyz, vb.ndisp.xyz, t),
        uv,
        tess_mix3(va.tan_u.xyz, vb.tan_u.xyz, t),
        tess_mix3(va.tan_v.xyz, vb.tan_v.xyz, t),
        h,
    );
}

fn tess_eval_point(tri: TessTri, a: vec3<u32>, n: u32) -> TessEval {
    let zeros = u32(a.x == 0u) + u32(a.y == 0u) + u32(a.z == 0u);
    if zeros == 2u {
        if a.x == n { return tess_eval_corner(tri.v.x); }
        if a.y == n { return tess_eval_corner(tri.v.y); }
        return tess_eval_corner(tri.v.z);
    }
    if zeros == 1u {
        if a.z == 0u { return tess_eval_edge(tri, 0u, tri.v.x, tri.v.y, a.y, n, tri.en.x); }
        if a.x == 0u { return tess_eval_edge(tri, 1u, tri.v.y, tri.v.z, a.z, n, tri.en.y); }
        return tess_eval_edge(tri, 2u, tri.v.z, tri.v.x, a.x, n, tri.en.z);
    }
    let w = vec3<f32>(a) / f32(n);
    let v0 = tess_verts[tri.v.x];
    let v1 = tess_verts[tri.v.y];
    let v2 = tess_verts[tri.v.z];
    let base = v0.pos.xyz * w.x + v1.pos.xyz * w.y + v2.pos.xyz * w.z;
    let uv = tess_vuv(tri.v.x) * w.x + tess_vuv(tri.v.y) * w.y + tess_vuv(tri.v.z) * w.z;
    let nd = v0.ndisp.xyz * w.x + v1.ndisp.xyz * w.y + v2.ndisp.xyz * w.z;
    let lam = v0.pos.w * w.x + v1.pos.w * w.y + v2.pos.w * w.z;
    let lod = lam - log2(f32(n));
    let h = tess_sample_height(uv, lod);
    let tu = v0.tan_u.xyz * w.x + v1.tan_u.xyz * w.y + v2.tan_u.xyz * w.z;
    let tv = v0.tan_v.xyz * w.x + v1.tan_v.xyz * w.y + v2.tan_v.xyz * w.z;
    return tess_finish(base, nd, uv, tu, tv, h);
}

@vertex
fn vs_main_tess(@builtin(vertex_index) vid: u32, @builtin(instance_index) inst: u32) -> VertexOutput {
    let tri = tess_tris[tess_inst_tri[inst]];
    let n = u32(uniforms.tess_n_global);
    let lattice = tess_decode(vid, n);
    let ev = tess_eval_point(tri, lattice.a, n);
    var out: VertexOutput;
    out.world_pos = ev.pos;
    out.world_normal = ev.nd;
    out.world_tangent = ev.tu;
    out.uv = ev.uv;
    out.clip_position = uniforms.view_proj * vec4<f32>(ev.pos, 1.0);
    return out;
}

const PI: f32 = 3.14159265358979;
const BLUR_TAP_COUNT: i32 = 8;

// Standard equirectangular (latitude/longitude) mapping: u wraps around the
// horizon, v runs from the +Y pole (0) to the -Y pole (1).
fn equirect_uv(direction: vec3<f32>) -> vec2<f32> {
    let d = normalize(direction);
    let u = atan2(d.z, d.x) / (2.0 * PI) + 0.5;
    let v = 0.5 - asin(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2<f32>(u, v);
}

fn sky_color(direction: vec3<f32>) -> vec3<f32> {
    return textureSample(environment_tex, tex_sampler, equirect_uv(direction)).rgb;
}

// `sky_color`, angularly softened by `blur` (0 = untouched, exact mirror; 1 =
// heavily blurred) — an artist-facing alternative to the environment just
// mirroring back exactly, for shiny/metallic materials where even a rough-
// surface-dampened reflection (see `fs_main`'s `specular_strength`) still
// reads as an obvious copy of the HDRI. A single 8-tap ring around
// `direction`, in a tangent basis built from it, rather than a real
// prefiltered mip chain — cheap, and this only ever runs once per shaded
// pixel (there's no separate skybox pass to also blur; see this file's
// module doc comment on the background being a flat clear color).
fn blurred_sky_color(direction: vec3<f32>, blur: f32) -> vec3<f32> {
    if blur <= 0.001 {
        return sky_color(direction);
    }
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if abs(direction.y) > 0.99 {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, direction));
    let bitangent = cross(direction, tangent);
    let radius = blur * 0.8;

    var sum = sky_color(direction);
    for (var i = 0; i < BLUR_TAP_COUNT; i = i + 1) {
        let angle = f32(i) / f32(BLUR_TAP_COUNT) * 2.0 * PI;
        let offset = (tangent * cos(angle) + bitangent * sin(angle)) * radius;
        sum = sum + sky_color(normalize(direction + offset));
    }
    return sum / f32(BLUR_TAP_COUNT + 1);
}

// The exact piecewise sRGB OETF (same curve `mc_export::linear_to_srgb_u8`
// and `display_encode.wgsl` use) rather than an approximate `pow(x, 1/2.2)`
// — this view is compared pixel-for-pixel against the 2D View and exported
// PNGs (Unlit Base Color mode), so it needs the same curve they use, not a
// close-enough one.
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let low = c * 12.92;
    let high = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, c <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let geometric_normal = normalize(in.world_normal);
    // Re-orthogonalize per pixel (Gram-Schmidt): interpolating the tangent
    // and normal separately across a triangle doesn't keep them exactly
    // perpendicular.
    let tangent = normalize(
        in.world_tangent - geometric_normal * dot(geometric_normal, in.world_tangent)
    );
    let bitangent = cross(geometric_normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, geometric_normal);
    let sampled_normal = textureSample(normal_tex, tex_sampler, in.uv).xyz * 2.0 - 1.0;
    let n = normalize(tbn * sampled_normal);

    let v = normalize(uniforms.camera_pos.xyz - in.world_pos);
    let l = normalize(-uniforms.light_dir.xyz);
    let h = normalize(v + l);

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0001);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    let base_color = textureSample(base_color_tex, tex_sampler, in.uv).rgb;
    // Named `surface_alpha`, not `alpha`, to avoid colliding with the GGX
    // roughness-squared `alpha` below — an unrelated quantity that happens
    // to share the conventional name in PBR shader code.
    let surface_alpha = textureSample(alpha_tex, tex_sampler, in.uv).r;

    if uniforms.shading_mode > 0.5 {
        return vec4<f32>(linear_to_srgb(base_color), surface_alpha);
    }

    let roughness = clamp(textureSample(roughness_tex, tex_sampler, in.uv).r, 0.04, 1.0);
    let metallic = textureSample(metallic_tex, tex_sampler, in.uv).r;
    let ao = textureSample(ao_tex, tex_sampler, in.uv).r;

    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;

    // GGX normal distribution.
    let denom = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    let d = alpha2 / (PI * denom * denom);

    // Smith geometry term (Schlick-GGX approximation).
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let g_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    let g = g_v * g_l;

    // Fresnel-Schlick.
    let f = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - v_dot_h, 5.0);

    let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * base_color / PI;

    let light_color = vec3<f32>(3.0, 2.9, 2.7);
    let direct = (diffuse + specular) * light_color * n_dot_l;

    // Environment ambient: a flat diffuse fill using the environment's
    // average color (a per-pixel raw sample here would paint a sharp image
    // of the sky onto every surface regardless of roughness — not what
    // diffuse/irradiance lighting looks like), plus a specular reflection of
    // the sky dampened by roughness below.
    let ambient_diffuse = uniforms.ambient_color.rgb * base_color * (1.0 - metallic) * ao * 0.6
        * uniforms.environment_intensity;
    let reflection = reflect(-v, n);
    // Roughness still dampens reflection *strength* toward zero (a cheap,
    // non-physical stand-in for prefiltering — see `blurred_sky_color`'s doc
    // comment above for why that alone isn't enough at low roughness), and
    // `environment_blur` separately softens the reflection's *content*
    // regardless of roughness.
    let specular_strength = pow(1.0 - roughness, 3.0);
    let ambient_specular = blurred_sky_color(reflection, uniforms.environment_blur)
        * f * ao * specular_strength * uniforms.environment_intensity;

    let color = (direct + ambient_diffuse + ambient_specular) * uniforms.exposure;

    // Reinhard tonemap on *luminance only*, not per-channel: compressing
    // each of R/G/B independently (the common naive version) distorts hue
    // and desaturates anything bright or saturated, since a channel's own
    // ratio to the others shifts as it individually approaches 1 — compared
    // directly against the 2D View's untonemapped, unlit color, a Base
    // Color this washes out reads as "duller/muddier than it should be."
    // Tonemapping luminance and rescaling the original color by the same
    // factor compresses brightness without touching the color's own ratios
    // at all.
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let tonemapped_luminance = luminance / (luminance + 1.0);
    let luminance_scale = select(0.0, tonemapped_luminance / luminance, luminance > 0.0001);
    let tonemapped = color * luminance_scale;
    let gamma_corrected = linear_to_srgb(tonemapped);

    return vec4<f32>(gamma_corrected, surface_alpha);
}
