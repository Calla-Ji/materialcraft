// viewport3d.wgsl — the 3D preview's render pipeline (CLAUDE.md §9's "3D
// View": GGX metallic-roughness PBR, a simple procedural sky instead of a
// bundled HDRI, height used as vertex displacement, normal mapping from a
// real per-vertex tangent basis — both on every mesh, not just the Plane).
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
    _pad2: f32,
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
    // implicit-derivative texture sampling, so this needs an explicit LOD.
    //
    // Sampling a higher (coarser, pre-blurred by the mip chain) level
    // rather than mip 0 here is deliberate: a sharp height edge (a hard
    // silhouette, not a gradual ramp) can cross within a single vertex-to-
    // vertex span even at this mesh's highest tessellation setting, so two
    // adjacent vertices land on wildly different heights — every quad along
    // that edge folds into a sharp "shard" instead of following the curve
    // smoothly, since vertex displacement has no sub-vertex resolution to
    // fall back on. Reading a coarser mip spreads that same transition
    // across more of the UV range before it ever reaches the vertex shader,
    // which turns the same sharp edge into a gradual ramp over several
    // vertices instead. This only affects the *live preview's* displaced
    // geometry — the exported Height map and the 2D View both still read
    // mip 0, full sharpness, unaffected by this.
    const DISPLACEMENT_LOD: f32 = 3.0;
    let height = textureSampleLevel(height_tex, tex_sampler, in.uv, DISPLACEMENT_LOD).r;
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

const PI: f32 = 3.14159265358979;

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
    let ambient_diffuse = uniforms.ambient_color.rgb * base_color * (1.0 - metallic) * ao * 0.6;
    let reflection = reflect(-v, n);
    // A single raw environment sample is mirror-sharp regardless of
    // roughness (no mip chain to blur it — see the module doc comment's
    // "known simplification" note), so even a "matte" rough surface showed
    // a crisp reflection of the HDRI's bright softbox. Damp it toward zero
    // as roughness increases: a cheap, non-physical stand-in for prefiltering
    // that keeps smooth/metal surfaces mirror-like while rough ones go flat.
    let specular_strength = pow(1.0 - roughness, 3.0);
    let ambient_specular = sky_color(reflection) * f * ao * specular_strength;

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
