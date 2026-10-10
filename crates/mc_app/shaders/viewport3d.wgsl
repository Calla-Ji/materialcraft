// viewport3d.wgsl — the 3D preview's render pipeline (CLAUDE.md §9's "3D
// View": GGX metallic-roughness PBR, a simple procedural sky instead of a
// bundled HDRI, height used as vertex displacement, normal mapping from a
// real per-vertex tangent basis — both on every mesh, not just the Plane).
//
// Unlike the node shaders in mc_nodes, this is a traditional vertex+fragment
// render pipeline, not a compute shader, and it's the only thing in mc_app
// that reads the graph's own Output textures directly (sampled, not
// textureLoad, since the mesh's UVs don't line up 1:1 with texels).

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
    _pad0: f32,
    _pad1: f32,
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

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Every generated mesh (and any mesh a user imports later) carries a
    // real per-vertex normal, so displacing along it is meaningful
    // everywhere — not just on a flat plane. Vertex shaders can't use
    // implicit-derivative texture sampling, so this needs an explicit LOD.
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

    let color = direct + ambient_diffuse + ambient_specular;

    // Reinhard tonemap + display gamma, since this texture is shown directly
    // to the user rather than fed back into the linear pipeline.
    let tonemapped = color / (color + vec3<f32>(1.0));
    let gamma_corrected = pow(tonemapped, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma_corrected, 1.0);
}
