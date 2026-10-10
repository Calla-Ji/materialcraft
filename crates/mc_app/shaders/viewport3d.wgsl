// viewport3d.wgsl — the 3D preview's render pipeline (CLAUDE.md §9's "3D
// View": GGX metallic-roughness PBR, a simple procedural sky instead of a
// bundled HDRI, height used as vertex displacement on the Plane mesh).
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
    height_scale: f32,
    // 1 for the Plane mesh (the only one with a meaningful "up" to displace
    // along), 0 otherwise.
    use_displacement: u32,
    _pad0: f32,
    _pad1: f32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
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

    var position = in.position;
    if uniforms.use_displacement != 0u {
        // Vertex shaders can't use implicit-derivative texture sampling, so
        // this needs an explicit LOD.
        let height = textureSampleLevel(height_tex, tex_sampler, in.uv, 0.0).r;
        position = position + in.normal * (height - 0.5) * uniforms.height_scale;
    }

    out.world_pos = position;
    // Not re-deriving the normal from the displaced surface (that would need
    // finite differences or precomputed tangents) — using the flat mesh
    // normal is a simplification, fine for a quick preview.
    out.world_normal = in.normal;
    out.uv = in.uv;
    out.clip_position = uniforms.view_proj * vec4<f32>(position, 1.0);
    return out;
}

const PI: f32 = 3.14159265358979;

// Builds an orthonormal (tangent, bitangent, normal) basis from just a
// normal — no precomputed per-vertex tangents needed. Standard branchless
// construction (Duff et al., "Building an Orthonormal Basis, Revisited").
// Used to fake tangent-space normal mapping on meshes that don't carry real
// tangents (good enough for a procedural preview, not a physically exact
// tangent space).
fn make_basis(n: vec3<f32>) -> mat3x3<f32> {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let tangent = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bitangent = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    return mat3x3<f32>(tangent, bitangent, n);
}

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
    let tbn = make_basis(geometric_normal);
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

    // Procedural-sky ambient: a diffuse term from the sky in the normal's
    // direction, plus a cheap (un-blurred, roughness-agnostic) specular
    // reflection of the sky — not real image-based lighting, but gives the
    // preview some non-black fill light and a sense of a reflective
    // environment without needing a bundled HDRI.
    let ambient_diffuse = sky_color(n) * base_color * (1.0 - metallic) * ao * 0.6;
    let reflection = reflect(-v, n);
    let ambient_specular = sky_color(reflection) * f * ao;

    let color = direct + ambient_diffuse + ambient_specular;

    // Reinhard tonemap + display gamma, since this texture is shown directly
    // to the user rather than fed back into the linear pipeline.
    let tonemapped = color / (color + vec3<f32>(1.0));
    let gamma_corrected = pow(tonemapped, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma_corrected, 1.0);
}
