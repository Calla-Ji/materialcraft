// SPDX-License-Identifier: MIT OR Apache-2.0
//! The 3D preview (CLAUDE.md §9's "3D View"): a render pipeline (as opposed
//! to the compute pipelines everywhere else in the app) that draws a sphere,
//! cube, or plane lit by one directional light plus a procedural sky,
//! textured from the graph's own Output nodes — falling back to flat
//! defaults for any channel the graph doesn't produce.
//!
//! Blender/Unreal analogy: this is a tiny built-in viewport renderer, the
//! same idea as Blender's Material Preview shading mode or Unreal's material
//! preview sphere, just scoped to exactly what CLAUDE.md asks for.

use std::collections::HashMap;

use glam::{Mat4, Vec3};

const VERTEX_ATTRS: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x3];

struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    /// Direction of increasing U across the surface — filled in by
    /// `compute_tangents` after a mesh's positions/normals/UVs are built, not
    /// set by hand per-generator. Lets the shader build a real tangent-space
    /// basis for normal mapping instead of faking one from the normal alone,
    /// which only looked right on the Plane (see `compute_tangents`'s doc
    /// comment) by coincidence.
    tangent: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeshKind {
    Sphere,
    Cube,
    Plane,
    RoundedCube,
    RoundedCylinder,
}

impl MeshKind {
    pub const ALL: [MeshKind; 5] = [
        MeshKind::Sphere,
        MeshKind::Cube,
        MeshKind::Plane,
        MeshKind::RoundedCube,
        MeshKind::RoundedCylinder,
    ];

    pub fn label(self) -> &'static str {
        match self {
            MeshKind::Sphere => "Sphere",
            MeshKind::Cube => "Cube",
            MeshKind::Plane => "Plane",
            MeshKind::RoundedCube => "Rounded Cube",
            MeshKind::RoundedCylinder => "Rounded Cylinder",
        }
    }
}

/// The 7 PBR maps the shader reads, each falling back to a flat default
/// texture when the graph has no node for that Output kind.
struct FallbackTextures {
    base_color: wgpu::TextureView,
    normal: wgpu::TextureView,
    roughness: wgpu::TextureView,
    metallic: wgpu::TextureView,
    ao: wgpu::TextureView,
    height: wgpu::TextureView,
    alpha: wgpu::TextureView,
}

pub struct Viewport3d {
    pipeline: wgpu::RenderPipeline,
    pipeline_transparent: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    sampler_smooth: wgpu::Sampler,
    sampler_pixel: wgpu::Sampler,
    // Built lazily and rebuilt whenever `(mesh_kind, tessellation)` changes,
    // rather than every `MeshKind` pre-built once at a fixed quality — see
    // `tessellated_mesh`. The tessellation value is rounded to a whole
    // percent for the comparison key so tiny float jitter doesn't trigger a
    // rebuild on every frame.
    current_mesh: (wgpu::Buffer, wgpu::Buffer, u32),
    current_mesh_key: (MeshKind, u32),
    fallback: FallbackTextures,
    environment_view: wgpu::TextureView,
    environment_ambient_color: Vec3,
    pub environment_source: EnvironmentSource,
    /// Set whenever the environment changes; `render()` regenerates and
    /// (re-)registers `environment_thumbnail_id` the next time it runs, the
    /// same lazy-rebuild-on-change pattern as `current_mesh_key`. Starts
    /// `true` so the default environment's thumbnail exists from the first
    /// frame without `new()` needing an `egui_wgpu::RenderState` of its own.
    environment_thumbnail_dirty: bool,
    environment_thumbnail_id: Option<egui::TextureId>,

    color_view: Option<wgpu::TextureView>,
    depth_view: Option<wgpu::TextureView>,
    texture_id: Option<egui::TextureId>,
    size: (u32, u32),

    pub mesh_kind: MeshKind,
    pub height_scale: f32,
    pub tessellation: f32,
    pub shading_mode: ShadingMode,
    pub exposure: f32,
    /// Scales just the environment's contribution (sky reflection + ambient
    /// fill), independent of `exposure` above — lets a too-bright HDRI be
    /// dimmed without affecting direct light.
    pub environment_intensity: f32,
    /// 0 = exact mirror reflection of the environment; 1 = heavily blurred.
    /// Independent of surface roughness (which already dampens reflection
    /// *strength* — see `fs_main`'s `specular_strength`): a perfectly smooth
    /// metal has roughness ~0 and so no damping at all, which is exactly the
    /// case this is for — letting the artist soften that mirror without
    /// having to fake it by raising roughness.
    pub environment_blur: f32,
    pub texture_filter: TextureFilter,
    yaw: f32,
    pitch: f32,
    distance: f32,
    /// World-space point the camera orbits around and looks at — shifted by
    /// right-click-drag panning (`handle_input`); the mesh itself always
    /// stays at the origin, only the camera moves.
    target: Vec3,
}

/// "Smooth" is trilinear + 16x anisotropic — the normal case. "Pixel"
/// disables all interpolation (texel and mip), for inspecting dither/
/// pixel-art materials without any blending softening the exact pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextureFilter {
    #[default]
    Smooth,
    Pixel,
}

/// Which HDRI is currently lighting the scene — a bundled preset, or a file
/// the user loaded themselves. Kept on `Viewport3d` (rather than recomputed
/// each frame) purely so the Environment panel can show which one is active;
/// the texture/ambient color it produced are what actually drive rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvironmentSource {
    Preset(crate::hdri::HdriPreset),
    /// The file name only (not the full path) — just for display, since
    /// nothing needs to re-read the file after it's loaded.
    Custom(String),
}

/// A debug ground-truth view (CLAUDE.md §8's linear/sRGB discipline):
/// bypasses PBR shading and exposure/tonemapping entirely, outputting Base
/// Color with just the sRGB encode applied — for comparing pixel-for-pixel
/// against the 2D View and an exported PNG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShadingMode {
    #[default]
    Lit,
    UnlitBaseColor,
}

impl Viewport3d {
    pub fn new(gpu: &mc_gpu::Context) -> Self {
        let device = &gpu.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport3d"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/viewport3d.wgsl").into()),
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("viewport3d uniforms layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("viewport3d textures layout"),
                entries: &[
                    texture_entry(0, wgpu::ShaderStages::FRAGMENT),
                    texture_entry(1, wgpu::ShaderStages::FRAGMENT),
                    texture_entry(2, wgpu::ShaderStages::FRAGMENT),
                    texture_entry(3, wgpu::ShaderStages::FRAGMENT),
                    texture_entry(4, wgpu::ShaderStages::FRAGMENT),
                    texture_entry(5, wgpu::ShaderStages::VERTEX),
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    texture_entry(7, wgpu::ShaderStages::FRAGMENT),
                    texture_entry(8, wgpu::ShaderStages::FRAGMENT),
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport3d pipeline layout"),
            bind_group_layouts: &[
                Some(&uniform_bind_group_layout),
                Some(&texture_bind_group_layout),
            ],
            immediate_size: 0,
        });

        // Two pipeline variants, identical except for depth write — see
        // `render()`'s doc comment on why a graph with an Alpha node needs
        // the other one. Built from a closure rather than two near-copies
        // of the same ~30-line descriptor.
        let build_pipeline = |label: &'static str, depth_write_enabled: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &VERTEX_ATTRS,
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        // Real alpha blending, not a flat overwrite, so the
                        // Output: Alpha channel (see viewport3d.wgsl's
                        // module doc comment) actually shows the background
                        // through a low-alpha area instead of being
                        // silently discarded. A graph with no Alpha node
                        // falls back to a fully opaque 1.0 (see
                        // `FallbackTextures::alpha`), which blends
                        // identically to the old `REPLACE` behavior, so
                        // this is a no-op for every existing graph.
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    // Preview-only: not worth getting every mesh's winding
                    // order exactly right when there's no performance need
                    // to cull.
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(depth_write_enabled),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline = build_pipeline("viewport3d pipeline (opaque)", true);
        // With `cull_mode: None`, several faces of a closed mesh (e.g. a
        // cube's far side) can land on the same screen pixel — depth write
        // is what makes that resolve correctly for fully-opaque geometry.
        // But once alpha blending is actually doing something (a real Alpha
        // node wired in), keeping depth write on breaks it: whichever
        // partially-transparent face happens to be drawn first still writes
        // full depth, so a face behind it that should show through gets
        // rejected by the depth test regardless of how transparent the
        // first one was — seen live as background "partially" showing
        // through in a way that depends on draw order, not on the actual
        // alpha value. Turning depth write off for this pipeline variant
        // fixes that; the known tradeoff (noted in viewport3d.wgsl's module
        // doc comment) is that it can't correctly sort two *different*
        // transparent layers of the same mesh against each other, only
        // against the opaque background — an acceptable simplification for
        // a material preview, not a compositor.
        let pipeline_transparent = build_pipeline("viewport3d pipeline (transparent)", false);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport3d uniforms"),
            size: UNIFORM_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport3d uniforms bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Anisotropic filtering needs mag/min/mipmap filter all `Linear` to
        // take effect at all (a wgpu validation requirement) — sharpens
        // textures viewed at a glancing angle, where trilinear alone still
        // blurs heavily since it can only pick an isotropic (square) mip
        // footprint for a genuinely elongated one.
        let sampler_smooth = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewport3d sampler (smooth)"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });
        // No interpolation at all, mip or texel — for inspecting dither/
        // pixel-art materials crisply, where any blending (even between
        // adjacent texels at mip 0) muddies the exact pattern.
        let sampler_pixel = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewport3d sampler (pixel)"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let initial_tessellation = 0.5;
        let current_mesh = upload_mesh(
            device,
            tessellated_mesh(MeshKind::RoundedCube, initial_tessellation),
        );
        let current_mesh_key = (
            MeshKind::RoundedCube,
            tessellation_bucket(initial_tessellation),
        );

        let fallback = FallbackTextures {
            base_color: solid_view(
                device,
                &gpu.queue,
                "fallback base color",
                [140, 140, 140, 255],
            ),
            normal: solid_view(device, &gpu.queue, "fallback normal", [128, 128, 255, 255]),
            // Fully rough (and, via `metallic` below, fully non-metal) by
            // default, so connecting a node to either Output clearly reads
            // as a change from a neutral matte baseline rather than from an
            // arbitrary partial value.
            roughness: solid_view(
                device,
                &gpu.queue,
                "fallback roughness",
                [255, 255, 255, 255],
            ),
            metallic: solid_view(device, &gpu.queue, "fallback metallic", [0, 0, 0, 255]),
            ao: solid_view(device, &gpu.queue, "fallback ao", [255, 255, 255, 255]),
            height: solid_view(device, &gpu.queue, "fallback height", [128, 128, 128, 255]),
            // Fully opaque by default — a graph with no Alpha node should
            // render exactly as it did before Alpha existed.
            alpha: solid_view(device, &gpu.queue, "fallback alpha", [255, 255, 255, 255]),
        };

        let initial_preset = crate::hdri::HdriPreset::StudioSmall03;
        let (environment_view, environment_ambient_color) =
            crate::hdri::load_preset(device, &gpu.queue, initial_preset);
        let environment_ambient_color = Vec3::from(environment_ambient_color);

        Self {
            pipeline,
            pipeline_transparent,
            uniform_buffer,
            uniform_bind_group,
            texture_bind_group_layout,
            sampler_smooth,
            sampler_pixel,
            current_mesh,
            current_mesh_key,
            fallback,
            environment_view,
            environment_ambient_color,
            environment_source: EnvironmentSource::Preset(initial_preset),
            environment_thumbnail_dirty: true,
            environment_thumbnail_id: None,
            color_view: None,
            depth_view: None,
            texture_id: None,
            size: (0, 0),
            mesh_kind: MeshKind::RoundedCube,
            height_scale: 0.15,
            tessellation: initial_tessellation,
            shading_mode: ShadingMode::default(),
            exposure: 1.0,
            environment_intensity: 1.0,
            environment_blur: 0.0,
            texture_filter: TextureFilter::default(),
            yaw: 0.6,
            pitch: 0.35,
            distance: 3.0,
            target: Vec3::ZERO,
        }
    }

    /// Switches the sky/reflection environment to a bundled preset. Infallible
    /// — presets are checked into the repo and covered by a decode test, so
    /// any failure here would mean a corrupted build, not bad user input.
    pub fn set_environment_preset(
        &mut self,
        gpu: &mc_gpu::Context,
        preset: crate::hdri::HdriPreset,
    ) {
        let (view, ambient) = crate::hdri::load_preset(&gpu.device, &gpu.queue, preset);
        self.environment_view = view;
        self.environment_ambient_color = Vec3::from(ambient);
        self.environment_source = EnvironmentSource::Preset(preset);
        self.environment_thumbnail_dirty = true;
    }

    /// Switches the sky/reflection environment to a user-supplied `.hdr`/
    /// `.exr` file. Unlike [`Self::set_environment_preset`], every failure
    /// mode here is reachable from an arbitrary file the user picked, so the
    /// current environment is left untouched on `Err` — the caller is
    /// expected to surface it as a toast rather than crash the app.
    pub fn set_environment_custom(
        &mut self,
        gpu: &mc_gpu::Context,
        path: &std::path::Path,
    ) -> Result<(), String> {
        let (view, ambient) = crate::hdri::load_custom(&gpu.device, &gpu.queue, path)?;
        self.environment_view = view;
        self.environment_ambient_color = Vec3::from(ambient);
        self.environment_source = EnvironmentSource::Custom(
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string()),
        );
        self.environment_thumbnail_dirty = true;
        Ok(())
    }

    /// The active environment's preview thumbnail, registered as an egui
    /// texture — `None` only for the first frame or two after startup or an
    /// environment change, before `render()` has run once to build it.
    pub fn environment_thumbnail(&self) -> Option<egui::TextureId> {
        self.environment_thumbnail_id
    }

    /// Orbit (left-drag), pan (right-drag), and zoom (scroll, while hovered)
    /// camera controls.
    pub fn handle_input(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            // Pan speed scales with distance so a dragged-screen-pixel keeps
            // tracking roughly the same world-space point regardless of
            // zoom level — fixed FOV means more world space fits in a pixel
            // the farther the camera is.
            let pan_speed = self.distance * 0.0015;
            let forward = -self.orbit_offset().normalize_or_zero();
            let right = forward.cross(Vec3::Y).normalize_or_zero();
            let up = right.cross(forward).normalize_or_zero();
            // Move the camera opposite the drag so the *content* tracks the
            // cursor (the same "grab the canvas" convention `canvas.rs`'s 2D
            // pan uses) — screen Y grows downward, world up doesn't, hence
            // the sign flip on the vertical term.
            self.target -= right * delta.x * pan_speed;
            self.target += up * delta.y * pan_speed;
        } else if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            self.yaw -= delta.x * 0.01;
            self.pitch = (self.pitch - delta.y * 0.01).clamp(-1.4, 1.4);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.distance = (self.distance - scroll * 0.01).clamp(1.5, 10.0);
            }
        }
    }

    /// The camera's offset from `target` — orbit radius/angles only, with no
    /// pan applied. `eye_position` below is just this plus `target`; pan
    /// (`handle_input`) only ever moves `target`, never this offset.
    fn orbit_offset(&self) -> Vec3 {
        Vec3::new(
            self.distance * self.pitch.cos() * self.yaw.sin(),
            self.distance * self.pitch.sin(),
            self.distance * self.pitch.cos() * self.yaw.cos(),
        )
    }

    fn eye_position(&self) -> Vec3 {
        self.target + self.orbit_offset()
    }

    fn recreate_targets(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport3d color target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport3d depth target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.color_view = Some(color_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.depth_view = Some(depth_texture.create_view(&wgpu::TextureViewDescriptor::default()));
    }

    fn build_texture_bind_group(
        &self,
        gpu: &mut mc_gpu::Context,
        pool: &mc_gpu::TexturePool,
        graph: &mc_core::Graph,
    ) -> wgpu::BindGroup {
        // Resolves `kind`'s channel to a view spanning its *entire* mip
        // chain (regenerating that chain first), or the 1x1 fallback —
        // which has only one mip to begin with, so it's used as-is.
        // `wgpu::TextureView` is a cheap `Clone` (a GPU-side handle, not a
        // borrow of `pool`), so there's no need to keep anything alive
        // beyond this closure call.
        let mut channel_view = |kind: &str, is_normal_map: bool, fallback: &wgpu::TextureView| {
            let Some(key) = resolve_channel_key(graph, kind) else {
                return fallback.clone();
            };
            // This bind group's own sampler (binding 6) always filters —
            // every real channel texture is `Rgba16Float`, which supports
            // that, *except* a node whose own output is deliberately
            // higher-precision (so far just Gabor Noise's `raw`, stored as
            // `Rgba32Float`, which WebGPU never allows filtered sampling
            // of without an opt-in device feature this app doesn't
            // request). Falling back here — treating it the same as
            // nothing meaningfully connected — is the same call its own
            // doc comment already makes ("for math nodes downstream, not
            // direct display"); the alternative (a second, non-filtering
            // bind group layout + pipeline variant just for this one
            // case) is a lot of machinery for a channel that was never a
            // sensible thing to view directly in the first place.
            let Some(texture) = pool.texture(&key) else {
                return fallback.clone();
            };
            if texture.format() != wgpu::TextureFormat::Rgba16Float {
                return fallback.clone();
            }
            crate::mipgen::generate_mip_chain(gpu, pool, &key, is_normal_map);
            crate::mipgen::full_mip_view(pool, &key).unwrap_or_else(|| fallback.clone())
        };

        let base_color = channel_view("output_base_color", false, &self.fallback.base_color);
        let normal = channel_view("output_normal", true, &self.fallback.normal);
        let roughness = channel_view("output_roughness", false, &self.fallback.roughness);
        let metallic = channel_view("output_metallic", false, &self.fallback.metallic);
        let ao = channel_view("output_ambient_occlusion", false, &self.fallback.ao);
        let height = channel_view("output_height", false, &self.fallback.height);
        let alpha = channel_view("output_alpha", false, &self.fallback.alpha);

        let device = &gpu.device;
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport3d textures bind group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base_color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&normal),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&roughness),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&metallic),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&ao),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&height),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(match self.texture_filter {
                        TextureFilter::Smooth => &self.sampler_smooth,
                        TextureFilter::Pixel => &self.sampler_pixel,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&self.environment_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&alpha),
                },
            ],
        })
    }

    /// Renders the current mesh lit by the current graph's outputs into an
    /// offscreen target sized `width x height`, and returns the egui texture
    /// id it's registered under (stable across frames and resizes, so the UI
    /// side can just keep displaying the same id).
    pub fn render(
        &mut self,
        gpu: &mut mc_gpu::Context,
        pool: &mc_gpu::TexturePool,
        graph: &mc_core::Graph,
        render_state: &egui_wgpu::RenderState,
        width: u32,
        height: u32,
    ) -> egui::TextureId {
        let width = width.max(1);
        let height = height.max(1);
        if self.size != (width, height) || self.color_view.is_none() {
            self.recreate_targets(&gpu.device, width, height);
            self.size = (width, height);
        }

        if self.environment_thumbnail_dirty {
            let thumbnail_view = crate::hdri::render_thumbnail(gpu, &self.environment_view);
            self.environment_thumbnail_id = Some(match self.environment_thumbnail_id {
                Some(texture_id) => {
                    render_state
                        .renderer
                        .write()
                        .update_egui_texture_from_wgpu_texture(
                            &gpu.device,
                            &thumbnail_view,
                            wgpu::FilterMode::Linear,
                            texture_id,
                        );
                    texture_id
                }
                None => render_state.renderer.write().register_native_texture(
                    &gpu.device,
                    &thumbnail_view,
                    wgpu::FilterMode::Linear,
                ),
            });
            self.environment_thumbnail_dirty = false;
        }

        let aspect = width as f32 / height as f32;
        let eye = self.eye_position();
        let view = glam::camera::rh::view::look_at_mat4(eye, self.target, Vec3::Y);
        let proj =
            glam::camera::rh::proj::directx::perspective(60f32.to_radians(), aspect, 0.1, 100.0);
        let light_dir = Vec3::new(-0.4, -1.0, -0.3).normalize();

        let bytes = uniform_bytes(
            proj * view,
            eye,
            light_dir,
            self.environment_ambient_color,
            RenderSettings {
                height_scale: self.height_scale,
                shading_mode: self.shading_mode,
                exposure: self.exposure,
                environment_intensity: self.environment_intensity,
                environment_blur: self.environment_blur,
            },
        );
        gpu.queue.write_buffer(&self.uniform_buffer, 0, &bytes);

        let texture_bind_group = self.build_texture_bind_group(gpu, pool, graph);

        let desired_key = (self.mesh_kind, tessellation_bucket(self.tessellation));
        if desired_key != self.current_mesh_key {
            self.current_mesh = upload_mesh(
                &gpu.device,
                tessellated_mesh(self.mesh_kind, self.tessellation),
            );
            self.current_mesh_key = desired_key;
        }
        let (vertex_buffer, index_buffer, index_count) = &self.current_mesh;

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("viewport3d encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport3d pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.color_view.as_ref().expect("just created"),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(crate::theme::NEUTRON_STAR_WGPU),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.depth_view.as_ref().expect("just created"),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // See `pipeline_transparent`'s doc comment in `new()`: depth
            // write has to turn off once a real Alpha node is actually
            // driving transparency, or partially-transparent faces fight
            // the depth buffer in a way that depends on draw order rather
            // than the alpha value itself.
            let pipeline = if resolve_channel_key(graph, "output_alpha").is_some() {
                &self.pipeline_transparent
            } else {
                &self.pipeline
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &texture_bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..*index_count, 0, 0..1);
        }
        gpu.queue.submit(Some(encoder.finish()));

        let color_view = self.color_view.as_ref().expect("just created");
        match self.texture_id {
            Some(texture_id) => {
                render_state
                    .renderer
                    .write()
                    .update_egui_texture_from_wgpu_texture(
                        &gpu.device,
                        color_view,
                        wgpu::FilterMode::Linear,
                        texture_id,
                    );
                texture_id
            }
            None => {
                let texture_id = render_state.renderer.write().register_native_texture(
                    &gpu.device,
                    color_view,
                    wgpu::FilterMode::Linear,
                );
                self.texture_id = Some(texture_id);
                texture_id
            }
        }
    }
}

fn texture_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// The graph's texture pool key for one Output kind (e.g.
/// `"output_base_color"`), if the graph has that Output node and it
/// resolves to something. Mirrors `mc_nodes::export::read_output`'s lookup,
/// minus the readback.
fn resolve_channel_key(graph: &mc_core::Graph, kind: &str) -> Option<String> {
    let id = graph
        .nodes()
        .find(|(_, n)| n.kind == kind)
        .map(|(id, _)| id)?;
    mc_nodes::registry::resolve_texture_key(graph, id)
}

fn solid_view(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    color: [u8; 4],
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &color,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn upload_mesh(
    device: &wgpu::Device,
    (vertices, indices): (Vec<Vertex>, Vec<u32>),
) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let mut vertex_bytes = Vec::with_capacity(vertices.len() * std::mem::size_of::<Vertex>());
    for v in &vertices {
        for c in v.position {
            vertex_bytes.extend_from_slice(&c.to_le_bytes());
        }
        for c in v.normal {
            vertex_bytes.extend_from_slice(&c.to_le_bytes());
        }
        for c in v.uv {
            vertex_bytes.extend_from_slice(&c.to_le_bytes());
        }
        for c in v.tangent {
            vertex_bytes.extend_from_slice(&c.to_le_bytes());
        }
    }
    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewport3d mesh vertices"),
        size: vertex_bytes.len() as u64,
        usage: wgpu::BufferUsages::VERTEX,
        mapped_at_creation: true,
    });
    vertex_buffer
        .slice(..)
        .get_mapped_range_mut()
        .expect("buffer was just created mapped")
        .copy_from_slice(&vertex_bytes);
    vertex_buffer.unmap();

    let mut index_bytes = Vec::with_capacity(indices.len() * 4);
    for i in &indices {
        index_bytes.extend_from_slice(&i.to_le_bytes());
    }
    let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewport3d mesh indices"),
        size: index_bytes.len() as u64,
        usage: wgpu::BufferUsages::INDEX,
        mapped_at_creation: true,
    });
    index_buffer
        .slice(..)
        .get_mapped_range_mut()
        .expect("buffer was just created mapped")
        .copy_from_slice(&index_bytes);
    index_buffer.unmap();

    (vertex_buffer, index_buffer, indices.len() as u32)
}

/// `view_proj`(mat4) + `camera_pos`(vec4) + `light_dir`(vec4) +
/// `ambient_color`(vec4) + 5 scalars + 3 padding floats = 144 bytes, matching
/// `Uniforms` in viewport3d.wgsl.
const UNIFORM_SIZE: usize = 144;

/// The render-settings scalars that make up `uniform_bytes`' tail — grouped
/// into one struct rather than 5 more positional args, which pushed that
/// function over clippy's `too_many_arguments` threshold.
struct RenderSettings {
    height_scale: f32,
    shading_mode: ShadingMode,
    exposure: f32,
    environment_intensity: f32,
    environment_blur: f32,
}

fn uniform_bytes(
    view_proj: Mat4,
    camera_pos: Vec3,
    light_dir: Vec3,
    ambient_color: Vec3,
    settings: RenderSettings,
) -> [u8; UNIFORM_SIZE] {
    let mut bytes = [0u8; UNIFORM_SIZE];
    let mut offset = 0;
    let put = |value: f32, bytes: &mut [u8; UNIFORM_SIZE], offset: &mut usize| {
        bytes[*offset..*offset + 4].copy_from_slice(&value.to_le_bytes());
        *offset += 4;
    };
    for component in view_proj.to_cols_array() {
        put(component, &mut bytes, &mut offset);
    }
    for component in [camera_pos.x, camera_pos.y, camera_pos.z, 1.0] {
        put(component, &mut bytes, &mut offset);
    }
    for component in [light_dir.x, light_dir.y, light_dir.z, 0.0] {
        put(component, &mut bytes, &mut offset);
    }
    for component in [ambient_color.x, ambient_color.y, ambient_color.z, 0.0] {
        put(component, &mut bytes, &mut offset);
    }
    put(settings.height_scale, &mut bytes, &mut offset);
    let shading_mode_value = match settings.shading_mode {
        ShadingMode::Lit => 0.0,
        ShadingMode::UnlitBaseColor => 1.0,
    };
    put(shading_mode_value, &mut bytes, &mut offset);
    put(settings.exposure, &mut bytes, &mut offset);
    put(settings.environment_intensity, &mut bytes, &mut offset);
    put(settings.environment_blur, &mut bytes, &mut offset);
    put(0.0, &mut bytes, &mut offset);
    put(0.0, &mut bytes, &mut offset);
    put(0.0, &mut bytes, &mut offset);
    bytes
}

/// Rounds `tessellation` (0.0..1.0) to a whole percent, so the "has anything
/// actually changed" check in `render` compares integers instead of floats.
fn tessellation_bucket(tessellation: f32) -> u32 {
    (tessellation.clamp(0.0, 1.0) * 100.0).round() as u32
}

fn lerp_segments(low: u32, high: u32, t: f32) -> u32 {
    (low as f32 + (high as f32 - low as f32) * t.clamp(0.0, 1.0)).round() as u32
}

/// Builds the given mesh kind at a quality level driven by `tessellation`
/// (0.0 = coarsest, 1.0 = finest), scaling each mesh's segment counts between
/// a usable minimum and a smooth maximum. Every mesh gets *some* subdivision
/// now, even flat-faced ones (`Cube`, and the straight side wall of
/// `RoundedCylinder`) — height displacement can only bend a face as a ruled
/// surface between however many vertices it has, so an undivided flat quad
/// showed none of a height map's actual detail, just the illusion of it from
/// normal mapping's shading.
fn tessellated_mesh(kind: MeshKind, tessellation: f32) -> (Vec<Vertex>, Vec<u32>) {
    match kind {
        // Rings (pole-to-pole) raised from 96 to 128 to match the same
        // floor the Cube/Rounded Cube audit settled on, after the owner
        // asked whether the other meshes had enough resolution too —
        // segments (192, around the equator) was already comfortably above
        // it.
        MeshKind::Sphere => generate_sphere(
            lerp_segments(8, 128, tessellation),
            lerp_segments(16, 192, tessellation),
        ),
        // 64 segments/face wasn't enough resolution to follow a sharp height
        // edge without visible faceting even at maximum tessellation,
        // confirmed live by the owner with a clean hard-edged cutout shape
        // (not fine texture detail the mesh "should" struggle with) — raised
        // to 128, doubling the vertex density along each axis (4x the quads
        // per face). Still not a complete fix for an arbitrarily sharp edge
        // (no fixed mesh resolution ever fully is), but a meaningfully
        // smaller artifact; see PROGRESS.md.
        MeshKind::Cube => generate_rounded_cube(lerp_segments(4, 128, tessellation), 0.0),
        MeshKind::Plane => generate_plane(lerp_segments(8, 256, tessellation)),
        MeshKind::RoundedCube => generate_rounded_cube(lerp_segments(4, 128, tessellation), 0.25),
        // Fillet/wall segments (the vertical axis) doubled from 24/48 to
        // 48/96 for the same reason as Sphere above — their sum (~72) was
        // noticeably below the new floor. radial_segments (192, around the
        // circumference) was already well above it.
        MeshKind::RoundedCylinder => generate_rounded_cylinder(
            lerp_segments(16, 192, tessellation),
            lerp_segments(2, 48, tessellation),
            lerp_segments(2, 96, tessellation),
            0.3,
        ),
    }
}

fn generate_sphere(rings: u32, segments: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let phi = v * std::f32::consts::PI;
        let y = phi.cos();
        let r = phi.sin();
        for seg in 0..=segments {
            let u = seg as f32 / segments as f32;
            let theta = u * std::f32::consts::TAU;
            let x = r * theta.cos();
            let z = r * theta.sin();
            let position = [x, y, z];
            // At a pole, every vertex around this "ring" sits at the exact
            // same 3D point despite having a different u — height
            // displacement moves each by however much *its own* u happens
            // to sample, which tears that single point into a star/rosette.
            // Giving them all the same u instead means they displace
            // together, by construction. Same pinch every UV sphere/cone has
            // at its poles, just put to use here rather than worked around.
            let pole_safe_u = if r.abs() < 1e-5 { 0.0 } else { u };
            vertices.push(Vertex {
                position,
                normal: position,
                uv: [pole_safe_u, v],
                tangent: [0.0; 3],
            });
        }
    }

    let stride = segments + 1;
    let mut indices = Vec::new();
    for ring in 0..rings {
        for seg in 0..segments {
            let a = ring * stride + seg;
            let b = a + stride;
            let c = a + 1;
            let d = b + 1;
            indices.extend_from_slice(&[a, b, c, c, b, d]);
        }
    }
    compute_tangents(&mut vertices, &indices);
    (vertices, indices)
}

/// A cube with filleted edges/corners: each face is a `segments`×`segments`
/// grid on the unit cube, then every vertex is pulled outward from an
/// "inner" box (half-extent `1 - radius`) by `radius` along the direction
/// it was clamped in. On the flat middle of a face that direction is just
/// the face normal (so the vertex doesn't move at all); near an edge or
/// corner it's the diagonal away from the inner box, tracing out exactly the
/// quarter-cylinder/eighth-sphere fillet a real rounded box has there.
///
/// `radius: 0.0` degenerates cleanly to a plain sharp-edged cube (`clamped`
/// never actually clamps anything, so every vertex just stays where the grid
/// put it) — `MeshKind::Cube` uses exactly that rather than a separate
/// unsubdivided cube generator, since a flat-faced-but-undivided mesh has no
/// geometry for height displacement to bend.
fn generate_rounded_cube(segments: u32, radius: f32) -> (Vec<Vertex>, Vec<u32>) {
    let radius = radius.clamp(0.0, 1.0);
    let inner = 1.0 - radius;
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ];

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for (normal, right, up) in faces {
        let normal = Vec3::from(normal);
        let right = Vec3::from(right);
        let up = Vec3::from(up);
        let base = vertices.len() as u32;
        for j in 0..=segments {
            let v = j as f32 / segments as f32 * 2.0 - 1.0;
            for i in 0..=segments {
                let u = i as f32 / segments as f32 * 2.0 - 1.0;
                let p = normal + right * u + up * v;
                let clamped = p.clamp(Vec3::splat(-inner), Vec3::splat(inner));
                let delta = p - clamped;
                let n = if delta.length_squared() > 1e-12 {
                    delta.normalize()
                } else {
                    normal
                };
                let position = clamped + n * radius;
                vertices.push(Vertex {
                    position: position.to_array(),
                    normal: n.to_array(),
                    uv: [i as f32 / segments as f32, 1.0 - j as f32 / segments as f32],
                    tangent: [0.0; 3],
                });
            }
        }
        let stride = segments + 1;
        for j in 0..segments {
            for i in 0..segments {
                let a = base + j * stride + i;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                indices.extend_from_slice(&[a, b, d, a, d, c]);
            }
        }
    }
    compute_tangents(&mut vertices, &indices);
    weld_seam_vertices(&vertices, &mut indices);
    (vertices, indices)
}

/// Welds vertices that land at the same physical position but came from
/// different faces — every edge of this per-face-UV-mapped cube has
/// exactly that (two faces' independently generated vertex grids meeting
/// at the same edge, but on opposite sides of a UV seam). Without this,
/// height displacement — which reads each vertex's own UV to sample the
/// Height texture (`vs_main` in `viewport3d.wgsl`) — pulls each face's
/// copy of a shared edge point by a *different* amount, whatever that
/// face's own, unrelated UV happens to sample there, tearing the mesh
/// open into a visible crack along every cube edge with any height
/// contrast nearby — worse with more height detail, and *not* fixed by
/// raising tessellation, since the crack isn't a resolution problem: it's
/// two genuinely different target positions for what should be one
/// shared point. Welding makes every triangle on both sides of a seam
/// reference the exact same vertex, so there is no way for them to end up
/// in different places — at the cost of a thin, one-row sliver where the
/// *losing* face's own edge row shows the winning face's color instead of
/// its own, the standard, much less visible trade-off box-mapped meshes
/// accept instead of an open seam. Call after `compute_tangents`, not
/// before — each vertex's tangent should come from its own face's UV
/// gradient, not get confused by a weld that's about to make two
/// differently-UV'd vertices share an index.
fn weld_seam_vertices(vertices: &[Vertex], indices: &mut [u32]) {
    // Quantized to 1e-6 — far finer than any real edge length this mesh
    // ever produces (even at the maximum 128 segments, the shortest edge
    // is still several orders of magnitude longer), so this only ever
    // merges points that are the same position up to float rounding, never
    // two distinct-but-nearby vertices.
    const SCALE: f32 = 1_000_000.0;
    let key = |p: [f32; 3]| -> (i64, i64, i64) {
        (
            (p[0] * SCALE).round() as i64,
            (p[1] * SCALE).round() as i64,
            (p[2] * SCALE).round() as i64,
        )
    };

    let mut canonical: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut remap = vec![0u32; vertices.len()];
    for (i, v) in vertices.iter().enumerate() {
        let canonical_index = *canonical.entry(key(v.position)).or_insert(i as u32);
        remap[i] = canonical_index;
    }
    for idx in indices.iter_mut() {
        *idx = remap[*idx as usize];
    }
}

/// One point of the 2D (radial-distance, height) profile that
/// [`revolve`] sweeps around the Y axis to build a surface of revolution —
/// `(nr, ny)` is that profile curve's own outward normal, not a 3D one.
struct ProfilePoint {
    r: f32,
    y: f32,
    nr: f32,
    ny: f32,
}

/// Sweeps a profile curve around the Y axis in `segments` angular steps,
/// the same ring-by-ring construction [`generate_sphere`] uses for its
/// lines of latitude — a cylinder, cone, torus, or (as used by
/// [`generate_rounded_cylinder`]) a rounded cylinder are all just different
/// profiles through this one function.
fn revolve(profile: &[ProfilePoint], segments: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    for (ring, p) in profile.iter().enumerate() {
        let v = ring as f32 / (profile.len() - 1) as f32;
        for seg in 0..=segments {
            let u = seg as f32 / segments as f32;
            let theta = u * std::f32::consts::TAU;
            let (sin_t, cos_t) = theta.sin_cos();
            // Same pole pinch as `generate_sphere`: at the center of a flat
            // cap (`p.r == 0`), every vertex around this ring is the same 3D
            // point — give them the same u so they all get the same height
            // displacement instead of tearing that point into a star.
            let pole_safe_u = if p.r.abs() < 1e-5 { 0.0 } else { u };
            vertices.push(Vertex {
                position: [p.r * cos_t, p.y, p.r * sin_t],
                normal: [p.nr * cos_t, p.ny, p.nr * sin_t],
                uv: [pole_safe_u, v],
                tangent: [0.0; 3],
            });
        }
    }

    let stride = segments + 1;
    let mut indices = Vec::new();
    for ring in 0..profile.len() as u32 - 1 {
        for seg in 0..segments {
            let a = ring * stride + seg;
            let b = a + stride;
            let c = a + 1;
            let d = b + 1;
            indices.extend_from_slice(&[a, b, c, c, b, d]);
        }
    }
    compute_tangents(&mut vertices, &indices);
    (vertices, indices)
}

/// A cylinder with filleted top/bottom rim edges, built as a surface of
/// revolution: flat top cap → quarter-circle fillet → straight side wall →
/// quarter-circle fillet → flat bottom cap.
fn generate_rounded_cylinder(
    radial_segments: u32,
    fillet_segments: u32,
    wall_segments: u32,
    fillet_radius: f32,
) -> (Vec<Vertex>, Vec<u32>) {
    let radius = 1.0_f32;
    let half_height = 1.0_f32;
    let fillet = fillet_radius.clamp(0.0, radius.min(half_height));
    let flat_radius = radius - fillet;
    let flat_half_height = half_height - fillet;

    // No flat caps: the mesh is open-ended, just the rounded rim curving
    // into the side wall and back out the other end — a flat cap built from
    // a fan of triangles meeting at the center point is exactly what was
    // causing the mapping problems (the center point is one 3D position
    // shared by every vertex around that fan, each sampling a different
    // part of the height/normal textures, which either tore the fan apart
    // under displacement or, pinned to a single UV to stop that, left the
    // whole cap without real texture detail). Simplest fix is to not have
    // that fan at all.
    let mut profile = Vec::new();
    profile.push(ProfilePoint {
        r: flat_radius,
        y: half_height,
        nr: 0.0,
        ny: 1.0,
    });
    for i in 1..=fillet_segments {
        let t = i as f32 / fillet_segments as f32 * std::f32::consts::FRAC_PI_2;
        let (nr, ny) = (t.sin(), t.cos());
        profile.push(ProfilePoint {
            r: flat_radius + fillet * nr,
            y: flat_half_height + fillet * ny,
            nr,
            ny,
        });
    }
    // The straight side wall, same radius/normal throughout — subdivided
    // along its height so vertex displacement has enough geometry to
    // actually trace the height map's detail. A single edge from top to
    // bottom (as this used to be) can only bend the whole wall as one flat
    // ruled strip between two displaced rings, which isn't fine enough to
    // show anything but the broadest height variation.
    for i in 1..wall_segments {
        let t = i as f32 / wall_segments as f32;
        profile.push(ProfilePoint {
            r: radius,
            y: flat_half_height - t * (2.0 * flat_half_height),
            nr: 1.0,
            ny: 0.0,
        });
    }
    profile.push(ProfilePoint {
        r: radius,
        y: -flat_half_height,
        nr: 1.0,
        ny: 0.0,
    });
    for i in 1..=fillet_segments {
        let t = i as f32 / fillet_segments as f32 * std::f32::consts::FRAC_PI_2;
        let (nr, ny) = (t.cos(), -t.sin());
        profile.push(ProfilePoint {
            r: flat_radius + fillet * nr,
            y: -flat_half_height + fillet * ny,
            nr,
            ny,
        });
    }
    // The lower fillet loop above already ends exactly on the bottom rim —
    // no closing cap here either, same reasoning as the top.

    revolve(&profile, radial_segments)
}

fn generate_plane(segments: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    for j in 0..=segments {
        let v = j as f32 / segments as f32;
        let z = v * 2.0 - 1.0;
        for i in 0..=segments {
            let u = i as f32 / segments as f32;
            let x = u * 2.0 - 1.0;
            vertices.push(Vertex {
                position: [x, 0.0, z],
                normal: [0.0, 1.0, 0.0],
                uv: [u, 1.0 - v],
                tangent: [0.0; 3],
            });
        }
    }

    let stride = segments + 1;
    let mut indices = Vec::new();
    for j in 0..segments {
        for i in 0..segments {
            let a = j * stride + i;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    compute_tangents(&mut vertices, &indices);
    (vertices, indices)
}

/// Computes a per-vertex tangent (the direction UV's U axis actually runs
/// across the surface) from each triangle's position/UV gradient — the
/// standard technique (Lengyel), averaged across every triangle a vertex
/// touches and re-orthogonalized against its normal. This is what lets the
/// shader build a real tangent-space basis for normal mapping instead of
/// faking one from the normal alone (which only happened to look right on
/// the Plane, where the normal is constant and the UV layout lines up with
/// it by coincidence — see `viewport3d.wgsl`).
///
/// Deliberately driven only by `position`/`normal`/`uv`, not by any
/// knowledge of which primitive generated them, so it works unchanged for
/// every mesh here *and* for a future user-imported mesh (CLAUDE.md's
/// mesh-import punch list item) — that code will just need to call this too.
fn compute_tangents(vertices: &mut [Vertex], indices: &[u32]) {
    let mut accumulated = vec![Vec3::ZERO; vertices.len()];

    for tri in indices.as_chunks::<3>().0 {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = Vec3::from(vertices[i0].position);
        let p1 = Vec3::from(vertices[i1].position);
        let p2 = Vec3::from(vertices[i2].position);
        let uv0 = vertices[i0].uv;
        let uv1 = vertices[i1].uv;
        let uv2 = vertices[i2].uv;

        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let duv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
        let duv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];

        let denom = duv1[0] * duv2[1] - duv2[0] * duv1[1];
        if denom.abs() < 1e-12 {
            // Degenerate UV triangle (e.g. zero UV area) — contributes no
            // usable tangent direction; its vertices still get one from
            // whichever other triangles they're part of.
            continue;
        }
        let r = 1.0 / denom;
        let tangent = (edge1 * duv2[1] - edge2 * duv1[1]) * r;
        if tangent.is_finite() {
            accumulated[i0] += tangent;
            accumulated[i1] += tangent;
            accumulated[i2] += tangent;
        }
    }

    for (vertex, accumulated_tangent) in vertices.iter_mut().zip(accumulated) {
        let normal = Vec3::from(vertex.normal);
        let tangent =
            (accumulated_tangent - normal * normal.dot(accumulated_tangent)).normalize_or_zero();
        let tangent = if tangent != Vec3::ZERO {
            tangent
        } else {
            // No triangle contributed a usable tangent (e.g. a sphere's pole,
            // where every triangle degenerates to a point) — any direction
            // perpendicular to the normal is as good as another there.
            let arbitrary = if normal.x.abs() < 0.9 {
                Vec3::X
            } else {
                Vec3::Y
            };
            (arbitrary - normal * normal.dot(arbitrary)).normalize()
        };
        vertex.tangent = tangent.to_array();
    }
}

#[cfg(test)]
mod tests {
    use super::{generate_rounded_cube, Viewport3d};

    /// Regression test for a reported bug: raising Height Scale on a Cube
    /// or Rounded Cube tore the mesh open at its edges (a visible crack
    /// showing the background through it), and raising Tessellation never
    /// fixed it — because it wasn't a resolution problem. Each of the 6
    /// faces generates its own independent UV-mapped vertex grid; two
    /// faces' vertices at a shared edge sit at the exact same position but
    /// carry different UVs, so height displacement (which reads each
    /// vertex's own UV) pulled them to different final positions. This
    /// asserts the actual fix's own invariant: among every vertex index
    /// the mesh's triangles reference, no two *different* indices ever
    /// point at the same physical position — so there's no shared edge
    /// point left for displacement to disagree about.
    #[test]
    fn rounded_cube_has_no_duplicate_indices_at_shared_edge_positions() {
        for radius in [0.0, 0.25] {
            let (vertices, indices) = generate_rounded_cube(4, radius);
            let key = |p: [f32; 3]| -> (i64, i64, i64) {
                const SCALE: f32 = 1_000_000.0;
                (
                    (p[0] * SCALE).round() as i64,
                    (p[1] * SCALE).round() as i64,
                    (p[2] * SCALE).round() as i64,
                )
            };
            let mut index_by_position: std::collections::HashMap<(i64, i64, i64), u32> =
                std::collections::HashMap::new();
            for &idx in &indices {
                let pos = vertices[idx as usize].position;
                let k = key(pos);
                match index_by_position.get(&k) {
                    Some(&existing) => assert_eq!(
                        existing, idx,
                        "radius {radius}: two different vertex indices ({existing} and {idx}) \
                         reference the same position {pos:?} — a shared cube edge wasn't \
                         welded, which is exactly what tears the mesh open under height \
                         displacement"
                    ),
                    None => {
                        index_by_position.insert(k, idx);
                    }
                }
            }
        }
    }

    /// Regression test for a crash found by manual testing: wiring Gabor
    /// Noise's `raw` output (deliberately `Rgba32Float`, not the usual
    /// `Rgba16Float` every other node output uses — see `gabor_noise.rs`'s
    /// own doc comment on why) into Output: Height and looking at the 3D
    /// viewport crashed with a wgpu validation error ("Texture binding 5
    /// expects sample type Float { filterable: true }, but was given a
    /// view with format Rgba32Float (sample type Float { filterable:
    /// false })"). `Rgba32Float` can't be sampled with filtering without
    /// an opt-in device feature this app doesn't request — fixed by having
    /// `build_texture_bind_group`'s own `channel_view` fall back to the
    /// placeholder texture for any channel whose resolved format isn't
    /// filterable, the same as if nothing were connected, rather than
    /// trying to bind it anyway.
    #[test]
    fn height_wired_to_a_non_filterable_format_falls_back_instead_of_crashing() {
        let mut graph = mc_core::Graph::new();
        let gabor = graph.add_node(mc_core::Node::new("gabor_noise"));
        let output = graph.add_node(mc_core::Node::new("output_height"));
        graph.connect(gabor, "raw", output, "input").unwrap();
        graph.settings.resolution = 64;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        mc_nodes::registry::evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let viewport = Viewport3d::new(&gpu);
        // Would panic inside wgpu validation before the fix.
        let _bind_group = viewport.build_texture_bind_group(&mut gpu, &pool, &graph);
    }
}
