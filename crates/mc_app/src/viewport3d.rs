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

const VERTEX_ATTRS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeshKind {
    Sphere,
    Cube,
    Plane,
}

impl MeshKind {
    pub const ALL: [MeshKind; 3] = [MeshKind::Sphere, MeshKind::Cube, MeshKind::Plane];

    pub fn label(self) -> &'static str {
        match self {
            MeshKind::Sphere => "Sphere",
            MeshKind::Cube => "Cube",
            MeshKind::Plane => "Plane",
        }
    }
}

/// The 6 PBR maps the shader reads, each falling back to a flat default
/// texture when the graph has no node for that Output kind.
struct FallbackTextures {
    base_color: wgpu::TextureView,
    normal: wgpu::TextureView,
    roughness: wgpu::TextureView,
    metallic: wgpu::TextureView,
    ao: wgpu::TextureView,
    height: wgpu::TextureView,
}

pub struct Viewport3d {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    meshes: HashMap<MeshKind, (wgpu::Buffer, wgpu::Buffer, u32)>,
    fallback: FallbackTextures,
    environment_view: wgpu::TextureView,

    color_view: Option<wgpu::TextureView>,
    depth_view: Option<wgpu::TextureView>,
    texture_id: Option<egui::TextureId>,
    size: (u32, u32),

    pub mesh_kind: MeshKind,
    pub height_scale: f32,
    yaw: f32,
    pitch: f32,
    distance: f32,
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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport3d pipeline"),
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
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Preview-only: not worth getting every mesh's winding
                // order exactly right when there's no performance need to
                // cull.
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewport3d sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let mut meshes = HashMap::new();
        meshes.insert(
            MeshKind::Sphere,
            upload_mesh(device, generate_sphere(24, 48)),
        );
        meshes.insert(MeshKind::Cube, upload_mesh(device, generate_cube()));
        meshes.insert(MeshKind::Plane, upload_mesh(device, generate_plane(64)));

        let fallback = FallbackTextures {
            base_color: solid_view(
                device,
                &gpu.queue,
                "fallback base color",
                [140, 140, 140, 255],
            ),
            normal: solid_view(device, &gpu.queue, "fallback normal", [128, 128, 255, 255]),
            roughness: solid_view(
                device,
                &gpu.queue,
                "fallback roughness",
                [180, 180, 180, 255],
            ),
            metallic: solid_view(device, &gpu.queue, "fallback metallic", [0, 0, 0, 255]),
            ao: solid_view(device, &gpu.queue, "fallback ao", [255, 255, 255, 255]),
            height: solid_view(device, &gpu.queue, "fallback height", [128, 128, 128, 255]),
        };

        let environment_view = crate::hdri::load_environment_view(device, &gpu.queue);

        Self {
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            texture_bind_group_layout,
            sampler,
            meshes,
            fallback,
            environment_view,
            color_view: None,
            depth_view: None,
            texture_id: None,
            size: (0, 0),
            mesh_kind: MeshKind::Sphere,
            height_scale: 0.15,
            yaw: 0.6,
            pitch: 0.35,
            distance: 3.0,
        }
    }

    /// Orbit (drag) and zoom (scroll, while hovered) camera controls.
    pub fn handle_input(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if response.dragged() {
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

    fn eye_position(&self) -> Vec3 {
        Vec3::new(
            self.distance * self.pitch.cos() * self.yaw.sin(),
            self.distance * self.pitch.sin(),
            self.distance * self.pitch.cos() * self.yaw.cos(),
        )
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
        device: &wgpu::Device,
        pool: &mc_gpu::TexturePool,
        graph: &mc_core::Graph,
    ) -> wgpu::BindGroup {
        let base_color =
            resolve_channel(graph, pool, "output_base_color").unwrap_or(&self.fallback.base_color);
        let normal = resolve_channel(graph, pool, "output_normal").unwrap_or(&self.fallback.normal);
        let roughness =
            resolve_channel(graph, pool, "output_roughness").unwrap_or(&self.fallback.roughness);
        let metallic =
            resolve_channel(graph, pool, "output_metallic").unwrap_or(&self.fallback.metallic);
        let ao =
            resolve_channel(graph, pool, "output_ambient_occlusion").unwrap_or(&self.fallback.ao);
        let height = resolve_channel(graph, pool, "output_height").unwrap_or(&self.fallback.height);

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport3d textures bind group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(base_color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normal),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(roughness),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(metallic),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(ao),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(height),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&self.environment_view),
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
        gpu: &mc_gpu::Context,
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

        let aspect = width as f32 / height as f32;
        let eye = self.eye_position();
        let view = glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y);
        let proj =
            glam::camera::rh::proj::directx::perspective(60f32.to_radians(), aspect, 0.1, 100.0);
        let light_dir = Vec3::new(-0.4, -1.0, -0.3).normalize();
        let use_displacement = self.mesh_kind == MeshKind::Plane;

        let bytes = uniform_bytes(
            proj * view,
            eye,
            light_dir,
            self.height_scale,
            use_displacement,
        );
        gpu.queue.write_buffer(&self.uniform_buffer, 0, &bytes);

        let texture_bind_group = self.build_texture_bind_group(&gpu.device, pool, graph);
        let (vertex_buffer, index_buffer, index_count) = self
            .meshes
            .get(&self.mesh_kind)
            .expect("every MeshKind has a mesh built in Viewport3d::new");

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
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.07,
                            a: 1.0,
                        }),
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

            pass.set_pipeline(&self.pipeline);
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

/// The graph's texture for one Output kind (e.g. `"output_base_color"`), if
/// the graph has that Output node and it resolves to something. Mirrors
/// `mc_nodes::export::read_output`'s lookup, minus the readback.
fn resolve_channel<'a>(
    graph: &mc_core::Graph,
    pool: &'a mc_gpu::TexturePool,
    kind: &str,
) -> Option<&'a wgpu::TextureView> {
    let id = graph
        .nodes()
        .find(|(_, n)| n.kind == kind)
        .map(|(id, _)| id)?;
    let key = mc_nodes::registry::resolve_texture_key(graph, id)?;
    pool.view(&key)
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

/// `view_proj`(mat4) + `camera_pos`(vec4) + `light_dir`(vec4) + 2 scalars +
/// 2 padding floats = 112 bytes, matching `Uniforms` in viewport3d.wgsl.
const UNIFORM_SIZE: usize = 112;

fn uniform_bytes(
    view_proj: Mat4,
    camera_pos: Vec3,
    light_dir: Vec3,
    height_scale: f32,
    use_displacement: bool,
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
    put(height_scale, &mut bytes, &mut offset);
    bytes[offset..offset + 4].copy_from_slice(&(use_displacement as u32).to_le_bytes());
    offset += 4;
    put(0.0, &mut bytes, &mut offset);
    put(0.0, &mut bytes, &mut offset);
    bytes
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
            vertices.push(Vertex {
                position,
                normal: position,
                uv: [u, v],
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
    (vertices, indices)
}

fn generate_cube() -> (Vec<Vertex>, Vec<u32>) {
    // (face normal, right axis, up axis) for each of the 6 faces.
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
        for (du, dv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let position = normal + right * du + up * dv;
            let uv = [du * 0.5 + 0.5, 1.0 - (dv * 0.5 + 0.5)];
            vertices.push(Vertex {
                position: position.to_array(),
                normal: normal.to_array(),
                uv,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (vertices, indices)
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
    (vertices, indices)
}
