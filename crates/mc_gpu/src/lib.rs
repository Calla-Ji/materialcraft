// SPDX-License-Identifier: MIT OR Apache-2.0
//! mc_gpu: shared wgpu device/queue, a texture pool, and a compute pipeline cache.
//!
//! Blender/Unreal analogy: this is the "render engine" plumbing underneath the
//! node graph. Nodes never talk to the GPU directly — they ask a [`Context`]
//! to compile their WGSL compute shader and dispatch it, and get a texture
//! back. `mc_app` shares eframe's own wgpu device with this crate (so the UI
//! and node textures live on one GPU); `mc_cli` and tests create a standalone
//! headless device instead.

use std::collections::HashMap;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};

use anyhow::{Context as _, Result};

/// A GPU device/queue pair, plus the compute pipelines compiled from it.
pub struct Context {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipelines: HashMap<(String, u64), Arc<wgpu::ComputePipeline>>,
}

impl Context {
    /// Wrap an existing device/queue, e.g. eframe's wgpu render state.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            pipelines: HashMap::new(),
        }
    }

    /// Create a standalone device/queue with no window. Used by `mc_cli` and
    /// tests, which never open an eframe window.
    pub fn new_headless() -> Result<Self> {
        block_on(Self::new_headless_async())
    }

    async fn new_headless_async() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .context("no suitable GPU adapter found")?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("mc_gpu headless device"),
                ..Default::default()
            })
            .await
            .context("failed to create GPU device")?;
        Ok(Self::new(device, queue))
    }

    /// Get or compile the compute pipeline for a node's shader source.
    ///
    /// The cache key includes a hash of `wgsl_source`, so editing a shader
    /// file on disk (hot-reload, debug builds — see `mc_nodes::shader_source`)
    /// transparently recompiles it the next time this is called.
    pub fn compute_pipeline(
        &mut self,
        node_kind: &str,
        wgsl_source: &str,
        entry_point: &str,
    ) -> Arc<wgpu::ComputePipeline> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        wgsl_source.hash(&mut hasher);
        let cache_key = (node_kind.to_string(), hasher.finish());

        if let Some(pipeline) = self.pipelines.get(&cache_key) {
            return pipeline.clone();
        }

        // Drop any stale pipeline(s) for this node kind (e.g. the pre-edit shader).
        self.pipelines.retain(|(kind, _), _| kind != node_kind);

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(node_kind),
                source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
            });
        let pipeline = Arc::new(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(node_kind),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));

        self.pipelines.insert(cache_key, pipeline.clone());
        pipeline
    }

    /// Fill `key`'s texture in `pool` with a single solid RGBA color (0-255
    /// per channel), creating it at `size x size` if needed. Used for a
    /// node's optional inputs that fall back to a flat default (e.g. Blend's
    /// mask defaulting to solid white) rather than running a shader over
    /// them.
    pub fn fill_rgba8(&self, pool: &mut TexturePool, key: &str, size: u32, color: [u8; 4]) {
        pool.get_or_create(&self.device, key, size, wgpu::TextureFormat::Rgba8Unorm);
        let texture = pool.texture(key).expect("just created above");

        let pixels: Vec<u8> = color.repeat((size * size) as usize);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Read a square `Rgba8Unorm` texture back to a tightly-packed `Vec<u8>`
    /// of `resolution * resolution * 4` bytes (one row after another, no
    /// padding) — used for PNG export and for tests that inspect pixels.
    pub fn read_rgba8(&self, texture: &wgpu::Texture, resolution: u32) -> Vec<u8> {
        const BYTES_PER_PIXEL: u32 = 4;
        let unpadded_bytes_per_row = resolution * BYTES_PER_PIXEL;
        let padded_bytes_per_row = unpadded_bytes_per_row
            .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc_gpu readback buffer"),
            size: (padded_bytes_per_row * resolution) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mc_gpu readback encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(resolution),
                },
            },
            wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        rx.recv()
            .expect("map_async callback never fired")
            .expect("failed to map readback buffer");

        let mapped = slice
            .get_mapped_range()
            .expect("buffer was just mapped successfully");
        let mut pixels = Vec::with_capacity((resolution * unpadded_bytes_per_row) as usize);
        for row in 0..resolution {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + unpadded_bytes_per_row as usize;
            pixels.extend_from_slice(&mapped[start..end]);
        }
        drop(mapped);
        buffer.unmap();
        pixels
    }
}

/// Caches node output textures by a caller-chosen key (typically the node id),
/// so re-evaluating the same node reuses its texture instead of reallocating.
#[derive(Default)]
pub struct TexturePool {
    textures: HashMap<String, (wgpu::Texture, wgpu::TextureView)>,
}

impl TexturePool {
    /// Get the texture view for `key`, creating (or recreating, if the
    /// requested size/format changed) a square `size x size` storage+sampled
    /// texture as needed.
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        key: &str,
        size: u32,
        format: wgpu::TextureFormat,
    ) -> &wgpu::TextureView {
        let needs_create = match self.textures.get(key) {
            Some((texture, _)) => texture.width() != size || texture.format() != format,
            None => true,
        };
        if needs_create {
            // Every pool texture gets a full mip chain allocated, even
            // though only the 6 PBR channel textures the 3D viewport reads
            // (`mc_app::mipgen`) ever actually have mips beyond 0 computed —
            // the extra levels just sit unused (and unsampled, since nothing
            // ever binds a full-mip-range view of an intermediate node's
            // texture) for any other node. Simpler than a second texture
            // path just for "the 6 that matter," at the cost of ~33% more
            // VRAM per texture, which isn't a concern at this project's
            // scale.
            let mip_level_count = size.max(1).ilog2() + 1;
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(key),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            // Pinned to mip 0 only, regardless of how many mips the texture
            // actually has: every existing caller (compute-shader storage
            // write targets, `textureLoad` reads, the 2D View/thumbnails)
            // expects exactly mip 0, and a storage-texture binding requires
            // a single-mip view anyway — wgpu rejects a multi-mip view
            // there. `mc_app::mipgen::full_mip_view` creates the
            // all-mips view the 3D viewport's sampler actually needs,
            // separately, on demand.
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                base_mip_level: 0,
                mip_level_count: Some(1),
                ..Default::default()
            });
            self.textures.insert(key.to_string(), (texture, view));
        }
        &self
            .textures
            .get(key)
            .expect("just inserted or already present")
            .1
    }

    /// The underlying texture for `key`, e.g. for a `copy_texture_to_buffer` readback.
    pub fn texture(&self, key: &str) -> Option<&wgpu::Texture> {
        self.textures.get(key).map(|(texture, _)| texture)
    }

    /// The view for `key`, e.g. to register with egui for display.
    pub fn view(&self, key: &str) -> Option<&wgpu::TextureView> {
        self.textures.get(key).map(|(_, view)| view)
    }
}

/// A tiny single-future executor.
///
/// wgpu's native `request_adapter`/`request_device` resolve immediately on
/// desktop (the `Future` wrapper exists only for API parity with the browser,
/// where it genuinely awaits). This avoids pulling in an async runtime crate
/// just to unwrap that; `Waker::noop()` is stable in std.
fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = Box::pin(fut);
    let waker = std::task::Waker::noop();
    let mut cx = TaskContext::from_waker(waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}
