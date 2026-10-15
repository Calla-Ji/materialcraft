// SPDX-License-Identifier: MIT OR Apache-2.0
//! Converters / Channel Packer.
//!
//! Display name: "Channel Packer". Tooltip: "Pack three grayscale inputs
//! into one texture's R/G/B channels — e.g. Ambient Occlusion, Roughness,
//! and Metallic into an ORM map for Unreal Engine."
//!
//! No params — every "setting" this node has is which textures are wired
//! into its three inputs. Each is independently optional: an unconnected
//! input falls back to a flat default (CLAUDE.md §5's rule), chosen per
//! channel to match the Unreal Engine 5 export preset's own ORM fallbacks
//! exactly (`mc_nodes::export`'s `resolve_or_fallback`) — white for R (no
//! occlusion), mid-grey for G (a neutral roughness), black for B (fully
//! non-metal) — so a graph gets the identical result whether ORM packing
//! happens explicitly via this node or implicitly at export time.

use crate::shaders::full_source;

const NODE_KIND: &str = "channel_pack";

fn pack_params(resolution: u32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes
}

/// Evaluate Channel Packer at `resolution x resolution`, reading
/// `r_key`/`g_key`/`b_key`'s textures from the pool — each `None` falls
/// back to a flat default matching this node's own doc comment above.
/// Writes the result into `output_key`.
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    r_key: Option<&str>,
    g_key: Option<&str>,
    b_key: Option<&str>,
    output_key: &str,
    resolution: u32,
) {
    let default_r_key = format!("{output_key}__default_r");
    let default_g_key = format!("{output_key}__default_g");
    let default_b_key = format!("{output_key}__default_b");

    let r_key = match r_key {
        Some(key) => key,
        None => {
            gpu.fill_rgba8(pool, &default_r_key, resolution, [255, 255, 255, 255]);
            &default_r_key
        }
    };
    let g_key = match g_key {
        Some(key) => key,
        None => {
            gpu.fill_rgba8(pool, &default_g_key, resolution, [128, 128, 128, 255]);
            &default_g_key
        }
    };
    let b_key = match b_key {
        Some(key) => key,
        None => {
            gpu.fill_rgba8(pool, &default_b_key, resolution, [0, 0, 0, 255]);
            &default_b_key
        }
    };

    let r_view = pool
        .view(r_key)
        .unwrap_or_else(|| panic!("channel_pack: no texture at r key {r_key:?}"))
        .clone();
    let g_view = pool
        .view(g_key)
        .unwrap_or_else(|| panic!("channel_pack: no texture at g key {g_key:?}"))
        .clone();
    let b_view = pool
        .view(b_key)
        .unwrap_or_else(|| panic!("channel_pack: no texture at b key {b_key:?}"))
        .clone();

    let shader_source = full_source("channel_pack.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("channel_pack_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let view = pool.get_or_create(
        &gpu.device,
        output_key,
        resolution,
        wgpu::TextureFormat::Rgba16Float,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("channel_pack_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&r_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&g_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&b_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("channel_pack_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("channel_pack_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

#[cfg(test)]
mod tests {
    use super::dispatch;
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn solid(gpu: &mut mc_gpu::Context, pool: &mut mc_gpu::TexturePool, key: &str, value: u8) {
        gpu.fill_rgba8(pool, key, RESOLUTION, [value, value, value, 255]);
    }

    fn render(
        r: Option<u8>,
        g: Option<u8>,
        b: Option<u8>,
    ) -> (mc_gpu::Context, mc_gpu::TexturePool, Vec<u8>) {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        if let Some(v) = r {
            solid(&mut gpu, &mut pool, "r", v);
        }
        if let Some(v) = g {
            solid(&mut gpu, &mut pool, "g", v);
        }
        if let Some(v) = b {
            solid(&mut gpu, &mut pool, "b", v);
        }
        dispatch(
            &mut gpu,
            &mut pool,
            r.map(|_| "r"),
            g.map(|_| "g"),
            b.map(|_| "b"),
            "output",
            RESOLUTION,
        );
        let texture = pool.texture("output").expect("channel_pack texture");
        let pixels = gpu.read_rgba8(texture, RESOLUTION);
        (gpu, pool, pixels)
    }

    #[test]
    fn packs_each_inputs_red_channel_into_its_own_output_channel() {
        let (_, _, pixels) = render(Some(10), Some(20), Some(30));
        assert_eq!(&pixels[0..4], &[10, 20, 30, 255]);
    }

    #[test]
    fn unconnected_inputs_fall_back_to_the_orm_defaults() {
        let (_, _, pixels) = render(None, None, None);
        assert_eq!(
            &pixels[0..4],
            &[255, 128, 0, 255],
            "R should default to white, G to mid-grey, B to black"
        );
    }

    #[test]
    fn tiles_seamlessly() {
        let (_, _, pixels) = render(Some(10), Some(20), Some(30));
        assert_tiles_seamlessly(&pixels, RESOLUTION);
    }

    /// Hard rule #3 (CLAUDE.md §6): same graph + same seed = bit-identical
    /// output on the same GPU. No seed here (no randomness at all), but the
    /// same invariant still applies: identical inputs, identical output.
    #[test]
    fn is_deterministic() {
        let (_, _, a) = render(Some(10), Some(20), Some(30));
        let (_, _, b) = render(Some(10), Some(20), Some(30));
        assert_eq!(a, b, "same inputs produced different pixels");
    }
}
