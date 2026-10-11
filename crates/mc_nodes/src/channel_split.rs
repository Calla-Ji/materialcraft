// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Glitch / Channel Split.
//!
//! Display name: "Channel Split". Tooltip: "Samples each RGB channel at its
//! own offset, for chromatic-aberration/glitch looks."

use crate::shaders::full_source;

const NODE_KIND: &str = "channel_split";

pub struct ChannelSplitParams {
    pub offset_r: [f32; 2],
    pub offset_g: [f32; 2],
    pub offset_b: [f32; 2],
    pub strength: f32,
}

impl Default for ChannelSplitParams {
    fn default() -> Self {
        Self {
            offset_r: [0.01, 0.0],
            offset_g: [0.0, 0.0],
            offset_b: [-0.01, 0.0],
            strength: 1.0,
        }
    }
}

impl ChannelSplitParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            offset_r: params
                .get("offset_r")
                .and_then(mc_core::ParamValue::as_vec2)
                .unwrap_or(defaults.offset_r),
            offset_g: params
                .get("offset_g")
                .and_then(mc_core::ParamValue::as_vec2)
                .unwrap_or(defaults.offset_g),
            offset_b: params
                .get("offset_b")
                .and_then(mc_core::ParamValue::as_vec2)
                .unwrap_or(defaults.offset_b),
            strength: params
                .get("strength")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.strength),
        }
    }
}

fn pack_params(resolution: u32, params: &ChannelSplitParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.strength.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.offset_r[0].to_le_bytes());
    bytes[12..16].copy_from_slice(&params.offset_r[1].to_le_bytes());
    bytes[16..20].copy_from_slice(&params.offset_g[0].to_le_bytes());
    bytes[20..24].copy_from_slice(&params.offset_g[1].to_le_bytes());
    bytes[24..28].copy_from_slice(&params.offset_b[0].to_le_bytes());
    bytes[28..32].copy_from_slice(&params.offset_b[1].to_le_bytes());
    bytes
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    image_key: &str,
    mask_key: Option<&str>,
    output_key: &str,
    resolution: u32,
    params: &ChannelSplitParams,
) {
    let default_mask_key = format!("{output_key}__default_mask");
    let mask_key = match mask_key {
        Some(key) => key,
        None => {
            gpu.fill_rgba8(pool, &default_mask_key, resolution, [255, 255, 255, 255]);
            &default_mask_key
        }
    };

    let image_view = pool
        .view(image_key)
        .unwrap_or_else(|| panic!("channel_split: no texture at image key {image_key:?}"))
        .clone();
    let mask_view = pool
        .view(mask_key)
        .unwrap_or_else(|| panic!("channel_split: no texture at mask key {mask_key:?}"))
        .clone();

    let shader_source = full_source("channel_split.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("channel_split_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let output_view = pool.get_or_create(
        &gpu.device,
        output_key,
        resolution,
        wgpu::TextureFormat::Rgba8Unorm,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("channel_split_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&image_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&mask_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("channel_split_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("channel_split_pass"),
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
    use super::{dispatch, ChannelSplitParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render_from_noise(params: &ChannelSplitParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::perlin::dispatch(
            &mut gpu,
            &mut pool,
            "image",
            RESOLUTION,
            &crate::perlin::PerlinParams::default(),
        );
        dispatch(
            &mut gpu, &mut pool, "image", None, "out", RESOLUTION, params,
        );
        let texture = pool.texture("out").expect("channel split texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn tiles_seamlessly() {
        assert_tiles_seamlessly(
            &render_from_noise(&ChannelSplitParams::default()),
            RESOLUTION,
        );
    }

    #[test]
    fn is_deterministic() {
        let params = ChannelSplitParams::default();
        assert_eq!(render_from_noise(&params), render_from_noise(&params));
    }

    #[test]
    fn zero_offsets_are_a_no_op() {
        let params = ChannelSplitParams {
            offset_r: [0.0, 0.0],
            offset_g: [0.0, 0.0],
            offset_b: [0.0, 0.0],
            strength: 1.0,
        };
        let split = render_from_noise(&params);

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::perlin::dispatch(
            &mut gpu,
            &mut pool,
            "image",
            RESOLUTION,
            &crate::perlin::PerlinParams::default(),
        );
        let original = gpu.read_rgba8(pool.texture("image").expect("image texture"), RESOLUTION);

        assert_eq!(
            split, original,
            "zero offsets must leave the image completely unchanged"
        );
    }

    #[test]
    fn zero_strength_is_also_a_no_op() {
        let params = ChannelSplitParams {
            strength: 0.0,
            ..ChannelSplitParams::default()
        };
        let split = render_from_noise(&params);

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::perlin::dispatch(
            &mut gpu,
            &mut pool,
            "image",
            RESOLUTION,
            &crate::perlin::PerlinParams::default(),
        );
        let original = gpu.read_rgba8(pool.texture("image").expect("image texture"), RESOLUTION);

        assert_eq!(
            split, original,
            "zero strength must leave the image completely unchanged regardless of offsets"
        );
    }

    #[test]
    fn nonzero_offsets_change_the_output() {
        let unsplit = render_from_noise(&ChannelSplitParams {
            offset_r: [0.0, 0.0],
            offset_g: [0.0, 0.0],
            offset_b: [0.0, 0.0],
            strength: 1.0,
        });
        let split = render_from_noise(&ChannelSplitParams::default());
        assert_ne!(unsplit, split);
    }
}
