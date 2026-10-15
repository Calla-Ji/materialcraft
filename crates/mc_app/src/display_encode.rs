// SPDX-License-Identifier: MIT OR Apache-2.0
//! Encodes a node's linear-space output texture to sRGB gamma for on-screen
//! display — see `shaders/display_encode.wgsl`'s doc comment for why this
//! has to be a real GPU pass and not just "register the texture as-is."

const NODE_KIND: &str = "display_encode";

/// CLAUDE.md §9's "channel view (RGB / R / G / B / A)" — which channel(s)
/// of a preview to show. Isolating one is the only way to actually judge
/// it on its own merit for a node whose combined RGB reads as an arbitrary
/// color by design (e.g. a packed ORM texture — see
/// `display_encode.wgsl`'s own doc comment).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChannelView {
    #[default]
    Rgb,
    R,
    G,
    B,
    A,
}

impl ChannelView {
    pub const ALL: [ChannelView; 5] = [
        ChannelView::Rgb,
        ChannelView::R,
        ChannelView::G,
        ChannelView::B,
        ChannelView::A,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ChannelView::Rgb => "RGB",
            ChannelView::R => "R",
            ChannelView::G => "G",
            ChannelView::B => "B",
            ChannelView::A => "A",
        }
    }

    fn as_u32(self) -> u32 {
        match self {
            ChannelView::Rgb => 0,
            ChannelView::R => 1,
            ChannelView::G => 2,
            ChannelView::B => 3,
            ChannelView::A => 4,
        }
    }
}

/// Returns a view onto `source_key`'s sRGB-encoded display copy (optionally
/// isolated to one channel — see [`ChannelView`]), creating and/or
/// refreshing it as needed. The copy lives under a derived key in the same
/// pool so it's cheap to look up again (`get_or_create` only reallocates on
/// an actual size change) — called once per visible preview per frame,
/// which is the same cadence egui's own texture registration already runs
/// at, so this doesn't add a new per-frame cost class, just a small compute
/// dispatch alongside an existing one.
pub fn encode_for_display(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    source_key: &str,
    channel: ChannelView,
) -> Option<wgpu::TextureView> {
    let resolution = pool.texture(source_key)?.width();
    let source_view = pool.view(source_key)?.clone();

    let shader_source = include_str!("../shaders/display_encode.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&channel.as_u32().to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("display_encode_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    // Thumbnails (`canvas.rs`) always encode RGB for the same `source_key`
    // a selected node's own (possibly channel-isolated) 2D View preview
    // encodes in the same frame — sharing one key between them would have
    // whichever call ran last silently overwrite the other's result. Only
    // non-RGB modes get their own distinct key, so the common RGB case
    // (thumbnails, and the 2D View itself when no channel is isolated)
    // keeps using exactly the key it always has.
    let dest_key = match channel {
        ChannelView::Rgb => format!("{source_key}__display_srgb"),
        _ => format!("{source_key}__display_srgb_{}", channel.label()),
    };
    let dest_view = pool.get_or_create(
        &gpu.device,
        &dest_key,
        resolution,
        wgpu::TextureFormat::Rgba8Unorm,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("display_encode_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&source_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(dest_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("display_encode_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("display_encode_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));

    pool.view(&dest_key).cloned()
}

#[cfg(test)]
mod tests {
    use super::{encode_for_display, ChannelView};

    const RESOLUTION: u32 = 16;

    /// The standard sRGB EOTF (decode): the inverse of what
    /// `mc_export::linear_to_srgb_u8` and this module's own
    /// `display_encode.wgsl` both encode with.
    fn srgb_u8_to_linear(value: u8) -> f32 {
        let s = value as f32 / 255.0;
        if s <= 0.040_45 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Reproduces the color-picker's own path (CLAUDE.md §8: internal
    /// working space is linear) — a Uniform Color node's stored param is
    /// already linear, decoded from whatever the picker showed. Picking a
    /// mid-gray (sRGB 128/255, the brief's own test case) and running it
    /// through `encode_for_display` should come back out at ~128/255 again,
    /// and — more importantly — match what `mc_export` would write for the
    /// same linear value exactly, since both are meant to implement the
    /// identical sRGB OETF (`display_encode.wgsl`'s `linear_to_srgb` is a
    /// line-for-line copy of `mc_export::linear_to_srgb_u8`'s formula).
    #[test]
    fn mid_gray_round_trips_through_display_encode_and_matches_export() {
        let linear = srgb_u8_to_linear(128);

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        mc_nodes::uniform_color::dispatch(
            &mut gpu,
            &mut pool,
            "image",
            RESOLUTION,
            &mc_nodes::uniform_color::UniformColorParams {
                color: [linear, linear, linear, 1.0],
            },
        );

        let display_view = encode_for_display(&mut gpu, &mut pool, "image", ChannelView::Rgb)
            .expect("image texture exists");
        let display_texture = pool
            .texture("image__display_srgb")
            .expect("display copy exists");
        let _ = display_view;
        let display_pixels = gpu.read_rgba8(display_texture, RESOLUTION);

        let linear_pixels = gpu.read_rgba8(pool.texture("image").expect("image"), RESOLUTION);
        let mut exported_pixels = linear_pixels.clone();
        mc_export::apply_color_space(&mut exported_pixels, mc_export::ColorSpace::Srgb);

        assert_eq!(
            display_pixels, exported_pixels,
            "the 2D View's display encode and mc_export's export encode must produce \
             byte-identical output for the same linear source — otherwise the 2D View and \
             an exported PNG would visibly disagree"
        );

        let px = &display_pixels[0..3];
        for &channel in px {
            assert!(
                channel.abs_diff(128) <= 2,
                "sRGB 128 round-tripped through linear and back should land close to 128, \
                 got {channel}"
            );
        }
    }

    /// An isolated channel is broadcast across R/G/B (so it actually shows
    /// up as a visible grey, not a single colored line) — and picking a
    /// different source channel each time (not just always red-on-red)
    /// actually proves `channel` selects the right one, rather than the
    /// test passing by coincidence if the shader ignored it.
    #[test]
    fn isolating_a_channel_broadcasts_just_that_one_across_rgb() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        mc_nodes::uniform_color::dispatch(
            &mut gpu,
            &mut pool,
            "image",
            RESOLUTION,
            &mc_nodes::uniform_color::UniformColorParams {
                color: [0.1, 0.5, 0.9, 1.0],
            },
        );

        for (channel, linear_source) in [
            (ChannelView::R, 0.1_f32),
            (ChannelView::G, 0.5_f32),
            (ChannelView::B, 0.9_f32),
        ] {
            let _ = encode_for_display(&mut gpu, &mut pool, "image", channel)
                .expect("image texture exists");
            let dest_key = format!("image__display_srgb_{}", channel.label());
            let texture = pool.texture(&dest_key).expect("display copy exists");
            let pixels = gpu.read_rgba8(texture, RESOLUTION);

            let expected = linear_to_srgb_u8_approx(linear_source);
            for px in pixels.as_chunks::<4>().0 {
                assert_eq!(px[0], px[1], "isolated channel should broadcast to G too");
                assert_eq!(px[1], px[2], "isolated channel should broadcast to B too");
                assert!(
                    px[0].abs_diff(expected) <= 2,
                    "channel {channel:?}: expected ~{expected}, got {}",
                    px[0]
                );
                assert_eq!(px[3], 255, "isolated channel view should always be opaque");
            }
        }
    }

    /// Same sRGB OETF `display_encode.wgsl`'s own `linear_to_srgb` uses,
    /// reimplemented in Rust so the test above has an independent
    /// expectation to check the shader's output against.
    fn linear_to_srgb_u8_approx(linear: f32) -> u8 {
        let srgb = if linear <= 0.003_130_8 {
            linear * 12.92
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        };
        (srgb.clamp(0.0, 1.0) * 255.0).round() as u8
    }

    #[test]
    fn isolating_alpha_broadcasts_raw_unencoded_value() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        mc_nodes::uniform_color::dispatch(
            &mut gpu,
            &mut pool,
            "image",
            RESOLUTION,
            &mc_nodes::uniform_color::UniformColorParams {
                color: [0.1, 0.5, 0.9, 0.4],
            },
        );

        let _ = encode_for_display(&mut gpu, &mut pool, "image", ChannelView::A)
            .expect("image texture exists");
        let texture = pool
            .texture("image__display_srgb_A")
            .expect("display copy exists");
        let pixels = gpu.read_rgba8(texture, RESOLUTION);
        for px in pixels.as_chunks::<4>().0 {
            // Raw, not sRGB-encoded — 0.4 * 255 = 102, not the ~180 sRGB
            // encoding 0.4 would produce.
            assert!(px[0].abs_diff(102) <= 2, "got {}", px[0]);
            assert_eq!(px[3], 255, "isolated channel view should always be opaque");
        }
    }
}
