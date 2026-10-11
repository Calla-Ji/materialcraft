// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared helpers for the Dither family (Dither Pattern, Ordered Dither,
//! Palette Dither) — the bundled CC0 blue noise mask and the boilerplate to
//! get it onto the GPU, which all three need identically (see ASSETS.md).

pub(crate) const BLUE_NOISE_BYTES: &[u8] =
    include_bytes!("../../../assets/dither/blue_noise_64.png");
pub(crate) const BLUE_NOISE_SIZE: u32 = 64;

/// Uploads the bundled blue noise mask into `pool` under a fixed reserved
/// key (shared across every caller, so they all reuse the same GPU texture
/// rather than each keeping their own copy), returning an owned clone of its
/// view ready to bind. Re-uploads the pixel data every call rather than
/// tracking "already uploaded" state — the texture is tiny (64x64x4 bytes)
/// and a caller's `dispatch` only runs when it's actually dirty (CLAUDE.md
/// §5's cache), not every frame.
pub(crate) fn upload_blue_noise_mask(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
) -> wgpu::TextureView {
    let (pixels, width, height) = mc_export::decode_png_rgba8(BLUE_NOISE_BYTES)
        .expect("bundled blue noise asset (assets/dither/blue_noise_64.png) is a valid PNG");
    debug_assert_eq!((width, height), (BLUE_NOISE_SIZE, BLUE_NOISE_SIZE));

    let view = pool
        .get_or_create(
            &gpu.device,
            "__builtin_blue_noise_64",
            BLUE_NOISE_SIZE,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .clone();
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: pool
                .texture("__builtin_blue_noise_64")
                .expect("just created above"),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    view
}

/// Rounds `value` up to the nearest multiple of `unit` (minimum `unit`
/// itself) — every Dither-family "Cells" param needs this so its chosen
/// pattern (Bayer's power-of-two matrix, or the 64x64 blue noise mask)
/// divides evenly into the texture and wraps seamlessly.
pub(crate) fn round_up_to_multiple(value: i32, unit: i32) -> i32 {
    let value = value.max(1);
    ((value + unit - 1) / unit) * unit
}
