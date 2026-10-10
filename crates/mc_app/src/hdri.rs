//! Loads the bundled CC0 HDRI (see ASSETS.md for source/license) as an
//! equirectangular environment texture, so the 3D viewport's sky and
//! reflections have real detail to show off Metallic/Roughness instead of a
//! flat two-color gradient.
//!
//! Embedded via `include_bytes!` (not read from disk at runtime) so the app
//! works regardless of the working directory `cargo run`/the built binary is
//! launched from.

use std::io::Cursor;

use exr::prelude::{ReadChannels, ReadLayers};

const HDRI_BYTES: &[u8] = include_bytes!("../../../assets/hdri/studio_small_03_1k.exr");

/// IEEE 754 half-float bit pattern for `value`. wgpu's `Rgba16Float` is
/// filterable on every backend with no extra device feature (unlike
/// `Rgba32Float`, which needs `FLOAT32_FILTERABLE`), so the HDRI is converted
/// down to half floats here rather than uploaded as-is. Flushes very small
/// values to zero and saturates very bright ones (e.g. a sun disc) to
/// infinity instead of handling denormals — fine for a lighting texture.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x007f_ffff;

    if exponent <= 0 {
        sign
    } else if exponent >= 0x1f {
        sign | 0x7c00
    } else {
        sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
    }
}

/// Decodes the embedded HDRI and uploads it as an `Rgba16Float` texture.
pub fn load_environment_view(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let image = exr::prelude::read()
        .no_deep_data()
        .largest_resolution_level()
        .rgba_channels(
            |resolution, _channels| {
                (
                    resolution.width(),
                    resolution.height(),
                    vec![0f32; resolution.width() * resolution.height() * 4],
                )
            },
            |(width, _height, pixels): &mut (usize, usize, Vec<f32>),
             position,
             (r, g, b, a): (f32, f32, f32, f32)| {
                let index = (position.1 * *width + position.0) * 4;
                pixels[index] = r;
                pixels[index + 1] = g;
                pixels[index + 2] = b;
                pixels[index + 3] = a;
            },
        )
        .first_valid_layer()
        .all_attributes()
        .from_buffered(Cursor::new(HDRI_BYTES))
        .expect("bundled HDRI (assets/hdri/studio_small_03_1k.exr) is a valid EXR file");

    let (width, height, pixels) = image.layer_data.channel_data.pixels;
    let width = width as u32;
    let height = height as u32;

    let mut bytes = Vec::with_capacity(pixels.len() * 2);
    for value in &pixels {
        bytes.extend_from_slice(&f32_to_f16_bits(*value).to_le_bytes());
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viewport3d environment map"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
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
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 8),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
