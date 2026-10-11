// SPDX-License-Identifier: MIT OR Apache-2.0
//! Loads equirectangular HDRI environment maps for the 3D viewport's sky and
//! reflections: a handful of bundled CC0 presets (see ASSETS.md for each
//! one's source/license) plus an arbitrary user-supplied `.hdr`/`.exr` file,
//! both decoded through the same path so a custom upload gets identical
//! treatment to a bundled preset.
//!
//! Presets are embedded via `include_bytes!` (not read from disk at
//! runtime) so the app works regardless of the working directory
//! `cargo run`/the built binary is launched from.

use std::io::Cursor;
use std::path::Path;

use exr::prelude::{ReadChannels, ReadLayers};
use image::ImageDecoder;

/// A bundled CC0 HDRI the user can pick from the Environment panel, in
/// addition to loading their own file. See ASSETS.md for source/license on
/// each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdriPreset {
    StudioSmall03,
    VeniceSunset,
    SyferfonteinClear,
    RoglandNight,
}

impl HdriPreset {
    pub const ALL: [HdriPreset; 4] = [
        HdriPreset::StudioSmall03,
        HdriPreset::VeniceSunset,
        HdriPreset::SyferfonteinClear,
        HdriPreset::RoglandNight,
    ];

    pub fn label(self) -> &'static str {
        match self {
            HdriPreset::StudioSmall03 => "Studio Small 03",
            HdriPreset::VeniceSunset => "Venice Sunset",
            HdriPreset::SyferfonteinClear => "Syferfontein 18d Clear",
            HdriPreset::RoglandNight => "Rogland Clear Night",
        }
    }

    fn bytes(self) -> &'static [u8] {
        match self {
            HdriPreset::StudioSmall03 => {
                include_bytes!("../../../assets/hdri/studio_small_03_1k.exr")
            }
            HdriPreset::VeniceSunset => {
                include_bytes!("../../../assets/hdri/venice_sunset_1k.exr")
            }
            HdriPreset::SyferfonteinClear => {
                include_bytes!("../../../assets/hdri/syferfontein_18d_clear_1k.exr")
            }
            // .hdr, not .exr like the others — Poly Haven's EXR encode of
            // this particular HDRI happens to come out ~4x larger than its
            // Radiance one (unusual; most assets go the other way), so this
            // one keeps the repo smaller by using the format `load_custom`
            // already has to support anyway for user uploads.
            HdriPreset::RoglandNight => {
                include_bytes!("../../../assets/hdri/rogland_clear_night_1k.hdr")
            }
        }
    }

    fn format(self) -> HdriFormat {
        match self {
            HdriPreset::RoglandNight => HdriFormat::Radiance,
            _ => HdriFormat::OpenExr,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HdriFormat {
    OpenExr,
    Radiance,
}

impl HdriFormat {
    /// Guesses the format from a file's extension — the only format tag a
    /// user-supplied file comes with.
    fn from_path(path: &Path) -> Result<Self, String> {
        match path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("exr") => Ok(Self::OpenExr),
            Some("hdr") => Ok(Self::Radiance),
            Some(other) => Err(format!(
                "unsupported HDRI file extension \".{other}\" (expected .hdr or .exr)"
            )),
            None => Err("HDRI file has no extension (expected .hdr or .exr)".to_string()),
        }
    }
}

/// IEEE 754 half-float bit pattern for `value`. wgpu's `Rgba16Float` is
/// filterable on every backend with no extra device feature (unlike
/// `Rgba32Float`, which needs `FLOAT32_FILTERABLE`), so an HDRI is converted
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

/// Solid-angle-weighted average color of an equirectangular environment map,
/// used as a flat ambient/irradiance stand-in (see `viewport3d.wgsl`'s
/// `ambient_color` uniform): a per-pixel raw sample of the HDRI looks like a
/// sharp reflection even on the diffuse term, since equirect rows near the
/// poles cover much less actual sky than rows near the equator, a plain
/// unweighted average would be skewed toward the poles.
fn weighted_average_color(width: usize, height: usize, pixels: &[f32]) -> [f32; 3] {
    let mut sum = [0f64; 3];
    let mut weight_total = 0f64;
    for row in 0..height {
        // `v` mirrors `equirect_uv`'s mapping (v=0 at the +Y pole, v=1 at
        // -Y), so `v * PI` is the polar angle and `sin` of it is the
        // standard equirect row weight.
        let v = (row as f64 + 0.5) / height as f64;
        let weight = (v * std::f64::consts::PI).sin();
        for col in 0..width {
            let index = (row * width + col) * 4;
            sum[0] += pixels[index] as f64 * weight;
            sum[1] += pixels[index + 1] as f64 * weight;
            sum[2] += pixels[index + 2] as f64 * weight;
        }
        weight_total += weight * width as f64;
    }
    [
        (sum[0] / weight_total) as f32,
        (sum[1] / weight_total) as f32,
        (sum[2] / weight_total) as f32,
    ]
}

/// Decodes an OpenEXR equirectangular map into interleaved `(width, height,
/// rgba_f32)`.
fn decode_exr(bytes: &[u8]) -> Result<(u32, u32, Vec<f32>), String> {
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
        .from_buffered(Cursor::new(bytes))
        .map_err(|err| format!("couldn't read EXR: {err}"))?;

    let (width, height, pixels) = image.layer_data.channel_data.pixels;
    Ok((width as u32, height as u32, pixels))
}

/// Decodes a Radiance (`.hdr`/`.pic`) equirectangular map into interleaved
/// `(width, height, rgba_f32)` — Radiance has no alpha channel, so every
/// pixel gets alpha = 1.
fn decode_radiance(bytes: &[u8]) -> Result<(u32, u32, Vec<f32>), String> {
    let decoder = image::codecs::hdr::HdrDecoder::new(Cursor::new(bytes))
        .map_err(|err| format!("couldn't read Radiance HDR: {err}"))?;
    let (width, height) = decoder.dimensions();
    let mut raw = vec![0u8; decoder.total_bytes() as usize];
    decoder
        .read_image(&mut raw)
        .map_err(|err| format!("couldn't decode Radiance HDR: {err}"))?;

    let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
    let (rgb_chunks, _) = raw.as_chunks::<12>();
    for rgb in rgb_chunks {
        pixels.push(f32::from_ne_bytes([rgb[0], rgb[1], rgb[2], rgb[3]]));
        pixels.push(f32::from_ne_bytes([rgb[4], rgb[5], rgb[6], rgb[7]]));
        pixels.push(f32::from_ne_bytes([rgb[8], rgb[9], rgb[10], rgb[11]]));
        pixels.push(1.0);
    }
    Ok((width, height, pixels))
}

fn decode(bytes: &[u8], format: HdriFormat) -> Result<(u32, u32, Vec<f32>), String> {
    match format {
        HdriFormat::OpenExr => decode_exr(bytes),
        HdriFormat::Radiance => decode_radiance(bytes),
    }
}

/// Uploads a decoded equirectangular map as an `Rgba16Float` texture,
/// alongside its solid-angle-weighted average color (see
/// [`weighted_average_color`]).
fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    pixels: &[f32],
) -> (wgpu::TextureView, [f32; 3]) {
    let average_color = weighted_average_color(width as usize, height as usize, pixels);

    let mut bytes = Vec::with_capacity(pixels.len() * 2);
    for value in pixels {
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

    (
        texture.create_view(&wgpu::TextureViewDescriptor::default()),
        average_color,
    )
}

const THUMBNAIL_WIDTH: u32 = 128;
const THUMBNAIL_HEIGHT: u32 = 64;

/// Renders a small sRGB-encoded preview of an environment map — see
/// `hdri_thumbnail.wgsl`'s doc comment for why this needs its own tonemap
/// pass rather than reusing `display_encode`. Returns a plain
/// `Rgba8Unorm` texture the caller can register as an egui texture
/// directly, with no further encoding needed.
pub fn render_thumbnail(
    gpu: &mut mc_gpu::Context,
    environment_view: &wgpu::TextureView,
) -> wgpu::TextureView {
    let shader_source = include_str!("../shaders/hdri_thumbnail.wgsl");
    let pipeline = gpu.compute_pipeline("hdri_thumbnail", shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&THUMBNAIL_WIDTH.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&THUMBNAIL_HEIGHT.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hdri_thumbnail_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    // Wraps horizontally (matching `equirect_uv`'s own addressing, so the
    // thumbnail has no seam at the left/right edge) but clamps vertically —
    // the poles aren't periodic.
    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("hdri_thumbnail_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let output_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hdri_thumbnail"),
        size: wgpu::Extent3d {
            width: THUMBNAIL_WIDTH,
            height: THUMBNAIL_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hdri_thumbnail_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(environment_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&output_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hdri_thumbnail_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hdri_thumbnail_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(THUMBNAIL_WIDTH.div_ceil(8), THUMBNAIL_HEIGHT.div_ceil(8), 1);
    }
    gpu.queue.submit(Some(encoder.finish()));

    output_view
}

/// Decodes and uploads a bundled preset. The bundled bytes are checked into
/// this repo and covered by a decode test (see `tests` below), so a failure
/// here means a corrupted build, not bad user input — a panic is correct.
pub fn load_preset(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    preset: HdriPreset,
) -> (wgpu::TextureView, [f32; 3]) {
    let (width, height, pixels) = decode(preset.bytes(), preset.format())
        .unwrap_or_else(|err| panic!("bundled HDRI preset {preset:?} is invalid: {err}"));
    upload(device, queue, width, height, &pixels)
}

/// Decodes and uploads a user-supplied `.hdr`/`.exr` file. Unlike
/// [`load_preset`], every failure mode here is reachable from an arbitrary
/// file the user picked, so nothing panics — the caller is expected to
/// surface `Err` as a toast rather than crash the app.
pub fn load_custom(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &Path,
) -> Result<(wgpu::TextureView, [f32; 3]), String> {
    let format = HdriFormat::from_path(path)?;
    let bytes =
        std::fs::read(path).map_err(|err| format!("couldn't read {}: {err}", path.display()))?;
    let (width, height, pixels) = decode(&bytes, format)?;
    Ok(upload(device, queue, width, height, &pixels))
}

#[cfg(test)]
mod tests {
    use super::{decode, HdriPreset};

    /// Every bundled preset must actually decode — a corrupted or
    /// wrong-format asset file would otherwise only surface as a runtime
    /// panic the first time someone picks it in the UI.
    #[test]
    fn every_bundled_preset_decodes() {
        for preset in HdriPreset::ALL {
            let (width, height, pixels) = decode(preset.bytes(), preset.format())
                .unwrap_or_else(|err| panic!("{preset:?} failed to decode: {err}"));
            assert!(width > 0 && height > 0, "{preset:?} has zero size");
            assert_eq!(
                pixels.len(),
                width as usize * height as usize * 4,
                "{preset:?} pixel count doesn't match its reported dimensions"
            );
        }
    }
}
