//! Filters / Blending & Masking / Blend.
//!
//! Display name: "Blend". Tooltip: "Combine two inputs with a blend mode,
//! opacity, and optional mask."

use crate::shaders::full_source;

const NODE_KIND: &str = "blend";

/// Blend mode (CLAUDE.md §7). `as_u32`/`NAMES` must stay in sync with the
/// `switch` in `blend.wgsl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Copy,
    Add,
    Subtract,
    Multiply,
    Screen,
    Overlay,
    Max,
    Min,
    Difference,
}

impl BlendMode {
    pub const NAMES: [&'static str; 9] = [
        "Copy",
        "Add",
        "Subtract",
        "Multiply",
        "Screen",
        "Overlay",
        "Max",
        "Min",
        "Difference",
    ];

    fn as_u32(self) -> u32 {
        match self {
            BlendMode::Copy => 0,
            BlendMode::Add => 1,
            BlendMode::Subtract => 2,
            BlendMode::Multiply => 3,
            BlendMode::Screen => 4,
            BlendMode::Overlay => 5,
            BlendMode::Max => 6,
            BlendMode::Min => 7,
            BlendMode::Difference => 8,
        }
    }

    pub fn name(self) -> &'static str {
        Self::NAMES[self.as_u32() as usize]
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::NAMES
            .iter()
            .position(|n| *n == name)
            .map(|i| match i {
                0 => BlendMode::Copy,
                1 => BlendMode::Add,
                2 => BlendMode::Subtract,
                3 => BlendMode::Multiply,
                4 => BlendMode::Screen,
                5 => BlendMode::Overlay,
                6 => BlendMode::Max,
                7 => BlendMode::Min,
                _ => BlendMode::Difference,
            })
    }
}

pub struct BlendParams {
    pub mode: BlendMode,
    pub opacity: f32,
}

impl Default for BlendParams {
    fn default() -> Self {
        Self {
            mode: BlendMode::Copy,
            opacity: 1.0,
        }
    }
}

impl BlendParams {
    /// Reads a graph node's generic param map into typed `BlendParams`,
    /// falling back to defaults for anything missing or the wrong type.
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let mode = params
            .get("mode")
            .and_then(|v| match v {
                mc_core::ParamValue::Enum(name) => BlendMode::from_name(name),
                _ => None,
            })
            .unwrap_or(defaults.mode);
        let opacity = params
            .get("opacity")
            .and_then(mc_core::ParamValue::as_float)
            .unwrap_or(defaults.opacity);
        Self { mode, opacity }
    }
}

/// Evaluate Blend at `resolution x resolution`, reading `base_key` and
/// `blend_key`'s textures from the pool, and `mask_key`'s if given —
/// otherwise a solid-white (fully visible) default, per CLAUDE.md §5's rule
/// that unconnected required inputs fall back to a sensible default. Writes
/// the result into `output_key`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    base_key: &str,
    blend_key: &str,
    mask_key: Option<&str>,
    output_key: &str,
    resolution: u32,
    params: &BlendParams,
) {
    let default_mask_key = format!("{output_key}__default_mask");
    let mask_key = match mask_key {
        Some(key) => key,
        None => {
            gpu.fill_rgba8(pool, &default_mask_key, resolution, [255, 255, 255, 255]);
            &default_mask_key
        }
    };

    let base_view = pool
        .view(base_key)
        .unwrap_or_else(|| panic!("blend: no texture at base key {base_key:?}"))
        .clone();
    let blend_view = pool
        .view(blend_key)
        .unwrap_or_else(|| panic!("blend: no texture at blend key {blend_key:?}"))
        .clone();
    let mask_view = pool
        .view(mask_key)
        .unwrap_or_else(|| panic!("blend: no texture at mask key {mask_key:?}"))
        .clone();

    let shader_source = full_source("blend.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("blend_params"),
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
        label: Some("blend_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&base_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&blend_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&mask_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("blend_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("blend_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

/// Packs params into the exact byte layout of `BlendParams` in `blend.wgsl`
/// (4 x 4-byte fields = 16 bytes; see that file for the struct definition
/// this must stay in sync with).
fn pack_params(resolution: u32, params: &BlendParams) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.mode.as_u32().to_le_bytes());
    bytes[8..12].copy_from_slice(&params.opacity.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, BlendMode, BlendParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &BlendParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "base",
            RESOLUTION,
            &PerlinParams {
                seed: 1,
                ..PerlinParams::default()
            },
        );
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "blend",
            RESOLUTION,
            &PerlinParams {
                seed: 2,
                ..PerlinParams::default()
            },
        );
        dispatch(
            &mut gpu, &mut pool, "base", "blend", None, "output", RESOLUTION, params,
        );
        let texture = pool.texture("output").expect("blend texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_inputs() {
        assert_tiles_seamlessly(&render(&BlendParams::default()), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = BlendParams {
            mode: BlendMode::Multiply,
            opacity: 0.7,
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same inputs + params produced different pixels"
        );
    }

    #[test]
    fn copy_at_full_opacity_outputs_the_blend_layer() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "base",
            RESOLUTION,
            &PerlinParams {
                seed: 1,
                ..PerlinParams::default()
            },
        );
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "blend",
            RESOLUTION,
            &PerlinParams {
                seed: 2,
                ..PerlinParams::default()
            },
        );
        let blend_only = gpu.read_rgba8(pool.texture("blend").unwrap(), RESOLUTION);

        dispatch(
            &mut gpu,
            &mut pool,
            "base",
            "blend",
            None,
            "output",
            RESOLUTION,
            &BlendParams {
                mode: BlendMode::Copy,
                opacity: 1.0,
            },
        );
        let output = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        assert_eq!(
            blend_only, output,
            "Copy mode at opacity 1.0 with no mask must pass the blend layer straight through"
        );
    }

    #[test]
    fn mode_name_round_trips() {
        for name in BlendMode::NAMES {
            let mode = BlendMode::from_name(name).expect("every declared name parses");
            assert_eq!(mode.name(), name);
        }
    }
}
