//! Generators / Patterns / Pattern Generator.
//!
//! Display name: "Pattern Generator". Tooltip: "A brick or grid pattern —
//! use Offset 0.5 for a classic running-bond brick layout."
//!
//! For the brick stagger to tile seamlessly top-to-bottom, `rows` needs to
//! be even — see the shader's doc comment.

use crate::shaders::full_source;

const NODE_KIND: &str = "tile_generator";

pub struct TileGeneratorParams {
    pub rows: u32,
    pub columns: u32,
    pub seed: u32,
    pub offset: f32,
    pub gap: f32,
    /// Shown to the artist as "Edge Darkening" (see `defs.rs`) — a flat
    /// per-tile luminance fade toward the gap, not a height-shaping bevel.
    /// Field/param-key name left as `bevel` for `.mcgraph` compatibility.
    pub bevel: f32,
    pub size_jitter: f32,
    pub rotation_jitter: f32,
    pub luminance_jitter: f32,
}

impl Default for TileGeneratorParams {
    fn default() -> Self {
        Self {
            rows: 6,
            columns: 4,
            seed: 0,
            offset: 0.5,
            gap: 0.06,
            bevel: 0.08,
            size_jitter: 0.08,
            rotation_jitter: 0.0,
            luminance_jitter: 0.15,
        }
    }
}

impl TileGeneratorParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let float = |name: &str, default: f32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(default)
        };
        Self {
            rows: params
                .get("rows")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.rows),
            columns: params
                .get("columns")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.columns),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
            offset: float("offset", defaults.offset),
            gap: float("gap", defaults.gap),
            bevel: float("bevel", defaults.bevel),
            size_jitter: float("size_jitter", defaults.size_jitter),
            rotation_jitter: float("rotation_jitter", defaults.rotation_jitter),
            luminance_jitter: float("luminance_jitter", defaults.luminance_jitter),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &TileGeneratorParams,
) {
    let shader_source = full_source("tile_generator.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tile_generator_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let view = pool.get_or_create(
        &gpu.device,
        key,
        resolution,
        wgpu::TextureFormat::Rgba8Unorm,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tile_generator_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tile_generator_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_generator_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

fn pack_params(resolution: u32, params: &TileGeneratorParams) -> [u8; 48] {
    let mut bytes = [0u8; 48];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.rows.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.columns.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.seed.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.offset.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.gap.to_le_bytes());
    bytes[24..28].copy_from_slice(&params.bevel.to_le_bytes());
    bytes[28..32].copy_from_slice(&params.size_jitter.to_le_bytes());
    bytes[32..36].copy_from_slice(&params.rotation_jitter.to_le_bytes());
    bytes[36..40].copy_from_slice(&params.luminance_jitter.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, TileGeneratorParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &TileGeneratorParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("tile_generator texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    /// Default params use an even row count — see the node's own doc comment
    /// on why that's required for the brick stagger to tile vertically.
    #[test]
    fn tiles_seamlessly_with_default_even_rows() {
        assert_tiles_seamlessly(&render(&TileGeneratorParams::default()), RESOLUTION);
    }

    #[test]
    fn tiles_seamlessly_with_no_stagger() {
        let params = TileGeneratorParams {
            offset: 0.0,
            ..TileGeneratorParams::default()
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = TileGeneratorParams {
            seed: 3,
            ..TileGeneratorParams::default()
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same params produced different pixels"
        );
    }

    #[test]
    fn seed_changes_output() {
        let a = render(&TileGeneratorParams {
            seed: 1,
            ..TileGeneratorParams::default()
        });
        let b = render(&TileGeneratorParams {
            seed: 2,
            ..TileGeneratorParams::default()
        });
        assert_ne!(a, b, "different seeds produced identical pixels");
    }
}
