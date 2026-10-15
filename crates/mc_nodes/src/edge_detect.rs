// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Spatial / Edge Detect.
//!
//! Display name: "Edge Detect". Tooltip: "Highlights where the input changes
//! sharply — bright on edges, black over flat areas."

use crate::shaders::full_source;
use crate::tiling::TilingMode;

const NODE_KIND: &str = "edge_detect";

pub struct EdgeDetectParams {
    pub intensity: f32,
    pub tiling_mode: TilingMode,
}

impl Default for EdgeDetectParams {
    fn default() -> Self {
        Self {
            intensity: 4.0,
            tiling_mode: TilingMode::default(),
        }
    }
}

impl EdgeDetectParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            intensity: params
                .get("intensity")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.intensity),
            tiling_mode: params
                .get("tiling_mode")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => TilingMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.tiling_mode),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &EdgeDetectParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("edge_detect: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("edge_detect.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&params.intensity.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&params.tiling_mode.as_u32().to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("edge_detect_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let output_view = pool.get_or_create(
        &gpu.device,
        output_key,
        resolution,
        wgpu::TextureFormat::Rgba16Float,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("edge_detect_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&input_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("edge_detect_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("edge_detect_pass"),
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
    use super::{dispatch, EdgeDetectParams};
    use crate::checker::{dispatch as checker_dispatch, CheckerParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};
    use crate::tiling::TilingMode;
    use crate::uniform_color::{dispatch as uniform_color_dispatch, UniformColorParams};

    fn render_perlin(params: &EdgeDetectParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams::default(),
        );
        dispatch(&mut gpu, &mut pool, "input", "output", RESOLUTION, params);
        let texture = pool.texture("output").expect("edge_detect texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(&render_perlin(&EdgeDetectParams::default()), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = EdgeDetectParams {
            intensity: 6.0,
            tiling_mode: TilingMode::Both,
        };
        assert_eq!(render_perlin(&params), render_perlin(&params));
    }

    #[test]
    fn flat_input_has_no_edges() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        uniform_color_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &UniformColorParams {
                color: [0.5, 0.5, 0.5, 1.0],
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output",
            RESOLUTION,
            &EdgeDetectParams::default(),
        );
        let pixels = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
        assert!(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|px| px[0] <= 1 && px[1] <= 1 && px[2] <= 1),
            "a perfectly flat input must produce an all-black edge map"
        );
    }

    /// Checker's own cells straddle the wrap seam (see other Bevel/Height
    /// to Normal tests using it for the same reason), so with tiling on,
    /// Edge Detect correctly flags that seam as an edge. With tiling off,
    /// that edge should vanish — it was only ever a wraparound artifact,
    /// not real content. (A generic "is this still seamless" check doesn't
    /// fit this node: both edges of a detected line drop to the same flat
    /// 0 together when clamped, so column 0 still matches column
    /// resolution-1 either way — confirmed by hand. The real effect is a
    /// detected edge *disappearing*, which this test checks directly.)
    #[test]
    fn tiling_off_removes_the_wraparound_edge() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        checker_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &CheckerParams { scale: 2 },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "wrapped",
            RESOLUTION,
            &EdgeDetectParams::default(),
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "clamped",
            RESOLUTION,
            &EdgeDetectParams {
                tiling_mode: TilingMode::Off,
                ..EdgeDetectParams::default()
            },
        );
        let wrapped = gpu.read_rgba8(pool.texture("wrapped").unwrap(), RESOLUTION);
        let clamped = gpu.read_rgba8(pool.texture("clamped").unwrap(), RESOLUTION);
        let at_seam = |pixels: &[u8]| pixels[0];
        assert!(
            at_seam(&wrapped) > 200,
            "with tiling on, the wrap seam should read as a detected edge"
        );
        assert!(
            at_seam(&clamped) < 10,
            "with tiling off, that same seam should no longer read as an edge at all"
        );
    }
}
