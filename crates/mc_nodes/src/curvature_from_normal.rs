//! Converters / Curvature from Normal.
//!
//! Display name: "Curvature from Normal". Tooltip: "Convex edges read
//! bright, concave crevices read dark, flat areas sit at mid-gray."

use crate::shaders::full_source;

const NODE_KIND: &str = "curvature_from_normal";

pub struct CurvatureFromNormalParams {
    pub intensity: f32,
}

impl Default for CurvatureFromNormalParams {
    fn default() -> Self {
        Self { intensity: 4.0 }
    }
}

impl CurvatureFromNormalParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            intensity: params
                .get("intensity")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.intensity),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &CurvatureFromNormalParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("curvature_from_normal: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("curvature_from_normal.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&params.intensity.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("curvature_from_normal_params"),
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
        label: Some("curvature_from_normal_bind_group"),
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
            label: Some("curvature_from_normal_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("curvature_from_normal_pass"),
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
    use super::{dispatch, CurvatureFromNormalParams};
    use crate::height_to_normal::{dispatch as height_to_normal_dispatch, HeightToNormalParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};
    use crate::uniform_color::{dispatch as uniform_color_dispatch, UniformColorParams};

    /// Height to Normal is the natural upstream of this node in a real
    /// graph, so tests build a normal map that way rather than from a flat
    /// uniform color (which would only exercise the flat/no-op case).
    fn render(params: &CurvatureFromNormalParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "height",
            RESOLUTION,
            &PerlinParams::default(),
        );
        height_to_normal_dispatch(
            &mut gpu,
            &mut pool,
            "height",
            "normal",
            RESOLUTION,
            &HeightToNormalParams::default(),
        );
        dispatch(&mut gpu, &mut pool, "normal", "output", RESOLUTION, params);
        let texture = pool
            .texture("output")
            .expect("curvature_from_normal texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(&render(&CurvatureFromNormalParams::default()), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = CurvatureFromNormalParams { intensity: 6.0 };
        assert_eq!(render(&params), render(&params));
    }

    /// A perfectly flat normal map (pointing straight up everywhere) has
    /// zero curvature everywhere, so the output should sit at flat mid-gray.
    #[test]
    fn flat_normal_is_mid_gray() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        uniform_color_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &UniformColorParams {
                color: [0.5, 0.5, 1.0, 1.0],
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output",
            RESOLUTION,
            &CurvatureFromNormalParams::default(),
        );
        let pixels = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
        for px in pixels.as_chunks::<4>().0 {
            assert!(
                (px[0] as i32 - 128).abs() <= 2,
                "a flat normal map must produce flat mid-gray curvature, got {}",
                px[0]
            );
        }
    }
}
