//! Filters / Blending & Masking / Histogram Scan.
//!
//! Display name: "Histogram Scan". Tooltip: "Threshold/window the input
//! around a position, with adjustable sharpness."

use crate::shaders::full_source;

const NODE_KIND: &str = "histogram_scan";

pub struct HistogramScanParams {
    pub position: f32,
    pub contrast: f32,
}

impl Default for HistogramScanParams {
    fn default() -> Self {
        Self {
            position: 0.5,
            contrast: 0.0,
        }
    }
}

impl HistogramScanParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            position: params
                .get("position")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.position),
            contrast: params
                .get("contrast")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.contrast),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &HistogramScanParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("histogram_scan: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("histogram_scan.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("histogram_scan_params"),
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
        label: Some("histogram_scan_bind_group"),
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
            label: Some("histogram_scan_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("histogram_scan_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

fn pack_params(resolution: u32, params: &HistogramScanParams) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.position.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.contrast.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, HistogramScanParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &HistogramScanParams) -> Vec<u8> {
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
        let texture = pool.texture("output").expect("histogram_scan texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(&render(&HistogramScanParams::default()), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = HistogramScanParams {
            position: 0.3,
            contrast: 0.8,
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same input + params produced different pixels"
        );
    }

    #[test]
    fn hard_threshold_is_almost_all_binary() {
        // contrast = 1.0 narrows the ramp to a thin (but nonzero) band, so a
        // handful of pixels landing exactly inside it can be non-binary —
        // this checks the overwhelming majority are, not literally every one.
        let params = HistogramScanParams {
            position: 0.5,
            contrast: 1.0,
        };
        let pixels = render(&params);
        let channels: Vec<u8> = pixels
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| p[0..3].to_vec())
            .collect();
        let binary_count = channels.iter().filter(|&&c| c == 0 || c == 255).count();
        let ratio = binary_count as f64 / channels.len() as f64;
        assert!(
            ratio > 0.95,
            "expected >95% binary pixels at contrast 1.0, got {:.1}%",
            ratio * 100.0
        );
    }
}
