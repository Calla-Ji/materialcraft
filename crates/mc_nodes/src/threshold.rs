// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Blending & Masking / Threshold.
//!
//! Display name: "Threshold". Tooltip: "Hard black/white cutoff: channels at
//! or above the threshold become 1, everything else becomes 0."

use crate::shaders::full_source;

const NODE_KIND: &str = "threshold";

pub struct ThresholdParams {
    pub threshold: f32,
}

impl Default for ThresholdParams {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

impl ThresholdParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            threshold: params
                .get("threshold")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.threshold),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &ThresholdParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("threshold: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("threshold.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&params.threshold.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("threshold_params"),
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
        label: Some("threshold_bind_group"),
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
            label: Some("threshold_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("threshold_pass"),
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
    use super::{dispatch, ThresholdParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::RESOLUTION;

    fn render(params: &ThresholdParams) -> Vec<u8> {
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
        let texture = pool.texture("output").expect("threshold texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn output_is_binary() {
        let pixels = render(&ThresholdParams::default());
        for px in pixels.as_chunks::<4>().0 {
            for &channel in &px[..3] {
                assert!(
                    channel == 0 || channel == 255,
                    "channel {channel} is not a hard 0/255 cutoff"
                );
            }
        }
    }

    /// Threshold is pointwise (no neighborhood sampling, confirmed in
    /// `threshold.wgsl`), so it can't introduce a seam on its own, and its
    /// hard step makes the usual `assert_tiles_seamlessly` check vacuous: a
    /// continuous noise field crossing the cutoff anywhere produces a
    /// 0-to-255 jump just like a seam would, so "no jump bigger than the
    /// interior baseline" stops meaning anything once the baseline *is*
    /// 255. A direct cross-check against thresholding the raw input bytes in
    /// Rust is the real test for a pointwise node: it would catch an actual
    /// implementation bug (e.g. sampling the wrong pixel, an off-by-one in
    /// the cutoff comparison) that a tiling check could miss entirely.
    ///
    /// `input`'s own node texture is `Rgba16Float` (see the banding fix in
    /// `mc_gpu::Context::read_rgba8`'s own doc comment), so `threshold.wgsl`
    /// compares against the real, un-rounded value — while `input` here is
    /// `gpu.read_rgba8`'s necessarily-8-bit *readback* of it. The two only
    /// disagree on which side of the cutoff a pixel falls within one 8-bit
    /// step of `threshold` itself (where the readback's own rounding is the
    /// uncertain part, not the shader), so pixels in that narrow band are
    /// skipped rather than asserted on — everywhere else still gets a
    /// strict check, which is what actually catches a sampling or off-by-
    /// one bug.
    #[test]
    fn matches_thresholding_the_input_directly() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams::default(),
        );
        let input = gpu.read_rgba8(pool.texture("input").unwrap(), RESOLUTION);

        let params = ThresholdParams { threshold: 0.4 };
        dispatch(&mut gpu, &mut pool, "input", "output", RESOLUTION, &params);
        let output = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        let cutoff_u8 = (params.threshold * 255.0).round() as i32;
        for (input_px, output_px) in input
            .as_chunks::<4>()
            .0
            .iter()
            .zip(output.as_chunks::<4>().0)
        {
            for c in 0..3 {
                if (i32::from(input_px[c]) - cutoff_u8).abs() <= 1 {
                    continue;
                }
                let expected = if i32::from(input_px[c]) >= cutoff_u8 {
                    255
                } else {
                    0
                };
                assert_eq!(
                    output_px[c], expected,
                    "channel {c} didn't match a direct threshold of the input"
                );
            }
        }
    }

    #[test]
    fn is_deterministic() {
        let params = ThresholdParams { threshold: 0.3 };
        assert_eq!(
            render(&params),
            render(&params),
            "same input + params produced different pixels"
        );
    }

    #[test]
    fn threshold_one_is_all_black() {
        let pixels = render(&ThresholdParams { threshold: 1.1 });
        assert!(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|px| px[0] == 0 && px[1] == 0 && px[2] == 0),
            "a threshold above every possible value must produce all-black"
        );
    }
}
