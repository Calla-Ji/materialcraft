//! Filters / Spatial / Bevel.
//!
//! Display name: "Bevel". Tooltip: "Rounds a mask or height input's edges
//! into a beveled height map — chain into Height to Normal for the usual
//! beveled-edge look."
//!
//! Implementation: a *box* blur (not the existing Blur node's Gaussian —
//! see `bevel_box_blur.wgsl`'s doc comment for why) at `distance`, then
//! reshape that ramp with a power curve driven by `convexity`
//! (`bevel_remap.wgsl`). Blurring first — rather than computing a true
//! distance-to-edge field — is a standard, cheap approximation: a thin
//! feature naturally blurs down further than a thick one, which is the
//! physically right behavior for a bevel anyway (a narrow ridge has less
//! "flat top" to begin with), and far from any edge the blur hasn't
//! reached, so the plateau on either side stays exactly put regardless of
//! the remap curve.

const BOX_BLUR_NODE_KIND: &str = "bevel_box_blur";
const REMAP_NODE_KIND: &str = "bevel_remap";

pub struct BevelParams {
    /// In UV space (0..1) — how far the bevel reaches from an edge.
    pub distance: f32,
    /// -1..1. 0 is a plain linear chamfer; positive rounds the profile
    /// convex (a dome, rising quickly then flattening); negative makes it
    /// concave (a scoop, staying low then rising sharply).
    pub convexity: f32,
}

impl Default for BevelParams {
    fn default() -> Self {
        Self {
            distance: 0.05,
            convexity: 0.3,
        }
    }
}

impl BevelParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let float = |name: &str, default: f32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(default)
        };
        Self {
            distance: float("distance", defaults.distance),
            convexity: float("convexity", defaults.convexity),
        }
    }
}

/// Runs the separable box-blur pass twice (horizontal into a temp texture,
/// then vertical from the temp texture into `output_key`) to get a full 2D
/// box blur — mirrors `blur::dispatch`'s own two-pass structure.
fn box_blur(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    radius_px: f32,
) {
    let temp_key = format!("{output_key}__box_blur_temp");
    box_blur_pass(gpu, pool, input_key, &temp_key, resolution, 0, radius_px);
    box_blur_pass(gpu, pool, &temp_key, output_key, resolution, 1, radius_px);
}

#[allow(clippy::too_many_arguments)]
fn box_blur_pass(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    direction: u32,
    radius_px: f32,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("bevel_box_blur: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = crate::shaders::full_source("bevel_box_blur.wgsl");
    let pipeline = gpu.compute_pipeline(BOX_BLUR_NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&direction.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&radius_px.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bevel_box_blur_params"),
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
        label: Some("bevel_box_blur_bind_group"),
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
            label: Some("bevel_box_blur_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bevel_box_blur_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &BevelParams,
) {
    let blurred_key = format!("{output_key}__bevel_blur");
    let radius_px = (params.distance * resolution as f32).max(0.0);
    box_blur(gpu, pool, input_key, &blurred_key, resolution, radius_px);

    let input_view = pool
        .view(&blurred_key)
        .unwrap_or_else(|| panic!("bevel: no blurred texture at key {blurred_key:?}"))
        .clone();

    let shader_source = crate::shaders::full_source("bevel_remap.wgsl");
    let pipeline = gpu.compute_pipeline(REMAP_NODE_KIND, &shader_source, "main");

    // Convexity 0 -> exponent 1 (no change). Positive convexity -> exponent
    // < 1 (rises quickly, flattens near 1 — convex/dome). Negative
    // convexity -> exponent > 1 (stays low, rises sharply near 1 —
    // concave/scoop).
    let exponent = 2f32.powf(-params.convexity * 2.0);

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&exponent.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bevel_remap_params"),
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
        label: Some("bevel_remap_bind_group"),
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
            label: Some("bevel_remap_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bevel_remap_pass"),
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
    use super::{dispatch, BevelParams};
    use crate::checker::{dispatch as checker_dispatch, CheckerParams};
    use crate::shape::{dispatch as shape_dispatch, ShapeParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &BevelParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        shape_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &ShapeParams::default(),
        );
        dispatch(&mut gpu, &mut pool, "input", "output", RESOLUTION, params);
        let texture = pool.texture("output").expect("bevel texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(&render(&BevelParams::default()), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = BevelParams {
            distance: 0.03,
            convexity: -0.5,
        };
        assert_eq!(render(&params), render(&params));
    }

    /// Far outside and far inside a shape (well beyond the bevel's reach),
    /// the output should stay exactly at the original flat/raised values,
    /// regardless of convexity — only the edge transition should reshape.
    #[test]
    fn plateaus_are_undisturbed_regardless_of_convexity() {
        for convexity in [-1.0, 0.0, 1.0] {
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
                "output",
                RESOLUTION,
                &BevelParams {
                    distance: 0.02,
                    convexity,
                },
            );
            let pixels = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
            // x=16 sits deep in the white cell (x<32), x=48 deep in the
            // black cell (x>=32) — both far outside a 0.02*64=~1.3px bevel
            // reach from the x=32 edge (or the x=0/64 wrap edge).
            let white_plateau = pixels[((16 * RESOLUTION + 16) * 4) as usize];
            let black_plateau = pixels[((16 * RESOLUTION + 48) * 4) as usize];
            assert!(
                white_plateau >= 254,
                "white plateau should stay ~255 at convexity {convexity} (got {white_plateau})"
            );
            assert!(
                black_plateau <= 1,
                "black plateau should stay ~0 at convexity {convexity} (got {black_plateau})"
            );
        }
    }

    /// Convexity should visibly change the shape of the edge transition:
    /// at a fixed distance from the edge, positive (convex) convexity
    /// should read brighter than negative (concave) convexity, since a
    /// convex profile rises quickly while a concave one stays low longer.
    #[test]
    fn convexity_reshapes_the_edge_transition() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        checker_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &CheckerParams { scale: 2 },
        );
        let distance = 0.08;
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output_convex",
            RESOLUTION,
            &BevelParams {
                distance,
                convexity: 1.0,
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output_concave",
            RESOLUTION,
            &BevelParams {
                distance,
                convexity: -1.0,
            },
        );
        let convex = gpu.read_rgba8(pool.texture("output_convex").unwrap(), RESOLUTION);
        let concave = gpu.read_rgba8(pool.texture("output_concave").unwrap(), RESOLUTION);
        // A few pixels into the white side of the x=32 edge, still within
        // the 0.08*64=~5px bevel reach.
        let at = |pixels: &[u8]| pixels[((16 * RESOLUTION + 29) * 4) as usize];
        assert!(
            at(&convex) > at(&concave),
            "convex profile ({}) should read brighter than concave ({}) at the same point \
             near the edge",
            at(&convex),
            at(&concave)
        );
    }
}
