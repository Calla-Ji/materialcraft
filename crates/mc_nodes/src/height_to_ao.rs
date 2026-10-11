//! Converters / Height to AO.
//!
//! Display name: "Height to AO". Tooltip: "A simple screen-space-style
//! ambient occlusion map from height — darker in valleys."
//!
//! This is the real Ambient Occlusion *generator* — `Output: Ambient
//! Occlusion` is just a pass-through marker like every other Output node,
//! with no shader of its own.
//!
//! Implementation: compare the height map against three *max-filtered*
//! (dilated) copies of itself — fine/medium/broad radii, each a multiple of
//! the node's own Radius param — then darken wherever the real height sits
//! notably below the strongest-occluding of the three. An earlier version
//! compared against *blurred* (averaged) copies instead; that fixed an even
//! earlier version's hard-edged ring artifact (see PROGRESS.md) but turned
//! out to have its own gap: averaging dilutes a thin, sharp feature (e.g. a
//! leaf shape's pointed tip) down to almost nothing once it's surrounded
//! mostly by low background, so the comparison missed occlusion exactly
//! where a shape narrows to a point, leaving a visible notch in the halo at
//! each tip. A max filter has no such dilution — if a tall neighbor is
//! anywhere within reach, it's reflected at full strength regardless of how
//! little area it covers — while keeping the earlier fix's smooth falloff
//! and multi-scale layering (see `height_to_ao_dilate.wgsl`'s doc comment).

use crate::shaders::full_source;

const NODE_KIND: &str = "height_to_ao";
const DILATE_NODE_KIND: &str = "height_to_ao_dilate";

/// Multipliers on the node's own `radius` param for the fine/medium/broad
/// max-filter passes `dispatch` compares the height map against. Fixed
/// rather than exposed as params: three extra sliders for what's meant to
/// be a one-knob "how far does occlusion reach" control would fight the
/// artist-friendly, minimal-controls goal (CLAUDE.md §12) for a tuning
/// knob most graphs won't need to touch.
const SCALE_MULTIPLIERS: [f32; 3] = [1.0, 2.0, 4.0];

/// Runs the separable max-filter pass twice (horizontal into a temp
/// texture, then vertical from the temp texture into `output_key`) to get a
/// full 2D max filter — mirrors `blur::dispatch`'s own two-pass structure.
fn dilate(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    radius_px: f32,
) {
    let temp_key = format!("{output_key}__dilate_temp");
    dilate_pass(gpu, pool, input_key, &temp_key, resolution, 0, radius_px);
    dilate_pass(gpu, pool, &temp_key, output_key, resolution, 1, radius_px);
}

#[allow(clippy::too_many_arguments)]
fn dilate_pass(
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
        .unwrap_or_else(|| panic!("height_to_ao_dilate: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("height_to_ao_dilate.wgsl");
    let pipeline = gpu.compute_pipeline(DILATE_NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&direction.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&radius_px.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("height_to_ao_dilate_params"),
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
        label: Some("height_to_ao_dilate_bind_group"),
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
            label: Some("height_to_ao_dilate_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("height_to_ao_dilate_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

pub struct HeightToAoParams {
    /// In UV space (0..1) — the *fine* scale's blur radius; the medium and
    /// broad scales are fixed multiples of this (see `SCALE_MULTIPLIERS`).
    pub radius: f32,
    pub intensity: f32,
}

impl Default for HeightToAoParams {
    fn default() -> Self {
        Self {
            radius: 0.02,
            intensity: 2.0,
        }
    }
}

impl HeightToAoParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let float = |name: &str, default: f32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(default)
        };
        Self {
            radius: float("radius", defaults.radius),
            intensity: float("intensity", defaults.intensity),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &HeightToAoParams,
) {
    // dilate's cost is bounded (a fixed 32-tap kernel regardless of radius —
    // see height_to_ao_dilate.wgsl), so three passes here stays cheap even
    // at the largest (broad) scale.
    let dilate_keys: Vec<String> = SCALE_MULTIPLIERS
        .iter()
        .enumerate()
        .map(|(i, multiplier)| {
            let dilate_key = format!("{output_key}__ao_dilate_{i}");
            let radius_px = (params.radius * multiplier * resolution as f32).max(1.0);
            dilate(gpu, pool, input_key, &dilate_key, resolution, radius_px);
            dilate_key
        })
        .collect();

    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("height_to_ao: no texture at input key {input_key:?}"))
        .clone();
    let dilated_views: Vec<wgpu::TextureView> = dilate_keys
        .iter()
        .map(|key| {
            pool.view(key)
                .unwrap_or_else(|| panic!("height_to_ao: no dilated texture at key {key:?}"))
                .clone()
        })
        .collect();

    let shader_source = full_source("height_to_ao.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&params.intensity.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("height_to_ao_params"),
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
        label: Some("height_to_ao_bind_group"),
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
                resource: wgpu::BindingResource::TextureView(&dilated_views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&dilated_views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&dilated_views[2]),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("height_to_ao_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("height_to_ao_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));

    // The max-filter combine above is sensitive to exactly which of its
    // discrete sample taps happens to be largest at each pixel — as the
    // query position shifts by a single texel, a different tap can take
    // over as the max, which (unlike a smooth weighted average, e.g.
    // blur.wgsl's own combine) shows up as visible fine banding/stepping in
    // otherwise-smooth falloff regions (confirmed via a rendered dump — see
    // PROGRESS.md). A small fixed-radius blur on the finished result
    // smooths that stepping away without reopening the thin-point gap this
    // whole max-filter approach exists to fix in the first place: every
    // pixel's occlusion is already correctly nonzero by this point, so a
    // light blur here only evens out noise between already-correct
    // neighboring values, rather than diluting a thin feature's influence
    // the way blurring the raw *height* map did.
    crate::blur::dispatch(
        gpu,
        pool,
        output_key,
        output_key,
        resolution,
        &crate::blur::BlurParams {
            radius: params.radius * 0.5,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::{dispatch, HeightToAoParams};
    use crate::checker::{dispatch as checker_dispatch, CheckerParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::shape::{dispatch as shape_dispatch, ShapeKind, ShapeParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};
    use crate::uniform_color::{dispatch as uniform_color_dispatch, UniformColorParams};

    fn render(params: &HeightToAoParams) -> Vec<u8> {
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
        let texture = pool.texture("output").expect("height_to_ao texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(&render(&HeightToAoParams::default()), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = HeightToAoParams {
            radius: 0.03,
            intensity: 3.0,
        };
        assert_eq!(render(&params), render(&params));
    }

    /// A perfectly flat height map has no valleys anywhere, so AO should
    /// read fully unoccluded (white).
    #[test]
    fn flat_height_is_fully_unoccluded() {
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
            &HeightToAoParams::default(),
        );
        let pixels = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
        assert!(
            pixels.as_chunks::<4>().0.iter().all(|px| px[0] >= 254),
            "a flat height map must produce fully white (unoccluded) AO"
        );
    }

    /// A low flat floor surrounded by a tall flat wall ring (built from two
    /// nested Shape discs) should occlude the floor and leave the high
    /// ring unoccluded — a concrete, known-correct sanity check beyond
    /// "doesn't crash."
    #[test]
    fn a_pit_is_darker_than_its_rim() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        // A radial gradient is lowest at the center, highest at the edge —
        // i.e. a bowl/pit shape, perfect for this check.
        crate::gradient_radial::dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &crate::gradient_radial::GradientRadialParams { radius: 0.5 },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output",
            RESOLUTION,
            &HeightToAoParams {
                radius: 0.04,
                intensity: 4.0,
            },
        );
        let pixels = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
        let center = RESOLUTION / 2;
        let center_ao = pixels[((center * RESOLUTION + center) * 4) as usize];
        let rim_ao = pixels[((center * RESOLUTION + (RESOLUTION - 2)) * 4) as usize];
        assert!(
            center_ao < rim_ao,
            "the pit's center ({center_ao}) should be darker than its rim ({rim_ao})"
        );
    }

    /// Regression test for a reported bug: a sharp height edge (e.g. a
    /// disc's silhouette) produced a thin, hard-edged occlusion *ring*
    /// traced at one fixed distance from the edge, rather than a smooth
    /// falloff — so dragging Radius/Intensity just relocated or recolored
    /// that ring instead of visibly widening/softening a gradient. A 2x2
    /// Checker gives a pixel-exact vertical edge at x=32 (white/raised for
    /// x<32, black/low for x>=32); AO should now fade in smoothly moving
    /// away from that edge on the low side, not jump straight from "fully
    /// occluded" to "fully unoccluded" at one specific distance.
    #[test]
    fn a_sharp_edge_occludes_smoothly_not_as_a_hard_ring() {
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
            &HeightToAoParams::default(),
        );
        let pixels = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
        let y = 16; // within the first row of cells, so only the x=32 edge matters here.
        let ao_at = |x: u32| pixels[((y * RESOLUTION + x) * 4) as usize];

        // At the default Intensity, a pixel-exact hard step legitimately
        // saturates to fully occluded for the first few pixels (a strong
        // intensity blowing out a genuinely sharp crease to black isn't a
        // bug), and the final smoothing blur (see `dispatch`'s doc comment)
        // means the one or two pixels immediately at the edge itself can
        // dip and resettle rather than being perfectly monotonic — neither
        // is what's being guarded here. The real thing this checks is what
        // happens a little further out: a *climb* back out through
        // intermediate values, not an instant jump straight from "fully
        // occluded" to "fully clear."
        let still_saturated = ao_at(35); // within the saturated band, past the edge's own dip
        let climbing = ao_at(38); // where the climb starts
        let climbing_more = ao_at(41); // partway up the climb
        let far = ao_at(48); // 16px in — the black cell's center, farthest from
                             // both this edge and the wrap-around edge at x=0/64.

        assert!(
            still_saturated <= climbing && climbing < climbing_more && climbing_more < far,
            "occlusion should climb smoothly back to unoccluded, not jump straight from fully \
             occluded to fully clear (still_saturated={still_saturated}, climbing={climbing}, \
             climbing_more={climbing_more}, far={far})"
        );
        assert!(
            far >= 254,
            "a pixel well outside every scale's reach should be fully unoccluded (got {far})"
        );
    }

    /// Regression test for a reported bug: a sharply-pointed shape (the
    /// owner's example was Shape's Leaf kind, curvature cranked up for a
    /// sharp tip) showed a visible gap in the occlusion halo right at each
    /// point — a blur-based comparison dilutes a thin feature's influence
    /// on nearby background down to almost nothing, so the halo vanished
    /// exactly where the shape narrowed to a tip, even though it wrapped
    /// the rest of the (broader) silhouette correctly. The pixel
    /// immediately outside a sharp tip must now be clearly occluded, not
    /// fully white.
    #[test]
    fn a_sharp_point_is_occluded_with_no_gap_in_the_halo() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        shape_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &ShapeParams {
                kind: ShapeKind::Leaf,
                curvature_top: 4.0,
                curvature_bottom: 4.0,
                ..ShapeParams::default()
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output",
            RESOLUTION,
            &HeightToAoParams::default(),
        );
        let pixels = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
        // The shape is centered and symmetric about the vertical midline,
        // with its top tip a handful of pixels below y=0 at this size/
        // resolution — y=16 sits just outside the tip, in the gap the old
        // blur-based version left fully white.
        let cx = RESOLUTION / 2;
        let just_outside_tip = pixels[((16 * RESOLUTION + cx) * 4) as usize];
        assert!(
            just_outside_tip < 200,
            "the point immediately outside a sharp tip should be clearly occluded, not left in \
             a gap (got {just_outside_tip})"
        );
    }
}
