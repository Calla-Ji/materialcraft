// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Circular Pattern.
//!
//! Display name: "Circular Pattern". Tooltip: "Distributes a Stamp input
//! evenly around a circle — a polar array, with optional jitter."
//!
//! This node's only real job is deciding *where* each copy goes
//! ([`circular_instances`]) — the actual compositing reuses Path's own
//! stamp-compositing GPU pass ([`crate::path::dispatch_stamp`]) directly,
//! which already handles per-instance position/rotation/scale and wrap-
//! around tiling. `path.rs`'s own `StampInstance::scale` field exists
//! because of this node specifically (Path's own stamp distribution never
//! varies it from 1.0; only this node's Scale Jitter does).

use crate::path::StampInstance;

pub struct CircularPatternParams {
    pub count: u32,
    pub radius: f32,
    /// Moves the circle's own center away from the tile's, tile units —
    /// `dispatch_stamp`'s existing wrap handling already keeps instances
    /// that land near or past a tile edge seamless, the same as it
    /// already does for a large Radius, so no extra wrapping is needed
    /// here beyond just offsetting `center`.
    pub offset_x: f32,
    pub offset_y: f32,
    pub start_angle_deg: f32,
    pub stamp_size: f32,
    pub rotation_jitter_deg: f32,
    /// Fraction of `stamp_size` each instance's own scale may vary by, in
    /// either direction (so 0.3 means each copy is somewhere in
    /// 0.7x..1.3x), clamped so a large value can never flip a copy to a
    /// negative or zero size.
    pub scale_jitter: f32,
    pub position_jitter_radial: f32,
    /// Tile units, converted to an angular offset via the instance's own
    /// radius (so the same jitter value reads as a similar-looking
    /// sideways nudge regardless of how large `radius` is).
    pub position_jitter_tangential: f32,
    /// On: each copy rotates to face outward from the circle's center
    /// (plus `rotation_jitter_deg`), the same "align to curve" idea
    /// Path's own Stamp distribution already has. Off: every copy keeps
    /// the same upright orientation (still subject to rotation jitter).
    pub align_to_tangent: bool,
    pub wrap: bool,
    pub seed: u32,
}

impl Default for CircularPatternParams {
    fn default() -> Self {
        Self {
            count: 8,
            radius: 0.35,
            offset_x: 0.0,
            offset_y: 0.0,
            start_angle_deg: 0.0,
            stamp_size: 0.08,
            rotation_jitter_deg: 0.0,
            scale_jitter: 0.0,
            position_jitter_radial: 0.0,
            position_jitter_tangential: 0.0,
            align_to_tangent: true,
            wrap: true,
            seed: 0,
        }
    }
}

impl CircularPatternParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            count: params
                .get("count")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.count),
            radius: params
                .get("radius")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.radius),
            offset_x: params
                .get("offset_x")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.offset_x),
            offset_y: params
                .get("offset_y")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.offset_y),
            start_angle_deg: params
                .get("start_angle")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.start_angle_deg),
            stamp_size: params
                .get("stamp_size")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.stamp_size),
            rotation_jitter_deg: params
                .get("rotation_jitter")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.rotation_jitter_deg),
            scale_jitter: params
                .get("scale_jitter")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.scale_jitter),
            position_jitter_radial: params
                .get("position_jitter_radial")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.position_jitter_radial),
            position_jitter_tangential: params
                .get("position_jitter_tangential")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.position_jitter_tangential),
            align_to_tangent: params
                .get("align_to_tangent")
                .and_then(mc_core::ParamValue::as_bool)
                .unwrap_or(defaults.align_to_tangent),
            wrap: params
                .get("wrap")
                .and_then(mc_core::ParamValue::as_bool)
                .unwrap_or(defaults.wrap),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
        }
    }
}

/// Places `params.count` instances evenly around a circle centered on the
/// tile (`[0.5, 0.5]`), each jittered deterministically from `params.seed`
/// (a different salt per jitter axis, same `hash01(seed, index, salt)`
/// convention Stitch and Edge Wear already use — same seed + index always
/// gives the same jitter, independent of which axes are even enabled).
pub fn circular_instances(params: &CircularPatternParams) -> Vec<StampInstance> {
    let count = params.count.max(1);
    let mut out = Vec::with_capacity(count as usize);
    let center = [
        0.5f64 + f64::from(params.offset_x),
        0.5f64 + f64::from(params.offset_y),
    ];
    let start = f64::from(params.start_angle_deg).to_radians();
    let radius = f64::from(params.radius).max(0.0);

    for i in 0..count {
        let h_radial = crate::stitch::hash01(params.seed, i, 101) - 0.5;
        let h_tangent = crate::stitch::hash01(params.seed, i, 102) - 0.5;
        let h_rotation = crate::stitch::hash01(params.seed, i, 103) - 0.5;
        let h_scale = crate::stitch::hash01(params.seed, i, 104) - 0.5;

        let base_angle = start + (f64::from(i) / f64::from(count)) * std::f64::consts::TAU;
        // A tangential nudge in tile units only reads as "about the same
        // sideways distance" at every radius if it's converted to an angle
        // via *this* instance's own radius — not a fixed angular jitter,
        // which would nudge a tight ring and a wide ring by wildly
        // different real distances for the same param value.
        let safe_radius = radius.max(1e-4);
        let tangential_shift = h_tangent * f64::from(params.position_jitter_tangential);
        let angle = base_angle + tangential_shift / safe_radius;
        let instance_radius = radius + h_radial * f64::from(params.position_jitter_radial);

        let pos = [
            (center[0] + instance_radius * angle.cos()) as f32,
            (center[1] + instance_radius * angle.sin()) as f32,
        ];

        let outward = angle as f32 + std::f32::consts::FRAC_PI_2;
        let base_rotation = if params.align_to_tangent {
            outward
        } else {
            0.0
        };
        let rotation =
            base_rotation + (h_rotation as f32) * params.rotation_jitter_deg.to_radians();

        let scale = (1.0 + (h_scale as f32) * params.scale_jitter).max(0.01);

        out.push(StampInstance {
            pos,
            angle: rotation,
            scale,
        });
    }
    out
}

/// Dispatches the circular distribution of `stamp_key` into `output_key` —
/// a thin wrapper over [`circular_instances`] plus
/// [`crate::path::dispatch_stamp`], which needs an existing "base" texture
/// to composite onto (this node has none of its own, unlike Path's
/// Preview Mask, so a blank scratch texture stands in).
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    stamp_key: &str,
    output_key: &str,
    resolution: u32,
    params: &CircularPatternParams,
) {
    let blank_key = format!("{output_key}__circular_pattern_blank_scratch");
    gpu.fill_rgba8(pool, &blank_key, resolution, [0, 0, 0, 255]);
    let instances = circular_instances(params);
    crate::path::dispatch_stamp(
        gpu,
        pool,
        stamp_key,
        &blank_key,
        output_key,
        resolution,
        &instances,
        params.stamp_size,
        params.wrap,
    );
}

#[cfg(test)]
mod tests {
    use super::{circular_instances, dispatch, CircularPatternParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &CircularPatternParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        // A small disc as the stamp, via the Shape node's own dispatch —
        // reusing a real node's output rather than a hand-built texture,
        // same precedent `path.rs`'s own stamp tests already follow.
        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "stamp",
            RESOLUTION,
            &crate::shape::ShapeParams {
                kind: crate::shape::ShapeKind::Disc,
                size: 0.9,
                softness: 0.0,
                ..crate::shape::ShapeParams::default()
            },
        );
        dispatch(&mut gpu, &mut pool, "stamp", "output", RESOLUTION, params);
        let texture = pool.texture("output").expect("circular_pattern texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn places_exactly_count_instances_evenly_spaced() {
        let params = CircularPatternParams {
            count: 6,
            ..CircularPatternParams::default()
        };
        let instances = circular_instances(&params);
        assert_eq!(instances.len(), 6);
        // Evenly spaced means every consecutive pair's angular gap (via
        // atan2 around the tile center) should match, within float
        // tolerance — a much more direct check than re-deriving each
        // expected position.
        let center = [0.5f64, 0.5f64];
        let mut angles: Vec<f64> = instances
            .iter()
            .map(|inst| {
                (f64::from(inst.pos[1]) - center[1]).atan2(f64::from(inst.pos[0]) - center[0])
            })
            .collect();
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let gaps: Vec<f64> = angles.windows(2).map(|w| w[1] - w[0]).collect();
        let expected_gap = std::f64::consts::TAU / 6.0;
        for gap in gaps {
            assert!(
                (gap - expected_gap).abs() < 1e-6,
                "expected every gap to be {expected_gap}, got {gap}"
            );
        }
    }

    /// Offset X/Y move the whole circle's center away from the tile's —
    /// every instance should land exactly `radius` from the *offset*
    /// center, not the tile's own [0.5, 0.5].
    #[test]
    fn offset_moves_the_whole_circle() {
        let params = CircularPatternParams {
            count: 4,
            radius: 0.2,
            offset_x: 0.15,
            offset_y: -0.1,
            position_jitter_radial: 0.0,
            position_jitter_tangential: 0.0,
            ..CircularPatternParams::default()
        };
        let instances = circular_instances(&params);
        let center = [0.5f64 + 0.15, 0.5f64 - 0.1];
        for inst in &instances {
            let dx = f64::from(inst.pos[0]) - center[0];
            let dy = f64::from(inst.pos[1]) - center[1];
            let dist = dx.hypot(dy);
            assert!(
                (dist - 0.2).abs() < 1e-5,
                "instance at {:?} should sit 0.2 from the offset center {center:?}, got {dist}",
                inst.pos
            );
        }
    }

    #[test]
    fn zero_jitter_is_fully_deterministic_and_matches_a_hand_computed_position() {
        let params = CircularPatternParams {
            count: 4,
            radius: 0.25,
            start_angle_deg: 0.0,
            ..CircularPatternParams::default()
        };
        let instances = circular_instances(&params);
        assert_eq!(instances.len(), 4);
        // First instance at angle 0: directly right of center.
        assert!((instances[0].pos[0] - 0.75).abs() < 1e-5);
        assert!((instances[0].pos[1] - 0.5).abs() < 1e-5);
        // Every instance keeps the default scale when scale_jitter is 0.
        for inst in &instances {
            assert_eq!(inst.scale, 1.0);
        }
    }

    #[test]
    fn align_to_tangent_points_each_instance_away_from_center() {
        let params = CircularPatternParams {
            count: 4,
            start_angle_deg: 0.0,
            align_to_tangent: true,
            ..CircularPatternParams::default()
        };
        let instances = circular_instances(&params);
        // The first instance sits at angle 0 (due "east" of center); its
        // own outward-facing angle should be a quarter turn from that.
        let expected = std::f32::consts::FRAC_PI_2;
        assert!((instances[0].angle - expected).abs() < 1e-5);
    }

    #[test]
    fn align_to_tangent_off_keeps_every_instance_upright() {
        let params = CircularPatternParams {
            count: 5,
            align_to_tangent: false,
            ..CircularPatternParams::default()
        };
        for inst in circular_instances(&params) {
            assert_eq!(inst.angle, 0.0);
        }
    }

    #[test]
    fn scale_jitter_stays_within_its_own_declared_range() {
        let params = CircularPatternParams {
            count: 64,
            scale_jitter: 0.4,
            ..CircularPatternParams::default()
        };
        for inst in circular_instances(&params) {
            assert!(
                (0.6..=1.4).contains(&inst.scale),
                "scale {} outside the expected 0.6..=1.4 band",
                inst.scale
            );
        }
    }

    #[test]
    fn same_seed_is_deterministic_different_seed_moves_jittered_instances() {
        let jittered = CircularPatternParams {
            count: 10,
            position_jitter_radial: 0.05,
            position_jitter_tangential: 0.05,
            rotation_jitter_deg: 30.0,
            scale_jitter: 0.3,
            seed: 7,
            ..CircularPatternParams::default()
        };
        let a = circular_instances(&jittered);
        let b = circular_instances(&jittered);
        assert_eq!(a, b, "same seed should reproduce identical instances");

        let different_seed = CircularPatternParams {
            seed: 8,
            ..jittered
        };
        let c = circular_instances(&different_seed);
        assert_ne!(
            a, c,
            "a different seed should actually move at least one instance"
        );
    }

    #[test]
    fn renders_and_tiles_seamlessly() {
        let params = CircularPatternParams {
            count: 10,
            radius: 0.3,
            stamp_size: 0.08,
            ..CircularPatternParams::default()
        };
        let pixels = render(&params);
        assert!(
            pixels.as_chunks::<4>().0.iter().any(|p| p[0] > 10),
            "a circular pattern with real instances should draw something"
        );
        assert_tiles_seamlessly(&pixels, RESOLUTION);
    }

    #[test]
    fn zero_count_clamps_to_one_instance_not_zero() {
        let params = CircularPatternParams {
            count: 0,
            ..CircularPatternParams::default()
        };
        assert_eq!(circular_instances(&params).len(), 1);
    }
}
