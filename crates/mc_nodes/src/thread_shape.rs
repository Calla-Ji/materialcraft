// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Thread.
//!
//! Display name: "Thread". Tooltip: "A single fiber's cross-section and
//! twist — wire into Fabric Generator's Thread input to control what its
//! threads look like."
//!
//! One shape, not tiled — the same relationship to Fabric Generator that
//! Shape has to Pattern Generator's own Shape input: a dedicated shape
//! generator feeding an arranger node, rather than the arranger owning the
//! shape itself.
//!
//! Redesigned around real yarn-construction references (the owner's own
//! textile reference book, not assumed from a generic "rope" mental image):
//! a twisted/plied yarn's visual identity comes from the strands' helical
//! path *along* the thread's length (the Yarn Twist S/Z-direction and
//! twist-level concepts every ply-yarn reference covers), not from its
//! cross-section silhouette alone — the first version of this node only
//! varied cross-section and, correctly, didn't read as "rope" at all. See
//! the shader's own doc comment for the periodicity requirement this
//! drives.
//!
//! Getting the twist to actually read as rope took several rounds, each
//! confirmed (and the previous one's shortfall found) against the owner's
//! own screenshots rather than assumed correct from the math alone:
//!
//! 1. Strands just orbiting the center, combined with a plain `max()` of
//!    brightness, only captured their sideways motion — for similarly-sized,
//!    evenly-spaced strands that produces a moire/lattice interference
//!    pattern, not distinct plies.
//! 2. Adding depth (`cos` of each strand's own orbit angle, the other half
//!    of the same sin/cos helix projection `sin` already provided for
//!    position) fixed that, and incidentally resolved an unrelated, now-
//!    stale limitation where Twist Direction had no visible effect at
//!    exactly `ply_count` 2 (see
//!    `twist_direction_also_differs_at_two_ply_now_that_strands_occlude`).
//!    But sizing strands to comfortably overlap (to guarantee the bundle's
//!    centerline was never literally empty) fused the two plies into what
//!    read as one continuous tube with no visible separation, and combining
//!    overlapping strands with a hard `max()`/winner-take-all pick drew a
//!    visible kink everywhere the dominant strand switched, since a hard
//!    max of several continuous sine curves is not itself continuous in
//!    slope — the "ragged outline" and "overcut groove" a screenshot showed.
//! 3. Strands sized to just touch (not overlap) rather than forcing
//!    coverage, combined with a proper SDF-style smooth union for *shape*
//!    (continuous-slope silhouette, no kink) and a *separate*,
//!    depth-only blend for *shading*. Fixed the ply_count 2 case well (see
//!    `two_plies_show_a_visible_valley_not_a_fused_bridge`), but
//!    reintroduced the very first version's moire/lattice pattern for
//!    ply_count 3+ — treating shape as depth-blind meant 3+ strands could
//!    be simultaneously present with no clear front/back winner, so all of
//!    them bulged the silhouette together, same as step 1 above.
//! 4. The shader's current approach: a *single* combined per-strand weight
//!    (geometric presence *and* depth folded into one softmax score) drives
//!    both the shape and the shading, instead of two independently-combined
//!    operations. A real twisted yarn only shows whichever strand currently
//!    faces the viewer — depth has to gate the shape too, not just the
//!    brightness — while strands close in both position and depth (an
//!    actual crossing) still blend smoothly between them, keeping the
//!    continuous-slope silhouette step 3 was for. See the shader's own doc
//!    comment for the exact formula, and
//!    `three_plies_dont_regress_into_a_lattice` /
//!    `higher_ply_counts_dont_regress_into_a_lattice_either` for how the
//!    lattice regression specifically is now caught.
//! 5. Step 4 fixed the lattice, but with the owner's own hand-drawn
//!    annotation over a screenshot confirming the actual complaint — each
//!    ply's own visible lobe should be a longer, pointed lens shape, not a
//!    short rounded bump — a fixed depth transition sharpness turned out
//!    to alias into a literal multi-pixel jump at 2 plies, not just a
//!    sharp-looking but technically smooth curve (confirmed by tracing the
//!    silhouette boundary pixel by pixel at a higher resolution than the
//!    other tests here use, not assumed from the math alone — see
//!    `two_ply_silhouette_boundary_has_no_single_column_jump`). But 2
//!    plies and 3+ plies need genuinely different transition sharpness,
//!    not just different constants on the same formula: with only 2
//!    strands, "who's in front" is always a binary comparison, so a gentle
//!    transition is safe; with 3+, several strands can be close in depth
//!    *simultaneously*, and the same gentle transition reopens the lattice
//!    from step 3. The shader now picks the two regimes separately rather
//!    than chasing one formula that satisfies both.

use crate::shaders::full_source;

const NODE_KIND: &str = "thread_shape";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwistDirection {
    S,
    Z,
}

impl TwistDirection {
    pub const NAMES: [&'static str; 2] = ["S", "Z"];

    fn as_u32(self) -> u32 {
        match self {
            TwistDirection::S => 0,
            TwistDirection::Z => 1,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "S" => Some(TwistDirection::S),
            "Z" => Some(TwistDirection::Z),
            _ => None,
        }
    }
}

pub struct ThreadShapeParams {
    pub ply_count: u32,
    pub thickness: f32,
    pub roundness: f32,
    pub twist_cycles: u32,
    pub twist_direction: TwistDirection,
}

impl Default for ThreadShapeParams {
    fn default() -> Self {
        Self {
            ply_count: 1,
            thickness: 0.5,
            roundness: 0.3,
            twist_cycles: 2,
            twist_direction: TwistDirection::Z,
        }
    }
}

impl ThreadShapeParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            ply_count: params
                .get("ply_count")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.ply_count),
            thickness: params
                .get("thickness")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.thickness),
            roundness: params
                .get("roundness")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.roundness),
            twist_cycles: params
                .get("twist_cycles")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(0) as u32)
                .unwrap_or(defaults.twist_cycles),
            twist_direction: params
                .get("twist_direction")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => TwistDirection::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.twist_direction),
        }
    }
}

fn pack_params(resolution: u32, params: &ThreadShapeParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.ply_count.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.thickness.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.roundness.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.twist_cycles.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.twist_direction.as_u32().to_le_bytes());
    bytes
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &ThreadShapeParams,
) {
    let shader_source = full_source("thread_shape.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("thread_shape_params"),
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
        label: Some("thread_shape_bind_group"),
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
            label: Some("thread_shape_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("thread_shape_pass"),
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
    use super::{dispatch, ThreadShapeParams, TwistDirection};
    use crate::test_support::RESOLUTION;

    fn render(params: &ThreadShapeParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("thread_shape texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn is_deterministic() {
        let params = ThreadShapeParams::default();
        assert_eq!(render(&params), render(&params));
    }

    /// The "ragged outline" complaint turned out, when traced pixel by
    /// pixel at a higher resolution than the other tests here use, to be a
    /// real discontinuity: the silhouette's own top/bottom boundary jumped
    /// by dozens of pixels between *adjacent* columns right at the
    /// crossing handoff, not a gradual curve. A smooth function can still
    /// alias into a visible seam if its transition is narrower than a
    /// pixel — this checks the boundary's actual column-to-column movement
    /// stays small, which a purely analytic "is it C1 continuous" argument
    /// wouldn't have caught.
    #[test]
    fn two_ply_silhouette_boundary_has_no_single_column_jump() {
        let hi_res = 256u32;
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(
            &mut gpu,
            &mut pool,
            "probe",
            hi_res,
            &ThreadShapeParams {
                ply_count: 2,
                thickness: 0.5,
                roundness: 0.3,
                twist_cycles: 2,
                twist_direction: TwistDirection::Z,
            },
        );
        let pixels = gpu.read_rgba8(pool.texture("probe").unwrap(), hi_res);
        let r = hi_res as usize;
        let tops: Vec<Option<usize>> = (0..r)
            .map(|x| (0..r).find(|&y| pixels[(y * r + x) * 4] > 128))
            .collect();
        let mut max_jump = 0usize;
        for pair in tops.windows(2) {
            if let [Some(a), Some(b)] = pair {
                max_jump = max_jump.max(a.abs_diff(*b));
            }
        }
        assert!(
            max_jump < 10,
            "the silhouette's top boundary jumped {max_jump} pixels between adjacent columns \
             (resolution {r}) — that reads as a visible seam, not a smooth curve"
        );
    }

    /// A screenshot showed the previous (overlap-guaranteed) version reading
    /// as one continuous fused tube with no visible separation between the
    /// two plies at all — the valley between them stayed too bright. The
    /// smooth-union rewrite should still show a real, clearly darker dip
    /// where the plies are furthest apart, not just a near-white bridge —
    /// confirmed numerically, not just visually.
    #[test]
    fn two_plies_show_a_visible_valley_not_a_fused_bridge() {
        let pixels = render(&ThreadShapeParams {
            ply_count: 2,
            ..ThreadShapeParams::default()
        });
        let r = RESOLUTION as usize;
        let row = r / 2;
        let row_values: Vec<u8> = (0..r).map(|x| pixels[(row * r + x) * 4]).collect();
        let min = *row_values.iter().min().unwrap();
        let max = *row_values.iter().max().unwrap();
        assert!(
            max > 200,
            "the two plies' crossing point should read bright (max={max})"
        );
        assert!(
            min < 100,
            "the valley between plies should read clearly darker than the crossing \
             point, not fused into one continuous bright bridge (min={min}, max={max})"
        );
    }

    #[test]
    fn centerline_is_brighter_than_the_edge() {
        let pixels = render(&ThreadShapeParams {
            ply_count: 1,
            ..ThreadShapeParams::default()
        });
        let r = RESOLUTION as usize;
        let center = pixels[(r / 2 * r + r / 2) * 4];
        let edge = pixels[(0 * r + r / 2) * 4];
        assert!(
            center > edge,
            "fiber centerline ({center}) should be brighter than the tile edge ({edge})"
        );
    }

    #[test]
    fn zero_thickness_is_blank() {
        let pixels = render(&ThreadShapeParams {
            ply_count: 1,
            thickness: 0.0,
            ..ThreadShapeParams::default()
        });
        assert!(
            pixels.iter().step_by(4).all(|&v| v == 0),
            "zero thickness should produce an entirely black fiber"
        );
    }

    #[test]
    fn roundness_changes_the_profile() {
        let flat = render(&ThreadShapeParams {
            ply_count: 1,
            roundness: 0.0,
            ..ThreadShapeParams::default()
        });
        let round = render(&ThreadShapeParams {
            ply_count: 1,
            roundness: 1.0,
            ..ThreadShapeParams::default()
        });
        assert_ne!(flat, round);
    }

    /// A screenshot showed 3+ plies reverting to the very first version's
    /// moire/lattice interference pattern — a dense ripple of many small
    /// bumps, not the handful of clean crossing-points real twisted yarn
    /// shows. Root cause: an earlier rewrite combined the strands' *shape*
    /// (geometric presence) independent of depth, so with 3+ strands,
    /// several could be simultaneously present with no clear front/back
    /// winner and all bulge the silhouette at once. Counting local maxima
    /// along the centerline is a coarse but effective proxy: each strand
    /// should get roughly one hump per full twist rotation as it takes its
    /// turn facing the viewer (`ply_count * twist_cycles`), not dozens of
    /// small interference ripples.
    #[test]
    fn three_plies_dont_regress_into_a_lattice() {
        let pixels = render(&ThreadShapeParams {
            ply_count: 3,
            thickness: 0.54,
            roundness: 0.3,
            twist_cycles: 2,
            twist_direction: TwistDirection::Z,
        });
        let r = RESOLUTION as usize;
        let row = r / 2;
        let row_values: Vec<u8> = (0..r).map(|x| pixels[(row * r + x) * 4]).collect();
        let mut maxima = 0;
        for x in 1..r - 1 {
            if row_values[x] > row_values[x - 1] && row_values[x] >= row_values[x + 1] {
                maxima += 1;
            }
        }
        let expected = 3 * 2; // ply_count * twist_cycles
        assert!(
            maxima <= expected + 2,
            "centerline shows {maxima} local maxima, far more than the ~{expected} expected \
             for 3 plies over 2 twist cycles — looks like the lattice regression, not a clean \
             braid"
        );
    }

    /// Same lattice check as `three_plies_dont_regress_into_a_lattice`, but
    /// swept across higher ply counts too — the 3-ply case alone wouldn't
    /// have caught a regression that only kicked in at, say, 5+ strands.
    #[test]
    fn higher_ply_counts_dont_regress_into_a_lattice_either() {
        let r = RESOLUTION as usize;
        let row = r / 2;
        for ply_count in [4u32, 5, 6] {
            let pixels = render(&ThreadShapeParams {
                ply_count,
                thickness: 0.54,
                roundness: 0.3,
                twist_cycles: 2,
                twist_direction: TwistDirection::Z,
            });
            let row_values: Vec<u8> = (0..r).map(|x| pixels[(row * r + x) * 4]).collect();
            let mut maxima = 0;
            for x in 1..r - 1 {
                if row_values[x] > row_values[x - 1] && row_values[x] >= row_values[x + 1] {
                    maxima += 1;
                }
            }
            let expected = ply_count * 2;
            assert!(
                maxima <= expected as usize + 2,
                "ply_count {ply_count}: centerline shows {maxima} local maxima, far more than \
                 the ~{expected} expected — looks like the lattice regression"
            );
        }
    }

    #[test]
    fn single_ply_is_constant_along_its_length() {
        // Fabric Generator relies on sampling at any along-length position
        // when there's nothing to twist (a single strand has no partner to
        // orbit around).
        let pixels = render(&ThreadShapeParams {
            ply_count: 1,
            ..ThreadShapeParams::default()
        });
        let r = RESOLUTION as usize;
        let row = r / 4;
        let left = pixels[(row * r) * 4];
        let right = pixels[(row * r + r - 1) * 4];
        assert_eq!(left, right);
    }

    #[test]
    fn multi_ply_varies_along_its_length_when_twisted() {
        // The whole point of twist: unlike a single ply, a twisted 2+ ply
        // thread's cross-section at a fixed height genuinely differs as you
        // move along its length. Checks the full spread across a row
        // rather than two specific columns — an orbiting strand's period
        // for N plies is 2*PI/N, not 2*PI, so two arbitrarily-chosen
        // columns can land exactly half a strand-period apart and
        // coincidentally read identical without the pattern actually being
        // constant (hit this for real while writing this test).
        let pixels = render(&ThreadShapeParams {
            ply_count: 2,
            twist_cycles: 2,
            thickness: 0.6,
            ..ThreadShapeParams::default()
        });
        let r = RESOLUTION as usize;
        let row = r / 2 + r / 6; // off-center, where an orbiting strand sweeps through
        let row_values: Vec<u8> = (0..r).map(|x| pixels[(row * r + x) * 4]).collect();
        let min = *row_values.iter().min().unwrap();
        let max = *row_values.iter().max().unwrap();
        assert!(
            max - min > 20,
            "a twisted multi-ply thread should vary along its own length (min={min}, max={max})"
        );
    }

    /// A screenshot once showed a real black void down the bundle's
    /// centerline, sustained over a wide stretch of its length — not a
    /// single-pixel touch where two strands sized to exactly meet happen
    /// to graze zero at the precise instant they're tangent (physically
    /// correct, and a much narrower thing), but a literal disconnect
    /// splitting the thread into two unrelated ribbons over a visible
    /// span. Checks the *width* of any near-zero stretch rather than
    /// requiring every single pixel stay strictly above zero — the
    /// weighted-average smooth max this shader now uses (unlike an
    /// earlier log-sum-exp version) is a true convex combination of the
    /// strands' own edges, so it can legitimately touch the hard max's own
    /// zero at an isolated tangent point without that being the bug this
    /// guards against.
    #[test]
    fn two_ply_never_shows_a_sustained_gap_down_the_centerline() {
        let pixels = render(&ThreadShapeParams {
            ply_count: 2,
            thickness: 0.73,
            roundness: 0.32,
            twist_cycles: 2,
            twist_direction: TwistDirection::Z,
        });
        let r = RESOLUTION as usize;
        let row = r / 2;
        let row_values: Vec<u8> = (0..r).map(|x| pixels[(row * r + x) * 4]).collect();
        let near_zero_run = row_values
            .iter()
            .fold((0usize, 0usize), |(longest, current), &v| {
                let current = if v <= 5 { current + 1 } else { 0 };
                (longest.max(current), current)
            })
            .0;
        let max_allowed = r / 32;
        assert!(
            near_zero_run <= max_allowed,
            "the bundle's centerline stayed near-black for {near_zero_run} pixels in a row \
             (resolution {r}, allowed up to {max_allowed}) — that's a sustained void splitting \
             the thread, not a single tangent touch"
        );
    }

    #[test]
    fn zero_twist_cycles_is_constant_along_its_length_even_with_multiple_plies() {
        let pixels = render(&ThreadShapeParams {
            ply_count: 3,
            twist_cycles: 0,
            ..ThreadShapeParams::default()
        });
        let r = RESOLUTION as usize;
        let row = r / 2;
        let left = pixels[(row * r) * 4];
        let right = pixels[(row * r + r - 1) * 4];
        assert_eq!(left, right);
    }

    #[test]
    fn is_seamless_along_its_own_length() {
        // twist_cycles must be a whole number of rotations across this
        // texture's own width, or Fabric Generator's per-cell wraparound
        // sampling would show a visible seam where the twist jumps. Uses
        // the same calibrated seam-vs-interior check every other periodic
        // node's tiling test uses, not a flat small-delta threshold — a
        // sharp strand edge crossing the seam at a slightly different pixel
        // center than its interior neighbors is the same ordinary texel-
        // quantization noise every periodic generator here has, not a real
        // discontinuity.
        let pixels = render(&ThreadShapeParams {
            ply_count: 3,
            twist_cycles: 3,
            thickness: 0.7,
            ..ThreadShapeParams::default()
        });
        crate::test_support::assert_tiles_seamlessly(&pixels, RESOLUTION);
    }

    #[test]
    fn twist_direction_mirrors_the_spiral() {
        let s_twist = render(&ThreadShapeParams {
            ply_count: 3,
            twist_cycles: 2,
            twist_direction: TwistDirection::S,
            thickness: 0.6,
            ..ThreadShapeParams::default()
        });
        let z_twist = render(&ThreadShapeParams {
            ply_count: 3,
            twist_cycles: 2,
            twist_direction: TwistDirection::Z,
            thickness: 0.6,
            ..ThreadShapeParams::default()
        });
        assert_ne!(s_twist, z_twist);
    }

    /// The old plain-`max()` combine was symmetric at exactly 2 plies (see
    /// the module doc comment's now-superseded note): flipping direction
    /// just swapped which strand was which, with no "which one is on top"
    /// cue to tell the difference. The depth-ordered occlusion added to fix
    /// the moire-lattice bug happens to also resolve this — a strand's
    /// depth (`cos` of its orbit angle) and its position (`sin` of the same
    /// angle) don't swap together under a direction flip the way position
    /// alone did, so which strand occludes which now differs too. Confirmed
    /// here rather than assumed.
    #[test]
    fn twist_direction_also_differs_at_two_ply_now_that_strands_occlude() {
        let s_twist = render(&ThreadShapeParams {
            ply_count: 2,
            twist_cycles: 2,
            twist_direction: TwistDirection::S,
            thickness: 0.6,
            ..ThreadShapeParams::default()
        });
        let z_twist = render(&ThreadShapeParams {
            ply_count: 2,
            twist_cycles: 2,
            twist_direction: TwistDirection::Z,
            thickness: 0.6,
            ..ThreadShapeParams::default()
        });
        assert_ne!(s_twist, z_twist);
    }

    #[test]
    fn every_twist_direction_name_round_trips() {
        for name in TwistDirection::NAMES {
            assert!(
                TwistDirection::from_name(name).is_some(),
                "{name:?} should parse"
            );
        }
        assert!(TwistDirection::from_name("Nonsense").is_none());
    }
}
