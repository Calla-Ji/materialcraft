// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Thread.
//!
//! Display name: "Thread". Tooltip: "Plied yarn built from real textile
//! geometry — wire its Height output into Fabric Generator's Thread input
//! to control what its threads look like."
//!
//! # History: five rounds of tuning a 1D cross-section profile, then a full
//! rebuild from real geometry
//!
//! Every earlier version of this node (see git history) modeled a ply as a
//! 1D brightness profile in Y that just moved up and down as X advanced —
//! not an actual round tube. That's a flat ribbon, not yarn: its own width
//! necessarily shrinks to zero wherever two plies cross, since the
//! "combine" step was always some kind of blend of two 1D profiles, not a
//! real occlusion test between two solid 3D shapes. Each round of tuning
//! (moire lattice -> depth cue -> smooth union -> depth-gated shape ->
//! transition sharpness) made that ribbon look better within its own
//! limits, but the limits themselves — constant width is impossible for a
//! profile pinched between two moving centers — were never fixable by
//! tuning.
//!
//! Rebuilt from scratch from `THREAD_NODE_SPEC.md` (owner-provided, derived
//! from her textile reference book's yarn-construction chapter and a
//! ground-truth Python reference renderer, `thread_reference.py`): every
//! ply is a genuine round tube of constant radius `r`, whose *centerline*
//! (not its own width) follows a helix around the thread axis. Which ply is
//! visible where is a real top-surface-height occlusion test (the highest
//! tube wins, same idea as a z-buffer), not a blend — so a ply's own width
//! never changes, and two plies crossing shows one tube physically in front
//! of the other, the way real rope actually looks. See the shader's own
//! doc comment for the exact geometry and the analytic tube/plane
//! intersection this depends on.
//!
//! `packing()` and `turns_from_angle()` below are ported 1:1 from
//! `thread_reference.py`'s functions of the same name — their own doc
//! comments carry the derivation. The shader's per-pixel tube-occlusion
//! loop, fiber groove carving, and cable (helix-of-helices) math are a
//! direct WGSL port of `thread_reference.py`'s `render()`/`build_strands()`,
//! verified against its own golden output images (`mc_cli`'s golden tests),
//! not just assumed correct from reading the spec.
//!
//! One deliberate deviation from the reference script, called out rather
//! than silently diverging: `thread_reference.py` uses numpy's RNG for each
//! stacked thread row's random twist-phase offset (purely so rows don't all
//! start their twist at the same point — cosmetic, not structural; it
//! doesn't affect tiling, geometry, or how Fabric Generator consumes this
//! node's output — see `phase0` in the shader). Reproducing numpy's PCG64
//! bit-for-bit in WGSL isn't worth it for a cosmetic offset, so the shader
//! uses this codebase's existing `hash2` helper instead, and the checked-in
//! golden images were regenerated from a copy of the reference script using
//! that same hash so the two stay comparable.

use crate::shaders::full_source;

const NODE_KIND: &str = "thread_shape";

/// Ply twist direction — the book's S/Z convention (THREAD_NODE_SPEC.md
/// §3.3). With the thread horizontal and image-Y pointing down, S makes the
/// front strands run `/` (up-right) and Z makes them run `\` (down-right) —
/// the mirror image of how S/Z read with the yarn held vertical, which is
/// how every textile reference (including the owner's book) actually
/// defines them; rotating the yarn 90° to horizontal flips the apparent
/// slope. `as_sign` below is where that convention actually lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwistDirection {
    S,
    Z,
}

impl TwistDirection {
    pub const NAMES: [&'static str; 2] = ["S", "Z"];

    /// +1 for S, -1 for Z — see the type's own doc comment for why this
    /// (not the reverse) is the correct sign for a *horizontal* thread.
    fn as_sign(self) -> i32 {
        match self {
            TwistDirection::S => 1,
            TwistDirection::Z => -1,
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

/// How the fibers inside each ply twist relative to the ply itself
/// (THREAD_NODE_SPEC.md §4). Real plied yarn is almost always "Balanced":
/// the single-yarn fibers twist opposite to the ply twist, so the two
/// slopes cancel and the visible fiber grooves on top of a ply read as
/// nearly parallel to the thread's own length — a textile-construction fact
/// from the owner's reference book, not a visual style choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiberMode {
    Balanced,
    Crepe,
    None,
}

impl FiberMode {
    pub const NAMES: [&'static str; 3] = ["Balanced", "Crepe", "None"];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Balanced" => Some(FiberMode::Balanced),
            "Crepe" => Some(FiberMode::Crepe),
            "None" => Some(FiberMode::None),
            _ => None,
        }
    }
}

/// N equal round plies packed touching each other, and touching the outer
/// circle of radius `big_r`, inside a yarn/group/cable of that radius
/// (THREAD_NODE_SPEC.md §3.1). Returns `(a, r)`: `a` is the distance from
/// the shared axis to each ply's own center, `r` is each ply's own radius.
///
/// Derivation (same as `thread_reference.py`'s `packing()`): neighboring
/// ply centers sit `2r` apart on a circle of radius `a`, so
/// `sin(pi/N) = r/a`, giving `a = r/sin(pi/N)`; combined with
/// `big_r = a + r` (the outer circle touches each ply's own outer edge)
/// solves to the formula below. `N <= 1` is the single-yarn case: one "ply"
/// at the axis (`a = 0`) filling the whole radius (`r = big_r`).
pub fn packing(big_r: f32, n: u32) -> (f32, f32) {
    if n <= 1 {
        return (0.0, big_r);
    }
    let s = (std::f32::consts::PI / n as f32).sin();
    let r = big_r / (1.0 + 1.0 / s);
    let a = r / s;
    (a, r)
}

/// Whole helix turns across the tile (the tile is 1 unit wide) for helix
/// angle `angle_deg` around radius `a` (THREAD_NODE_SPEC.md §3.2):
/// `tan(alpha) = 2*pi*a*k`, so `k = tan(alpha) / (2*pi*a)`. Rounded to the
/// nearest integer >= 1 so the helix completes a whole number of turns
/// across the tile and the thread tiles seamlessly in X — the actual angle
/// shown to the artist is approximate because of this rounding, same as
/// `thread_reference.py`'s `turns_from_angle()`.
pub fn turns_from_angle(angle_deg: f32, a: f32) -> u32 {
    if a <= 1e-9 {
        return 1;
    }
    let k = (angle_deg.to_radians().tan() / (2.0 * std::f32::consts::PI * a)).round();
    (k as i64).max(1) as u32
}

pub struct ThreadShapeParams {
    pub threads: u32,
    pub thickness: f32,
    pub plies: u32,
    pub ply_direction: TwistDirection,
    pub twist_angle: f32,
    pub fiber_mode: FiberMode,
    pub fibers_per_ply: u32,
    pub fiber_depth: f32,
    pub cable_groups: u32,
    pub cable_angle: f32,
    pub seed: u32,
}

impl Default for ThreadShapeParams {
    fn default() -> Self {
        Self {
            threads: 4,
            thickness: 0.8,
            plies: 2,
            ply_direction: TwistDirection::S,
            twist_angle: 30.0,
            fiber_mode: FiberMode::Balanced,
            fibers_per_ply: 7,
            fiber_depth: 0.12,
            cable_groups: 0,
            cable_angle: 25.0,
            seed: 1,
        }
    }
}

impl ThreadShapeParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let float = |name: &str, default: f32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(default)
        };
        Self {
            threads: params
                .get("threads")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.threads),
            thickness: float("thickness", defaults.thickness),
            plies: params
                .get("plies")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.plies),
            ply_direction: params
                .get("ply_direction")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => TwistDirection::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.ply_direction),
            twist_angle: float("twist_angle", defaults.twist_angle),
            fiber_mode: params
                .get("fiber_mode")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => FiberMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.fiber_mode),
            fibers_per_ply: params
                .get("fibers_per_ply")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.fibers_per_ply),
            fiber_depth: float("fiber_depth", defaults.fiber_depth),
            cable_groups: params
                .get("cable_groups")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(0) as u32)
                .unwrap_or(defaults.cable_groups),
            cable_angle: float("cable_angle", defaults.cable_angle),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
        }
    }
}

/// Every value `dispatch` needs that isn't a direct, per-pixel-identical
/// copy of an artist param — the one-time geometry solve
/// (THREAD_NODE_SPEC.md §3.1-§3.4) done once per dispatch in Rust rather
/// than redundantly on every pixel in WGSL. Kept as its own type (not
/// inlined into `pack_params`) so `ThreadShapeParams::derive` can be unit
/// tested directly against the spec's own check values, independent of the
/// GPU.
struct Derived {
    r0: f32,
    a: f32,
    r: f32,
    k_p: u32,
    s_ply: i32,
    fiber_sign: i32,
    fiber_turns: f32,
    has_cable: bool,
    s_cable: i32,
    k_c: u32,
    big_a: f32,
}

impl ThreadShapeParams {
    /// Mirrors `thread_reference.py`'s `render()`/`build_strands()` setup
    /// exactly: thread radius from `fill`/`threads`, ply packing (cable
    /// packs groups first, then plies within each group), turns from the
    /// artist-facing angles, and the fiber twist sign/turns balanced (or
    /// not) against the ply twist (§4).
    fn derive(&self) -> Derived {
        let row_h = 1.0 / self.threads as f32;
        let r0 = 0.5 * self.thickness.clamp(0.0, 1.0) * row_h;
        let s_ply = self.ply_direction.as_sign();

        let (a, r, k_p, has_cable, s_cable, k_c, big_a) = if self.cable_groups >= 2 {
            let (big_a, rg) = packing(r0, self.cable_groups);
            let (a, r) = packing(rg, self.plies);
            let s_cable = -s_ply;
            let k_c = turns_from_angle(self.cable_angle, big_a);
            let k_p = turns_from_angle(self.twist_angle, a);
            (a, r, k_p, true, s_cable, k_c, big_a)
        } else {
            let (a, r) = packing(r0, self.plies);
            let k_p = turns_from_angle(self.twist_angle, if self.plies > 1 { a } else { r0 });
            (a, r, k_p, false, 1, 0, 0.0)
        };

        // Balanced: fiber helix opposite the ply twist, with
        // k_f = a*k/r so the visible fiber slope cancels the ply slope
        // (spec §4's "why balanced cancels" derivation). Crepe is the same
        // direction as the ply twist, over-twisted (1.6x). Both rounded to
        // a whole number of turns so the groove tiles in X, same reasoning
        // as `k_p`/`k_c` above.
        let (fiber_sign, fiber_turns) = match self.fiber_mode {
            FiberMode::None => (0, 0.0),
            _ if r <= 0.0 => (0, 0.0),
            _ if a <= 1e-9 => (s_ply, k_p as f32), // single yarn: fibers follow the yarn twist
            FiberMode::Balanced => {
                let kf = a * k_p as f32 / r;
                (-s_ply, kf.round().max(1.0))
            }
            FiberMode::Crepe => {
                let kf = 1.6 * a * k_p as f32 / r;
                (s_ply, kf.round().max(1.0))
            }
        };

        Derived {
            r0,
            a,
            r,
            k_p,
            s_ply,
            fiber_sign,
            fiber_turns,
            has_cable,
            s_cable,
            k_c,
            big_a,
        }
    }
}

fn pack_params(resolution: u32, params: &ThreadShapeParams, d: &Derived) -> [u8; 80] {
    let mut bytes = [0u8; 80];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.threads.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.plies.to_le_bytes());
    bytes[12..16].copy_from_slice(&d.s_ply.to_le_bytes());
    bytes[16..20].copy_from_slice(&d.k_p.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.fibers_per_ply.to_le_bytes());
    bytes[24..28].copy_from_slice(&(d.has_cable as u32).to_le_bytes());
    bytes[28..32].copy_from_slice(&params.cable_groups.to_le_bytes());
    bytes[32..36].copy_from_slice(&d.s_cable.to_le_bytes());
    bytes[36..40].copy_from_slice(&d.k_c.to_le_bytes());
    bytes[40..44].copy_from_slice(&params.seed.to_le_bytes());
    bytes[44..48].copy_from_slice(&d.fiber_sign.to_le_bytes());
    bytes[48..52].copy_from_slice(&d.r0.to_le_bytes());
    bytes[52..56].copy_from_slice(&d.a.to_le_bytes());
    bytes[56..60].copy_from_slice(&d.r.to_le_bytes());
    bytes[60..64].copy_from_slice(&d.fiber_turns.to_le_bytes());
    bytes[64..68].copy_from_slice(&params.fiber_depth.to_le_bytes());
    bytes[68..72].copy_from_slice(&d.big_a.to_le_bytes());
    bytes
}

/// Where `dispatch` stores each of Thread's four outputs — see
/// `mc_nodes::defs`'s `thread_shape` `NodeDef` for what each one means to
/// an artist. Height must stay first in the `NodeDef`'s own `outputs` list
/// (THREAD_NODE_SPEC.md §6) so `mc_nodes::registry::resolve_texture_key`'s
/// "first declared output" default keeps showing Height in the node's
/// thumbnail/2D View/3D viewport, and so Fabric Generator's Thread input —
/// wired before this node had more than one output — keeps working when a
/// user reconnects it.
pub struct ThreadOutputKeys<'a> {
    pub height: &'a str,
    pub ply_id: &'a str,
    pub fiber: &'a str,
    pub mask: &'a str,
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    keys: ThreadOutputKeys,
    resolution: u32,
    params: &ThreadShapeParams,
) {
    let shader_source = full_source("thread_shape.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let derived = params.derive();
    let param_bytes = pack_params(resolution, params, &derived);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("thread_shape_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let height_view = pool
        .get_or_create(
            &gpu.device,
            keys.height,
            resolution,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .clone();
    let ply_id_view = pool
        .get_or_create(
            &gpu.device,
            keys.ply_id,
            resolution,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .clone();
    let fiber_view = pool
        .get_or_create(
            &gpu.device,
            keys.fiber,
            resolution,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .clone();
    let mask_view = pool
        .get_or_create(
            &gpu.device,
            keys.mask,
            resolution,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .clone();

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
                resource: wgpu::BindingResource::TextureView(&height_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&ply_id_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&fiber_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&mask_view),
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
    use super::{
        dispatch, packing, turns_from_angle, FiberMode, ThreadOutputKeys, ThreadShapeParams,
        TwistDirection,
    };
    use crate::test_support::RESOLUTION;

    struct Rendered {
        height: Vec<u8>,
        ply_id: Vec<u8>,
        fiber: Vec<u8>,
        mask: Vec<u8>,
    }

    fn render(params: &ThreadShapeParams) -> Rendered {
        render_at(params, RESOLUTION)
    }

    fn render_at(params: &ThreadShapeParams, resolution: u32) -> Rendered {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(
            &mut gpu,
            &mut pool,
            ThreadOutputKeys {
                height: "height",
                ply_id: "ply_id",
                fiber: "fiber",
                mask: "mask",
            },
            resolution,
            params,
        );
        Rendered {
            height: gpu.read_rgba8(pool.texture("height").unwrap(), resolution),
            ply_id: gpu.read_rgba8(pool.texture("ply_id").unwrap(), resolution),
            fiber: gpu.read_rgba8(pool.texture("fiber").unwrap(), resolution),
            mask: gpu.read_rgba8(pool.texture("mask").unwrap(), resolution),
        }
    }

    // --- packing() / turns_from_angle(): THREAD_NODE_SPEC.md §8 item 6 ---

    #[test]
    fn packing_matches_the_spec_check_values() {
        let (a, r) = packing(1.0, 1);
        assert_eq!((a, r), (0.0, 1.0), "single yarn: a=0, r=R");

        let (a, r) = packing(1.0, 2);
        assert!((a - 0.5).abs() < 1e-4 && (r - 0.5).abs() < 1e-4, "{a} {r}");

        let (a, r) = packing(1.0, 3);
        assert!(
            (a - 0.536).abs() < 1e-3 && (r - 0.464).abs() < 1e-3,
            "{a} {r}"
        );

        let (a, r) = packing(1.0, 4);
        assert!(
            (a - 0.586).abs() < 1e-3 && (r - 0.414).abs() < 1e-3,
            "{a} {r}"
        );
    }

    #[test]
    fn packing_always_reconstructs_the_outer_radius() {
        for n in 1..=8u32 {
            let (a, r) = packing(2.5, n);
            assert!(
                (a + r - 2.5).abs() < 1e-4,
                "n={n}: a+r should equal the outer radius (a={a}, r={r})"
            );
        }
    }

    #[test]
    fn turns_from_angle_is_always_at_least_one() {
        assert_eq!(turns_from_angle(0.1, 0.5), 1);
        assert_eq!(turns_from_angle(5.0, 100.0), 1);
    }

    #[test]
    fn turns_from_angle_grows_with_a_steeper_angle() {
        let low = turns_from_angle(15.0, 0.1);
        let high = turns_from_angle(45.0, 0.1);
        assert!(
            high > low,
            "a steeper helix angle should need more turns (low={low}, high={high})"
        );
    }

    // --- basic rendering sanity ---

    #[test]
    fn is_deterministic() {
        let params = ThreadShapeParams::default();
        let a = render(&params);
        let b = render(&params);
        assert_eq!(a.height, b.height);
        assert_eq!(a.ply_id, b.ply_id);
        assert_eq!(a.fiber, b.fiber);
        assert_eq!(a.mask, b.mask);
    }

    #[test]
    fn mask_is_set_only_where_a_thread_is_actually_drawn() {
        let r = render(&ThreadShapeParams::default());
        let any_set = r.mask.iter().step_by(4).any(|&v| v > 0);
        let any_clear = r.mask.iter().step_by(4).any(|&v| v == 0);
        assert!(any_set, "mask should be set somewhere a thread is drawn");
        assert!(
            any_clear,
            "mask should be clear in the gaps between stacked thread rows"
        );
    }

    #[test]
    fn height_is_zero_wherever_mask_is_zero() {
        let r = render(&ThreadShapeParams::default());
        for i in (0..r.mask.len()).step_by(4) {
            if r.mask[i] == 0 {
                assert_eq!(
                    r.height[i], 0,
                    "background (mask=0) should always be height=0 too"
                );
            }
        }
    }

    // --- tiling: THREAD_NODE_SPEC.md §8 item 2 ---

    #[test]
    fn tiles_seamlessly_in_x() {
        let r = render(&ThreadShapeParams::default());
        crate::test_support::assert_tiles_seamlessly(&r.height, RESOLUTION);
    }

    #[test]
    fn tiles_seamlessly_with_cable_enabled() {
        let r = render(&ThreadShapeParams {
            cable_groups: 2,
            threads: 3,
            twist_angle: 35.0,
            ..ThreadShapeParams::default()
        });
        crate::test_support::assert_tiles_seamlessly(&r.height, RESOLUTION);
    }

    // --- S/Z direction: THREAD_NODE_SPEC.md §8 item 4 ---

    /// For a 2-ply, 1-row thread, tracks the front ply's centerline across
    /// X by finding the brightest row in each column (the winning strand's
    /// surface is always the brightest point in its column, since it's the
    /// one facing the viewer). For S, that position should trend upward
    /// (decreasing row index) while the front ply is visible; for Z,
    /// downward — the exact opposite of what "S" and "Z" mean with the
    /// yarn held vertical, which is the point of `TwistDirection::as_sign`'s
    /// own doc comment.
    #[test]
    fn s_and_z_twist_produce_mirrored_front_ply_motion() {
        let params = |dir: TwistDirection| ThreadShapeParams {
            threads: 1,
            thickness: 0.7,
            plies: 2,
            ply_direction: dir,
            twist_angle: 30.0,
            fiber_mode: FiberMode::None,
            ..ThreadShapeParams::default()
        };
        let s = render(&params(TwistDirection::S));
        let z = render(&params(TwistDirection::Z));
        let r = RESOLUTION as usize;

        let brightest_row = |pixels: &[u8], x: usize| -> usize {
            (0..r)
                .max_by_key(|&y| pixels[(y * r + x) * 4])
                .expect("non-empty column")
        };

        // A short run near the start of the tile, away from the X seam.
        let xs: Vec<usize> = (r / 16..r / 16 + r / 8).collect();
        let s_delta: i64 = brightest_row(&s.height, *xs.last().unwrap()) as i64
            - brightest_row(&s.height, xs[0]) as i64;
        let z_delta: i64 = brightest_row(&z.height, *xs.last().unwrap()) as i64
            - brightest_row(&z.height, xs[0]) as i64;
        assert!(
            s_delta.signum() != 0 && s_delta.signum() == -z_delta.signum(),
            "S and Z should move the front ply's centerline in opposite directions \
             (s_delta={s_delta}, z_delta={z_delta})"
        );
    }

    // --- constant ply width: THREAD_NODE_SPEC.md §8 item 5 (the ribbon-bug regression test) ---

    /// For a fiber-free 2-ply thread, the front ply's own width — measured
    /// as a short vertical run of bright pixels around its own centerline —
    /// should stay close to constant across X, never pinching toward zero.
    /// This is the direct regression test for the original bug every
    /// earlier version of this node had (see the module doc comment): a
    /// flat 1D profile's width necessarily shrank to nothing wherever two
    /// plies crossed; a real tube's width doesn't.
    #[test]
    fn front_ply_width_stays_roughly_constant() {
        let r_res = 256u32;
        let rendered = render_at(
            &ThreadShapeParams {
                threads: 1,
                thickness: 0.8,
                plies: 2,
                twist_angle: 30.0,
                fiber_mode: FiberMode::None,
                ..ThreadShapeParams::default()
            },
            r_res,
        );
        let r = r_res as usize;
        let mut widths = Vec::new();
        for x in (0..r).step_by(4) {
            let col: Vec<u8> = (0..r).map(|y| rendered.height[(y * r + x) * 4]).collect();
            let peak = col.iter().copied().max().unwrap_or(0);
            if peak < 40 {
                continue; // no strand clearly dominant at this column, skip
            }
            let threshold = peak / 2;
            let width = col.iter().filter(|&&v| v >= threshold).count();
            widths.push(width);
        }
        let min = *widths.iter().min().unwrap();
        let max = *widths.iter().max().unwrap();
        assert!(
            min > 0,
            "the front ply's own width should never pinch to zero (min={min}, max={max})"
        );
        assert!(
            (max as f32) < (min as f32) * 3.0,
            "the front ply's width should stay roughly constant across X, not balloon or \
             pinch (min={min}, max={max})"
        );
    }

    // --- balanced vs crepe fibers: THREAD_NODE_SPEC.md §8 item 7 ---

    #[test]
    fn fiber_mode_none_leaves_fiber_output_at_full_value_wherever_hit() {
        // Matches thread_reference.py exactly: `ridge` with no groove is 1
        // (a flat, ungrooved surface — full roughness/AO value), not 0 —
        // 0 would mean "deepest possible groove", which is the opposite of
        // what "no fiber groove at all" should mean.
        let r = render(&ThreadShapeParams {
            fiber_mode: FiberMode::None,
            ..ThreadShapeParams::default()
        });
        for (i, &m) in r.mask.iter().enumerate().step_by(4) {
            if m > 0 {
                assert_eq!(
                    r.fiber[i], 255,
                    "Fiber output should be at full value (no groove carved) wherever a \
                     thread is actually drawn, with Fiber Twist None"
                );
            }
        }
    }

    #[test]
    fn balanced_and_crepe_fibers_produce_different_output() {
        let balanced = render(&ThreadShapeParams {
            fiber_mode: FiberMode::Balanced,
            ..ThreadShapeParams::default()
        });
        let crepe = render(&ThreadShapeParams {
            fiber_mode: FiberMode::Crepe,
            ..ThreadShapeParams::default()
        });
        assert_ne!(
            balanced.fiber, crepe.fiber,
            "Balanced and Crepe fiber twist should look different"
        );
    }

    // --- ply ID ---

    #[test]
    fn ply_id_output_is_blank_for_a_single_ply() {
        // One ply normalizes to (0+1)/1 = 1.0 everywhere it's hit — still a
        // single flat value, just not literally 0, so check it's constant
        // rather than blank.
        let r = render(&ThreadShapeParams {
            plies: 1,
            ..ThreadShapeParams::default()
        });
        let hit_values: Vec<u8> = r
            .ply_id
            .iter()
            .step_by(4)
            .zip(r.mask.iter().step_by(4))
            .filter(|(_, &m)| m > 0)
            .map(|(&v, _)| v)
            .collect();
        assert!(!hit_values.is_empty());
        assert!(
            hit_values.iter().all(|&v| v == hit_values[0]),
            "a single ply's Ply ID should be the same value everywhere it's drawn"
        );
    }

    #[test]
    fn ply_id_output_varies_across_multiple_plies() {
        let r = render(&ThreadShapeParams {
            plies: 3,
            fiber_mode: FiberMode::None,
            ..ThreadShapeParams::default()
        });
        let distinct: std::collections::HashSet<u8> = r
            .ply_id
            .iter()
            .step_by(4)
            .zip(r.mask.iter().step_by(4))
            .filter(|(_, &m)| m > 0)
            .map(|(&v, _)| v)
            .collect();
        assert!(
            distinct.len() >= 2,
            "3 plies should show at least 2 distinct Ply ID values where they're drawn \
             (got {})",
            distinct.len()
        );
    }

    // --- cable mode ---

    #[test]
    fn cable_mode_changes_the_output() {
        let plain = render(&ThreadShapeParams::default());
        let cabled = render(&ThreadShapeParams {
            cable_groups: 2,
            threads: 3,
            twist_angle: 35.0,
            ..ThreadShapeParams::default()
        });
        assert_ne!(plain.height, cabled.height);
    }

    // --- misc round-trips ---

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

    #[test]
    fn every_fiber_mode_name_round_trips() {
        for name in FiberMode::NAMES {
            assert!(
                FiberMode::from_name(name).is_some(),
                "{name:?} should parse"
            );
        }
        assert!(FiberMode::from_name("Nonsense").is_none());
    }
}
