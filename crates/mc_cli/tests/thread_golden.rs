// SPDX-License-Identifier: MIT OR Apache-2.0
//! Golden-image tests for the Thread node (THREAD_NODE_SPEC.md §8 item 1):
//! compares the WGSL renderer against `thread_reference.py`, the owner's
//! ground-truth CPU reference, rather than the usual whole-graph Base Color
//! comparison `golden.rs` does.
//!
//! Separate from `golden.rs` for two reasons: these PNGs are 16-bit
//! single-channel (height/ply_id/fiber/mask), not 8-bit RGBA, so they need
//! their own decode path; and the acceptance criterion the spec defines is
//! statistical (mean absolute difference, plus a tolerance on how many
//! pixels may be off by more) rather than `golden.rs`'s single max-diff
//! check — silhouette edges differing by a pixel between two independent
//! implementations of the same continuous geometry is expected and
//! harmless, so a handful of edge pixels being off by more than the mean
//! shouldn't fail the whole case the way it would for `golden.rs`'s
//! flat-color graphs.
//!
//! The checked-in PNGs under `tests/golden/thread/` were regenerated from a
//! copy of `thread_reference.py` with one line changed — its per-row
//! twist-phase offset uses this codebase's own `hash2` instead of numpy's
//! RNG (thread_shape.rs's module doc comment explains why) — everything
//! else is identical to the spec package's own reference script and
//! `params.json`.

use std::path::Path;

/// Mean absolute difference (THREAD_NODE_SPEC.md §8 item 1).
const MEAN_ABS_DIFF_TOLERANCE: f32 = 0.02;
/// No more than this fraction of pixels may be off by more than
/// `PER_PIXEL_TOLERANCE`.
const OUTLIER_FRACTION_TOLERANCE: f32 = 0.01;
const PER_PIXEL_TOLERANCE: f32 = 0.1;

const RESOLUTION: u32 = 512;

struct Case {
    name: &'static str,
    params: mc_nodes::thread_shape::ThreadShapeParams,
}

fn case(name: &'static str, params: mc_nodes::thread_shape::ThreadShapeParams) -> Case {
    Case { name, params }
}

fn cases() -> Vec<Case> {
    use mc_nodes::thread_shape::{FiberMode, ThreadShapeParams, TwistDirection};

    vec![
        case(
            "ply2_S_30",
            ThreadShapeParams {
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
            },
        ),
        case(
            "ply2_Z_30",
            ThreadShapeParams {
                threads: 4,
                thickness: 0.8,
                plies: 2,
                ply_direction: TwistDirection::Z,
                twist_angle: 30.0,
                fiber_mode: FiberMode::Balanced,
                fibers_per_ply: 7,
                fiber_depth: 0.12,
                cable_groups: 0,
                cable_angle: 25.0,
                seed: 1,
            },
        ),
        case(
            "ply3_S_30",
            ThreadShapeParams {
                threads: 4,
                thickness: 0.8,
                plies: 3,
                ply_direction: TwistDirection::S,
                twist_angle: 30.0,
                fiber_mode: FiberMode::Balanced,
                fibers_per_ply: 7,
                fiber_depth: 0.12,
                cable_groups: 0,
                cable_angle: 25.0,
                seed: 1,
            },
        ),
        case(
            "ply4_S_45",
            ThreadShapeParams {
                threads: 4,
                thickness: 0.8,
                plies: 4,
                ply_direction: TwistDirection::S,
                twist_angle: 45.0,
                fiber_mode: FiberMode::Balanced,
                fibers_per_ply: 7,
                fiber_depth: 0.12,
                cable_groups: 0,
                cable_angle: 25.0,
                seed: 1,
            },
        ),
        case(
            "single_1ply",
            ThreadShapeParams {
                threads: 4,
                thickness: 0.8,
                plies: 1,
                ply_direction: TwistDirection::S,
                twist_angle: 30.0,
                fiber_mode: FiberMode::Balanced,
                fibers_per_ply: 12,
                fiber_depth: 0.12,
                cable_groups: 0,
                cable_angle: 25.0,
                seed: 1,
            },
        ),
        case(
            "cable_2x2",
            ThreadShapeParams {
                threads: 3,
                thickness: 0.8,
                plies: 2,
                ply_direction: TwistDirection::S,
                twist_angle: 35.0,
                fiber_mode: FiberMode::Balanced,
                fibers_per_ply: 5,
                fiber_depth: 0.12,
                cable_groups: 2,
                cable_angle: 25.0,
                seed: 1,
            },
        ),
    ]
}

/// Decodes a 16-bit grayscale golden into normalized `[0,1]` `f32`s.
fn load_golden_normalized(path: &Path) -> (Vec<f32>, u32, u32) {
    let img = image::open(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()))
        .into_luma16();
    let (w, h) = img.dimensions();
    let pixels = img
        .into_raw()
        .into_iter()
        .map(|v| v as f32 / 65535.0)
        .collect();
    (pixels, w, h)
}

/// Compares `rendered` (8-bit, as read back from the GPU) against the
/// 16-bit golden at `golden_path`, per THREAD_NODE_SPEC.md §8 item 1's
/// exact acceptance criteria.
fn assert_matches_golden(rendered_rgba8: &[u8], golden_path: &Path, channel: &str) {
    assert!(
        golden_path.exists(),
        "missing golden image {} — see tests/golden/thread/ and thread_golden.rs's own doc \
         comment for how these are regenerated",
        golden_path.display()
    );
    let (golden, gw, gh) = load_golden_normalized(golden_path);
    assert_eq!(
        (gw, gh),
        (RESOLUTION, RESOLUTION),
        "{}: resolution mismatch",
        golden_path.display()
    );

    let rendered: Vec<f32> = rendered_rgba8
        .iter()
        .step_by(4) // R channel only — height/ply_id/fiber/mask are all grayscale (R=G=B)
        .map(|&v| v as f32 / 255.0)
        .collect();
    assert_eq!(rendered.len(), golden.len());

    let diffs: Vec<f32> = rendered
        .iter()
        .zip(golden.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();
    let mean_abs_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let outliers = diffs.iter().filter(|&&d| d > PER_PIXEL_TOLERANCE).count();
    let outlier_fraction = outliers as f32 / diffs.len() as f32;

    assert!(
        mean_abs_diff < MEAN_ABS_DIFF_TOLERANCE,
        "{} ({channel}): mean abs diff {mean_abs_diff:.4} exceeds tolerance \
         {MEAN_ABS_DIFF_TOLERANCE}",
        golden_path.display()
    );
    assert!(
        outlier_fraction <= OUTLIER_FRACTION_TOLERANCE,
        "{} ({channel}): {outliers} of {} pixels ({:.2}%) differ by more than \
         {PER_PIXEL_TOLERANCE}, exceeding the {:.0}% allowance",
        golden_path.display(),
        diffs.len(),
        outlier_fraction * 100.0,
        OUTLIER_FRACTION_TOLERANCE * 100.0,
    );
}

#[test]
fn thread_matches_the_python_reference_renderer() {
    let golden_dir = Path::new("../../tests/golden/thread");

    for case in cases() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let height_key = format!("{}_height", case.name);
        let ply_id_key = format!("{}_ply_id", case.name);
        let fiber_key = format!("{}_fiber", case.name);
        let mask_key = format!("{}_mask", case.name);

        mc_nodes::thread_shape::dispatch(
            &mut gpu,
            &mut pool,
            mc_nodes::thread_shape::ThreadOutputKeys {
                height: &height_key,
                ply_id: &ply_id_key,
                fiber: &fiber_key,
                mask: &mask_key,
            },
            RESOLUTION,
            &case.params,
        );

        for (key, channel) in [
            (&height_key, "height"),
            (&ply_id_key, "ply_id"),
            (&fiber_key, "fiber"),
            (&mask_key, "mask"),
        ] {
            let texture = pool.texture(key).expect("evaluated texture missing");
            let pixels = gpu.read_rgba8(texture, RESOLUTION);
            let golden_path = golden_dir.join(format!("{}_{channel}.png", case.name));
            assert_matches_golden(&pixels, &golden_path, channel);
        }
    }
}
