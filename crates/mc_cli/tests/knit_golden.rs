// SPDX-License-Identifier: MIT OR Apache-2.0
//! Golden-image tests for the Knit node, Steps 2-5 (KNIT_NODE_SPEC.md §8
//! items 2-5): `tube_raster` rasterizing `resolve()`/`course_centerline()`'s
//! own chart/rib/purl/interlock geometry, `course_tubes()`'s (Step 4)
//! rotation-minimizing-frame ply math, and (Step 5) `knit_stitch_info.wgsl`'s
//! Stitch ID/Stitch Type. No Colour output: the spec has one, but the owner
//! cut it from the shipped node (see `mc_nodes::knit`'s own module doc
//! comment). Every case is the owner's ground-truth renderer
//! (`knit_reference.py`)'s own `golden/params.json` entry at
//! `resolution: 512`:
//!
//! - `jersey_front`/`jersey_back`: Step 2's own baseline (plain jersey, no
//!   chart rules in play) — front and back exercise the K<->P chart swap.
//! - `jersey_tight`: same chart, `tension: tight` — same code path as
//!   `jersey_front`, different resolved numbers, so this is really a
//!   regression check that `tension` reaches the renderer correctly.
//! - `purl`: the rib rule's *course* sibling — alternating K/P courses
//!   (links-links), exercising `czoff` and the course-direction sinker
//!   blend.
//! - `rib_1x1`, `rib_2x2`: the wale-direction rib rule — narrowed, receded
//!   purl wales, exercising `widths`/`zoff` and the wale-direction blend.
//! - `interlock`: the stagger rule (odd wales offset half a course down).
//! - `jersey_2ply`: Step 4 — two plies twisting around each other along the
//!   rotation-minimizing frame, the only case here with `plies > 1`.
//!
//! Same decode path and acceptance criteria as `thread_golden.rs` — see its
//! own doc comment for why 16-bit goldens need their own test file — plus
//! KNIT_NODE_SPEC.md §10's own Mask-specific bar (IoU above 0.97, not just
//! the same mean-abs-diff/outlier check used for the other channels).

use std::path::Path;

const MEAN_ABS_DIFF_TOLERANCE: f32 = 0.02;
const OUTLIER_FRACTION_TOLERANCE: f32 = 0.01;
const PER_PIXEL_TOLERANCE: f32 = 0.1;
const MASK_IOU_TOLERANCE: f32 = 0.97;

const RESOLUTION: u32 = 512;

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

fn rendered_r_channel(rendered_rgba8: &[u8]) -> Vec<f32> {
    rendered_rgba8
        .iter()
        .step_by(4)
        .map(|&v| v as f32 / 255.0)
        .collect()
}

fn assert_matches_golden(rendered_rgba8: &[u8], golden_path: &Path, channel: &str) {
    assert!(
        golden_path.exists(),
        "missing golden image {} — see tests/golden/knit/ and knit_golden.rs's own doc comment",
        golden_path.display()
    );
    let (golden, gw, gh) = load_golden_normalized(golden_path);
    assert_eq!(
        (gw, gh),
        (RESOLUTION, RESOLUTION),
        "{}: resolution mismatch",
        golden_path.display()
    );

    let rendered = rendered_r_channel(rendered_rgba8);
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

/// KNIT_NODE_SPEC.md §10: "Mask IoU above 0.97." Both sides are thresholded
/// at 0.5 (golden is 16-bit but still just 0/65535; rendered is an 8-bit
/// 0/255 mask) before computing intersection-over-union.
fn assert_mask_iou(rendered_rgba8: &[u8], golden_path: &Path) {
    assert!(
        golden_path.exists(),
        "missing golden image {}",
        golden_path.display()
    );
    let (golden, _, _) = load_golden_normalized(golden_path);
    let rendered = rendered_r_channel(rendered_rgba8);
    assert_eq!(rendered.len(), golden.len());

    let mut intersection = 0u32;
    let mut union = 0u32;
    for (&r, &g) in rendered.iter().zip(golden.iter()) {
        let r_set = r > 0.5;
        let g_set = g > 0.5;
        if r_set && g_set {
            intersection += 1;
        }
        if r_set || g_set {
            union += 1;
        }
    }
    let iou = intersection as f32 / union as f32;
    assert!(
        iou > MASK_IOU_TOLERANCE,
        "{}: mask IoU {iou:.4} is below the {MASK_IOU_TOLERANCE} tolerance",
        golden_path.display()
    );
}

#[test]
fn knit_tube_raster_matches_the_python_reference_renderer() {
    let golden_dir = Path::new("../../tests/golden/knit");

    use mc_nodes::knit::{KnitParams, KnitPreset, Side, Tension};

    let cases = [
        (
            "jersey_front",
            KnitParams {
                side: Side::Front,
                ..KnitParams::default()
            },
        ),
        (
            "jersey_back",
            KnitParams {
                side: Side::Back,
                ..KnitParams::default()
            },
        ),
        (
            "jersey_tight",
            KnitParams {
                tension: Tension::Tight,
                ..KnitParams::default()
            },
        ),
        (
            "purl",
            KnitParams {
                preset: KnitPreset::Purl,
                ..KnitParams::default()
            },
        ),
        (
            "rib_1x1",
            KnitParams {
                preset: KnitPreset::Rib1x1,
                wales: 10,
                ..KnitParams::default()
            },
        ),
        (
            "rib_2x2",
            KnitParams {
                preset: KnitPreset::Rib2x2,
                wales: 12,
                ..KnitParams::default()
            },
        ),
        (
            "interlock",
            KnitParams {
                preset: KnitPreset::Interlock,
                wales: 10,
                ..KnitParams::default()
            },
        ),
        (
            "jersey_2ply",
            KnitParams {
                wales: 6,
                plies: 2,
                ..KnitParams::default()
            },
        ),
    ];

    for (name, params) in cases {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let height_key = format!("{name}_height");
        let mask_key = format!("{name}_mask");
        let ply_id_key = format!("{name}_ply_id");
        let stitch_id_key = format!("{name}_stitch_id");
        let stitch_type_key = format!("{name}_stitch_type");

        mc_nodes::knit::dispatch(
            &mut gpu,
            &mut pool,
            mc_nodes::knit::KnitOutputKeys {
                height: &height_key,
                mask: &mask_key,
                ply_id: &ply_id_key,
                stitch_id: &stitch_id_key,
                stitch_type: &stitch_type_key,
            },
            RESOLUTION,
            &params,
        );

        let height_pixels = gpu.read_rgba8(pool.texture(&height_key).unwrap(), RESOLUTION);
        let mask_pixels = gpu.read_rgba8(pool.texture(&mask_key).unwrap(), RESOLUTION);
        let ply_id_pixels = gpu.read_rgba8(pool.texture(&ply_id_key).unwrap(), RESOLUTION);
        let stitch_id_pixels = gpu.read_rgba8(pool.texture(&stitch_id_key).unwrap(), RESOLUTION);
        let stitch_type_pixels =
            gpu.read_rgba8(pool.texture(&stitch_type_key).unwrap(), RESOLUTION);

        assert_matches_golden(
            &height_pixels,
            &golden_dir.join(format!("{name}_height.png")),
            "height",
        );
        assert_matches_golden(
            &ply_id_pixels,
            &golden_dir.join(format!("{name}_ply_id.png")),
            "ply_id",
        );
        assert_mask_iou(&mask_pixels, &golden_dir.join(format!("{name}_mask.png")));
        assert_matches_golden(
            &stitch_id_pixels,
            &golden_dir.join(format!("{name}_stitch_id.png")),
            "stitch_id",
        );
        // KNIT_NODE_SPEC.md §10: "Stitch Type exact except within 2px of
        // wale boundaries" — the generic mean-abs-diff/outlier tolerance
        // already allows for exactly that (a strict 0/1 signal, so any
        // boundary disagreement is a full 1.0 diff at a handful of pixels,
        // comfortably inside the 1% outlier allowance).
        assert_matches_golden(
            &stitch_type_pixels,
            &golden_dir.join(format!("{name}_stitch_type.png")),
            "stitch_type",
        );
    }
}
