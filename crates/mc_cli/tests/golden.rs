//! Golden-image regression tests (CLAUDE.md §10's M6 item): loads an example
//! graph from disk, evaluates it exactly the way `mc_cli` does, and compares
//! its Base Color output against a checked-in reference PNG in
//! `tests/golden/`.
//!
//! These guard against an *unintentional* visual regression in an example
//! graph — a node's math changing in a way nobody meant to change how
//! `bricks.mcgraph` looks, say. They are **not** a substitute for a node's
//! own tiling/determinism/behavior tests (CLAUDE.md §6.7): those catch a
//! specific node breaking; these catch a whole graph's rendered look
//! drifting, from any cause.
//!
//! A small per-channel tolerance (not bit-exact equality) is deliberate:
//! CLAUDE.md §6.3's determinism guarantee is same-GPU only ("same graph +
//! same seed = bit-identical output on the *same* GPU"), so a reference
//! image checked in from one machine could differ by a texel or two of
//! floating-point noise on another GPU/driver without any real regression
//! having happened.
//!
//! To add or intentionally update a reference image, run:
//! `MC_UPDATE_GOLDEN=1 cargo test -p mc_cli --test golden`

use std::path::Path;

/// Max allowed per-channel 8-bit difference at any single pixel. Loose
/// enough to absorb cross-GPU floating-point noise, tight enough to still
/// catch a real visual change (an actual different color, a shifted
/// pattern, a missing node's worth of effect).
const TOLERANCE: u8 = 6;

fn render_base_color(graph_path: &str) -> (Vec<u8>, u32) {
    let mut graph = mc_core::Graph::load_from_file(graph_path)
        .unwrap_or_else(|err| panic!("failed to load {graph_path}: {err}"));

    let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
    let mut pool = mc_gpu::TexturePool::default();
    mc_nodes::registry::evaluate(&mut gpu, &mut pool, &mut graph)
        .unwrap_or_else(|err| panic!("failed to evaluate {graph_path}: {err}"));

    let resolution = graph.settings.resolution;
    let node_id = graph
        .nodes()
        .find(|(_, node)| node.kind == "output_base_color")
        .map(|(id, _)| id)
        .unwrap_or_else(|| panic!("{graph_path} has no output_base_color node"));
    let key = mc_nodes::registry::resolve_texture_key(&graph, node_id)
        .unwrap_or_else(|| panic!("{graph_path}'s Output: Base Color input isn't connected"));
    let texture = pool
        .texture(&key)
        .expect("evaluated output texture missing from the pool");

    (gpu.read_rgba8(texture, resolution), resolution)
}

/// Compares `pixels` against `golden_path`, or (with `MC_UPDATE_GOLDEN=1` set)
/// writes `pixels` there instead of comparing.
fn assert_matches_golden(pixels: &[u8], resolution: u32, golden_path: &str) {
    if std::env::var_os("MC_UPDATE_GOLDEN").is_some() {
        let image = image::RgbaImage::from_raw(resolution, resolution, pixels.to_vec())
            .expect("pixel buffer size matches the image dimensions");
        image
            .save(golden_path)
            .unwrap_or_else(|err| panic!("failed to write {golden_path}: {err}"));
        return;
    }

    assert!(
        Path::new(golden_path).exists(),
        "missing golden image {golden_path} — run `MC_UPDATE_GOLDEN=1 cargo test -p mc_cli \
         --test golden` to create it"
    );
    let golden = image::open(golden_path)
        .unwrap_or_else(|err| panic!("failed to read {golden_path}: {err}"))
        .to_rgba8();
    assert_eq!(
        golden.dimensions(),
        (resolution, resolution),
        "{golden_path} is a different resolution than the graph currently renders at"
    );

    let golden_pixels = golden.into_raw();
    let max_diff = pixels
        .iter()
        .zip(golden_pixels.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= TOLERANCE,
        "{golden_path}: rendered output differs from the golden image by up to {max_diff} \
         (tolerance {TOLERANCE}) — if this is an intentional change to the graph, re-run with \
         MC_UPDATE_GOLDEN=1 to update the reference"
    );
}

#[test]
fn bricks_matches_golden() {
    let (pixels, resolution) = render_base_color("../../examples/graphs/bricks.mcgraph");
    assert_matches_golden(
        &pixels,
        resolution,
        "../../tests/golden/bricks_base_color.png",
    );
}

#[test]
fn glitch_dither_demo_matches_golden() {
    let (pixels, resolution) =
        render_base_color("../../examples/graphs/glitch_dither_demo.mcgraph");
    assert_matches_golden(
        &pixels,
        resolution,
        "../../tests/golden/glitch_dither_demo_base_color.png",
    );
}

#[test]
fn rock_matches_golden() {
    let (pixels, resolution) = render_base_color("../../examples/graphs/rock.mcgraph");
    assert_matches_golden(
        &pixels,
        resolution,
        "../../tests/golden/rock_base_color.png",
    );
}

#[test]
fn brushed_metal_matches_golden() {
    let (pixels, resolution) = render_base_color("../../examples/graphs/brushed_metal.mcgraph");
    assert_matches_golden(
        &pixels,
        resolution,
        "../../tests/golden/brushed_metal_base_color.png",
    );
}

#[test]
fn fabric_matches_golden() {
    let (pixels, resolution) = render_base_color("../../examples/graphs/fabric.mcgraph");
    assert_matches_golden(
        &pixels,
        resolution,
        "../../tests/golden/fabric_base_color.png",
    );
}

#[test]
fn wood_matches_golden() {
    let (pixels, resolution) = render_base_color("../../examples/graphs/wood.mcgraph");
    assert_matches_golden(
        &pixels,
        resolution,
        "../../tests/golden/wood_base_color.png",
    );
}
