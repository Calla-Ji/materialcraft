//! Shader source loading, with hot-reload from disk in debug builds.
//!
//! WGSL has no `#include`, so a node's full shader is assembled by hand: the
//! shared helpers in `common.wgsl` are prepended to the node-specific source
//! before it's handed to `mc_gpu` for compilation.

use std::borrow::Cow;

/// The full WGSL source for a node: `common.wgsl` followed by `file_name`.
pub fn full_source(file_name: &'static str) -> String {
    format!("{}\n{}", node_source("common.wgsl"), node_source(file_name))
}

/// The WGSL source for a single shader file.
///
/// In debug builds this is read from disk on every call, so editing a
/// `.wgsl` file and re-evaluating the node picks up the change without
/// restarting the app. Release builds use the copy embedded at compile time.
fn node_source(file_name: &'static str) -> Cow<'static, str> {
    #[cfg(debug_assertions)]
    {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("shaders")
            .join(file_name);
        match std::fs::read_to_string(&path) {
            Ok(source) => return Cow::Owned(source),
            Err(err) => {
                log::warn!(
                    "could not hot-reload shader {}: {err}; using the embedded copy",
                    path.display()
                );
            }
        }
    }
    Cow::Borrowed(embedded_source(file_name))
}

fn embedded_source(file_name: &str) -> &'static str {
    match file_name {
        "common.wgsl" => include_str!("../shaders/common.wgsl"),
        "perlin_noise.wgsl" => include_str!("../shaders/perlin_noise.wgsl"),
        "levels.wgsl" => include_str!("../shaders/levels.wgsl"),
        "blend.wgsl" => include_str!("../shaders/blend.wgsl"),
        "voronoi.wgsl" => include_str!("../shaders/voronoi.wgsl"),
        "fractal_sum.wgsl" => include_str!("../shaders/fractal_sum.wgsl"),
        "tile_generator.wgsl" => include_str!("../shaders/tile_generator.wgsl"),
        "shape.wgsl" => include_str!("../shaders/shape.wgsl"),
        "gradient_linear.wgsl" => include_str!("../shaders/gradient_linear.wgsl"),
        "gradient_radial.wgsl" => include_str!("../shaders/gradient_radial.wgsl"),
        "uniform_color.wgsl" => include_str!("../shaders/uniform_color.wgsl"),
        "invert.wgsl" => include_str!("../shaders/invert.wgsl"),
        "blur.wgsl" => include_str!("../shaders/blur.wgsl"),
        "warp.wgsl" => include_str!("../shaders/warp.wgsl"),
        "transform2d.wgsl" => include_str!("../shaders/transform2d.wgsl"),
        "histogram_scan.wgsl" => include_str!("../shaders/histogram_scan.wgsl"),
        "height_to_normal.wgsl" => include_str!("../shaders/height_to_normal.wgsl"),
        "clamp.wgsl" => include_str!("../shaders/clamp.wgsl"),
        "threshold.wgsl" => include_str!("../shaders/threshold.wgsl"),
        "grayscale_conversion.wgsl" => include_str!("../shaders/grayscale_conversion.wgsl"),
        "checker.wgsl" => include_str!("../shaders/checker.wgsl"),
        "hsl_adjust.wgsl" => include_str!("../shaders/hsl_adjust.wgsl"),
        "curve.wgsl" => include_str!("../shaders/curve.wgsl"),
        "gradient_map.wgsl" => include_str!("../shaders/gradient_map.wgsl"),
        "simplex_noise.wgsl" => include_str!("../shaders/simplex_noise.wgsl"),
        "white_noise.wgsl" => include_str!("../shaders/white_noise.wgsl"),
        "clouds.wgsl" => include_str!("../shaders/clouds.wgsl"),
        "directional_blur.wgsl" => include_str!("../shaders/directional_blur.wgsl"),
        "directional_warp.wgsl" => include_str!("../shaders/directional_warp.wgsl"),
        "dither_pattern.wgsl" => include_str!("../shaders/dither_pattern.wgsl"),
        "edge_detect.wgsl" => include_str!("../shaders/edge_detect.wgsl"),
        "slope_blur.wgsl" => include_str!("../shaders/slope_blur.wgsl"),
        "height_to_ao.wgsl" => include_str!("../shaders/height_to_ao.wgsl"),
        "curvature_from_normal.wgsl" => include_str!("../shaders/curvature_from_normal.wgsl"),
        "channel_split.wgsl" => include_str!("../shaders/channel_split.wgsl"),
        "block_shift.wgsl" => include_str!("../shaders/block_shift.wgsl"),
        "scanline_jitter.wgsl" => include_str!("../shaders/scanline_jitter.wgsl"),
        "ordered_dither.wgsl" => include_str!("../shaders/ordered_dither.wgsl"),
        "palette_dither.wgsl" => include_str!("../shaders/palette_dither.wgsl"),
        "data_corrupt.wgsl" => include_str!("../shaders/data_corrupt.wgsl"),
        other => panic!("mc_nodes: unknown shader file {other:?}"),
    }
}
