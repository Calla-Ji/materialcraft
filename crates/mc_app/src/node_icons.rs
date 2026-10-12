// SPDX-License-Identifier: MIT OR Apache-2.0
//! A Material Icons Round glyph per node kind, for the Node Library panel
//! (same bundled font as the 3D viewport toolbar — see ASSETS.md). These are
//! placeholder choices: picked for a reasonable visual match per node, not
//! bespoke art, and several nodes share a glyph where nothing better fits or
//! the visual distinction isn't important. Codepoints verified against the
//! font's own published codepoints file (fonts.google.com/icons), not
//! guessed.

pub fn icon_for_kind(kind: &str) -> char {
    match kind {
        "perlin_noise" => '\u{e3ea}',             // grain
        "simplex_noise" => '\u{e421}',            // texture
        "white_noise" => '\u{e3ea}',              // grain
        "clouds" => '\u{e2bd}',                   // cloud
        "voronoi" => '\u{e268}',                  // scatter_plot
        "fractal_sum" => '\u{e53b}',              // layers
        "tile_generator" => '\u{e9b0}',           // grid_view
        "fabric_generator" => '\u{f19e}',         // checkroom
        "thread_shape" => '\u{e91a}',             // line_weight
        "image_import" => '\u{e43e}',             // add_photo_alternate
        "shape" => '\u{e574}',                    // category
        "gradient_linear" => '\u{e3e9}',          // gradient
        "gradient_radial" => '\u{e3a2}',          // blur_circular
        "uniform_color" => '\u{e23a}',            // format_color_fill
        "checker" => '\u{e3ec}',                  // grid_on
        "dither_pattern" => '\u{e421}',           // texture
        "levels" => '\u{e429}',                   // tune
        "invert" => '\u{e891}',                   // invert_colors
        "clamp" => '\u{e94d}',                    // compress
        "curve" => '\u{e6e1}',                    // show_chart
        "hsl_adjust" => '\u{e40a}',               // palette
        "gradient_map" => '\u{e3e9}',             // gradient
        "grayscale_conversion" => '\u{e3db}',     // filter_b_and_w
        "blend" => '\u{e53b}',                    // layers
        "histogram_scan" => '\u{e915}',           // compare_arrows
        "threshold" => '\u{e3ca}',                // exposure
        "blur" => '\u{e3a5}',                     // blur_on
        "bevel" => '\u{e920}',                    // rounded_corner
        "warp" => '\u{e176}',                     // waves
        "transform2d" => '\u{e428}',              // transform
        "directional_blur" => '\u{e3a3}',         // blur_linear
        "directional_warp" => '\u{e8e4}',         // trending_flat
        "edge_detect" => '\u{e3c2}',              // crop_free
        "slope_blur" => '\u{efd8}',               // air
        "height_to_normal" => '\u{e564}',         // terrain
        "height_to_ao" => '\u{e427}',             // tonality
        "curvature_from_normal" => '\u{ea3b}',    // architecture
        "channel_split" => '\u{e0b6}',            // call_split
        "block_shift" => '\u{e3ad}',              // broken_image
        "scanline_jitter" => '\u{e919}',          // line_style
        "data_corrupt" => '\u{e868}',             // bug_report
        "ordered_dither" => '\u{e421}',           // texture
        "palette_dither" => '\u{e3b8}',           // colorize
        "output_base_color" => '\u{e243}',        // format_paint
        "output_normal" => '\u{ebbb}',            // polyline
        "output_roughness" => '\u{e3ea}',         // grain
        "output_metallic" => '\u{e838}',          // star
        "output_height" => '\u{e3f7}',            // landscape
        "output_ambient_occlusion" => '\u{e435}', // vignette
        "output_emissive" => '\u{e430}',          // wb_sunny
        "output_alpha" => '\u{e91c}',             // opacity
        _ => '\u{e574}',                          // category (fallback)
    }
}
