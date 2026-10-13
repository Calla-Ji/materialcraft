// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bridges mc_core's generic graph model to actual GPU node dispatch.
//!
//! Given a [`mc_core::Graph`], [`evaluate`] walks it in topological order,
//! skips nodes the graph says are already up to date, and dispatches each
//! remaining node's GPU work by its `kind`. Each evaluated node's texture
//! lives in the [`mc_gpu::TexturePool`] under the key from [`texture_key`].

use std::collections::HashSet;

use crate::{
    bevel, blend, block_shift, blur, channel_split, checker, clamp, clouds, curvature_from_normal,
    curve, data_corrupt, directional_blur, directional_warp, dither_pattern, edge_detect,
    fabric_generator, fractal_sum, gradient_linear, gradient_map, gradient_radial,
    grayscale_conversion, height_to_ao, height_to_normal, histogram_scan, hsl_adjust, image_import,
    invert, levels, ordered_dither, palette_dither, perlin, scanline_jitter, shape, simplex_noise,
    slope_blur, thread_shape, threshold, tile_generator, transform2d, uniform_color, voronoi, warp,
    white_noise,
};

/// The default output port name every single-output `NodeDef` uses.
pub const DEFAULT_OUTPUT_PORT: &str = "output";

/// The texture pool key a node's output `port` is stored under. Every
/// single-output node (the overwhelming majority) always passes
/// [`DEFAULT_OUTPUT_PORT`] here; a multi-output node (currently just
/// Thread) stores each of its named outputs under its own key.
pub fn texture_key(id: mc_core::NodeId, port: &str) -> String {
    format!("node_{}_{}", id.0, port)
}

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error(transparent)]
    Graph(#[from] mc_core::GraphError),
    #[error("node {0:?} has kind {1:?}, which mc_nodes doesn't know how to evaluate")]
    UnknownKind(mc_core::NodeId, String),
    #[error("node {0:?}'s required input {1:?} isn't connected")]
    MissingInput(mc_core::NodeId, String),
}

/// Evaluate every node in `graph` that needs it, in dependency order.
///
/// Output nodes (`kind` starting with `"output_"`) are pass-through markers
/// with no shader of their own — see [`resolve_texture_key`] to find what
/// they actually point at.
///
/// A node with an unconnected required input (completely normal mid-edit —
/// e.g. a just-added filter node nobody's wired up yet) does *not* abort the
/// whole pass: it, and anything downstream of it, are skipped and left
/// dirty for next time, but every other, independent part of the graph
/// still evaluates and gets its texture this frame. Without that, one loose
/// node anywhere in the graph used to silently freeze every node after it
/// in topological order — including already-fully-wired, already-working
/// ones — since the first `?` on a missing input aborted the entire
/// function. The first error encountered is still returned (after the full
/// pass completes) so callers like `mc_cli`, which genuinely want a hard
/// failure for a broken graph file, still get one.
pub fn evaluate(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    graph: &mut mc_core::Graph,
) -> Result<(), EvalError> {
    let order = graph.topological_order()?;
    let resolution = graph.settings.resolution;

    let mut failed: HashSet<mc_core::NodeId> = HashSet::new();
    let mut first_error: Option<EvalError> = None;

    for id in order {
        if !graph.needs_evaluation(id) {
            continue;
        }

        let node = graph
            .node(id)
            .expect("node from topological_order always exists");
        if node.kind.starts_with("output_") {
            graph.mark_evaluated(id);
            continue;
        }

        // A required input: missing entirely, or wired to a node that
        // itself failed to evaluate this pass, are the same "can't run this
        // node right now" condition — either way there's no real texture to
        // dispatch with. Returns the texture key for the *specific upstream
        // output port* this input is wired to, not just "the" upstream
        // node's texture — needed now that a node (Thread) can have more
        // than one named output.
        let resolve_input = |port: &str| -> Result<String, EvalError> {
            let (input_id, from_port) = graph
                .upstream_from(id, port)
                .ok_or_else(|| EvalError::MissingInput(id, port.to_string()))?;
            if failed.contains(&input_id) {
                return Err(EvalError::MissingInput(id, port.to_string()));
            }
            Ok(texture_key(input_id, &from_port))
        };
        // An optional input (e.g. Blend's mask): absent is a supported
        // state on its own, so a failed upstream degrades to "absent"
        // rather than failing this node too.
        let resolve_optional_input = |port: &str| -> Option<String> {
            let (input_id, from_port) = graph.upstream_from(id, port)?;
            if failed.contains(&input_id) {
                return None;
            }
            Some(texture_key(input_id, &from_port))
        };

        let key = texture_key(id, DEFAULT_OUTPUT_PORT);
        let result: Result<(), EvalError> = (|| {
            match node.kind.as_str() {
                "perlin_noise" => {
                    let params = perlin::PerlinParams::from_graph(&node.params);
                    perlin::dispatch(gpu, pool, &key, resolution, &params);
                }
                "levels" => {
                    let input_key = resolve_input("input")?;
                    let params = levels::LevelsParams::from_graph(&node.params);
                    levels::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "blend" => {
                    let base_key = resolve_input("base")?;
                    let blend_key = resolve_input("blend")?;
                    let mask_key = resolve_optional_input("mask");
                    let params = blend::BlendParams::from_graph(&node.params);
                    blend::dispatch(
                        gpu,
                        pool,
                        &base_key,
                        &blend_key,
                        mask_key.as_deref(),
                        &key,
                        resolution,
                        &params,
                    );
                }
                "voronoi" => {
                    let params = voronoi::VoronoiParams::from_graph(&node.params);
                    voronoi::dispatch(gpu, pool, &key, resolution, &params);
                }
                "fractal_sum" => {
                    let params = fractal_sum::FractalSumParams::from_graph(&node.params);
                    fractal_sum::dispatch(gpu, pool, &key, resolution, &params);
                }
                "tile_generator" => {
                    let shape_key = resolve_optional_input("shape");
                    let params = tile_generator::TileGeneratorParams::from_graph(&node.params);
                    tile_generator::dispatch(
                        gpu,
                        pool,
                        shape_key.as_deref(),
                        &key,
                        resolution,
                        &params,
                    );
                }
                "fabric_generator" => {
                    let fiber_key = resolve_optional_input("fiber");
                    let params = fabric_generator::FabricGeneratorParams::from_graph(&node.params);
                    fabric_generator::dispatch(
                        gpu,
                        pool,
                        fiber_key.as_deref(),
                        &key,
                        resolution,
                        &params,
                    );
                }
                "thread_shape" => {
                    let height_key = texture_key(id, "height");
                    let ply_id_key = texture_key(id, "ply_id");
                    let fiber_key = texture_key(id, "fiber");
                    let mask_key = texture_key(id, "mask");
                    let params = thread_shape::ThreadShapeParams::from_graph(&node.params);
                    thread_shape::dispatch(
                        gpu,
                        pool,
                        thread_shape::ThreadOutputKeys {
                            height: &height_key,
                            ply_id: &ply_id_key,
                            fiber: &fiber_key,
                            mask: &mask_key,
                        },
                        resolution,
                        &params,
                    );
                }
                "shape" => {
                    let params = shape::ShapeParams::from_graph(&node.params);
                    shape::dispatch(gpu, pool, &key, resolution, &params);
                }
                "gradient_linear" => {
                    let params = gradient_linear::GradientLinearParams::from_graph(&node.params);
                    gradient_linear::dispatch(gpu, pool, &key, resolution, &params);
                }
                "gradient_radial" => {
                    let params = gradient_radial::GradientRadialParams::from_graph(&node.params);
                    gradient_radial::dispatch(gpu, pool, &key, resolution, &params);
                }
                "uniform_color" => {
                    let params = uniform_color::UniformColorParams::from_graph(&node.params);
                    uniform_color::dispatch(gpu, pool, &key, resolution, &params);
                }
                "image_import" => {
                    let params = image_import::ImageImportParams::from_graph(&node.params);
                    image_import::dispatch(gpu, pool, &key, resolution, &params);
                }
                "invert" => {
                    let input_key = resolve_input("input")?;
                    invert::dispatch(gpu, pool, &input_key, &key, resolution);
                }
                "histogram_scan" => {
                    let input_key = resolve_input("input")?;
                    let params = histogram_scan::HistogramScanParams::from_graph(&node.params);
                    histogram_scan::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "blur" => {
                    let input_key = resolve_input("input")?;
                    let params = blur::BlurParams::from_graph(&node.params);
                    blur::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "bevel" => {
                    let input_key = resolve_input("input")?;
                    let params = bevel::BevelParams::from_graph(&node.params);
                    bevel::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "warp" => {
                    let input_key = resolve_input("input")?;
                    let intensity_key = resolve_input("intensity")?;
                    let params = warp::WarpParams::from_graph(&node.params);
                    warp::dispatch(
                        gpu,
                        pool,
                        &input_key,
                        &intensity_key,
                        &key,
                        resolution,
                        &params,
                    );
                }
                "transform2d" => {
                    let input_key = resolve_input("input")?;
                    let params = transform2d::Transform2DParams::from_graph(&node.params);
                    transform2d::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "height_to_normal" => {
                    let input_key = resolve_input("input")?;
                    let params = height_to_normal::HeightToNormalParams::from_graph(&node.params);
                    height_to_normal::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "clamp" => {
                    let input_key = resolve_input("input")?;
                    let params = clamp::ClampParams::from_graph(&node.params);
                    clamp::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "threshold" => {
                    let input_key = resolve_input("input")?;
                    let params = threshold::ThresholdParams::from_graph(&node.params);
                    threshold::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "grayscale_conversion" => {
                    let input_key = resolve_input("input")?;
                    grayscale_conversion::dispatch(gpu, pool, &input_key, &key, resolution);
                }
                "checker" => {
                    let params = checker::CheckerParams::from_graph(&node.params);
                    checker::dispatch(gpu, pool, &key, resolution, &params);
                }
                "dither_pattern" => {
                    let params = dither_pattern::DitherPatternParams::from_graph(&node.params);
                    dither_pattern::dispatch(gpu, pool, &key, resolution, &params);
                }
                "ordered_dither" => {
                    let image_key = resolve_input("image")?;
                    let params = ordered_dither::OrderedDitherParams::from_graph(&node.params);
                    ordered_dither::dispatch(gpu, pool, &image_key, &key, resolution, &params);
                }
                "palette_dither" => {
                    let image_key = resolve_input("image")?;
                    let params = palette_dither::PaletteDitherParams::from_graph(&node.params);
                    palette_dither::dispatch(gpu, pool, &image_key, &key, resolution, &params);
                }
                "channel_split" => {
                    let image_key = resolve_input("image")?;
                    let mask_key = resolve_optional_input("mask");
                    let params = channel_split::ChannelSplitParams::from_graph(&node.params);
                    channel_split::dispatch(
                        gpu,
                        pool,
                        &image_key,
                        mask_key.as_deref(),
                        &key,
                        resolution,
                        &params,
                    );
                }
                "block_shift" => {
                    let image_key = resolve_input("image")?;
                    let mask_key = resolve_optional_input("mask");
                    let params = block_shift::BlockShiftParams::from_graph(&node.params);
                    block_shift::dispatch(
                        gpu,
                        pool,
                        &image_key,
                        mask_key.as_deref(),
                        &key,
                        resolution,
                        &params,
                    );
                }
                "data_corrupt" => {
                    let image_key = resolve_input("image")?;
                    let mask_key = resolve_optional_input("mask");
                    let params = data_corrupt::DataCorruptParams::from_graph(&node.params);
                    data_corrupt::dispatch(
                        gpu,
                        pool,
                        &image_key,
                        mask_key.as_deref(),
                        &key,
                        resolution,
                        &params,
                    );
                }
                "scanline_jitter" => {
                    let image_key = resolve_input("image")?;
                    let mask_key = resolve_optional_input("mask");
                    let params = scanline_jitter::ScanlineJitterParams::from_graph(&node.params);
                    scanline_jitter::dispatch(
                        gpu,
                        pool,
                        &image_key,
                        mask_key.as_deref(),
                        &key,
                        resolution,
                        &params,
                    );
                }
                "hsl_adjust" => {
                    let input_key = resolve_input("input")?;
                    let params = hsl_adjust::HslAdjustParams::from_graph(&node.params);
                    hsl_adjust::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "curve" => {
                    let input_key = resolve_input("input")?;
                    let params = curve::CurveParams::from_graph(&node.params);
                    curve::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "gradient_map" => {
                    let input_key = resolve_input("input")?;
                    let params = gradient_map::GradientMapParams::from_graph(&node.params);
                    gradient_map::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "simplex_noise" => {
                    let params = simplex_noise::SimplexParams::from_graph(&node.params);
                    simplex_noise::dispatch(gpu, pool, &key, resolution, &params);
                }
                "white_noise" => {
                    let params = white_noise::WhiteNoiseParams::from_graph(&node.params);
                    white_noise::dispatch(gpu, pool, &key, resolution, &params);
                }
                "clouds" => {
                    let params = clouds::CloudsParams::from_graph(&node.params);
                    clouds::dispatch(gpu, pool, &key, resolution, &params);
                }
                "directional_blur" => {
                    let input_key = resolve_input("input")?;
                    let params = directional_blur::DirectionalBlurParams::from_graph(&node.params);
                    directional_blur::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "directional_warp" => {
                    let input_key = resolve_input("input")?;
                    let intensity_key = resolve_input("intensity")?;
                    let params = directional_warp::DirectionalWarpParams::from_graph(&node.params);
                    directional_warp::dispatch(
                        gpu,
                        pool,
                        &input_key,
                        &intensity_key,
                        &key,
                        resolution,
                        &params,
                    );
                }
                "edge_detect" => {
                    let input_key = resolve_input("input")?;
                    let params = edge_detect::EdgeDetectParams::from_graph(&node.params);
                    edge_detect::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "slope_blur" => {
                    let input_key = resolve_input("input")?;
                    let control_key = resolve_input("control")?;
                    let params = slope_blur::SlopeBlurParams::from_graph(&node.params);
                    slope_blur::dispatch(
                        gpu,
                        pool,
                        &input_key,
                        &control_key,
                        &key,
                        resolution,
                        &params,
                    );
                }
                "height_to_ao" => {
                    let input_key = resolve_input("input")?;
                    let params = height_to_ao::HeightToAoParams::from_graph(&node.params);
                    height_to_ao::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "curvature_from_normal" => {
                    let input_key = resolve_input("input")?;
                    let params =
                        curvature_from_normal::CurvatureFromNormalParams::from_graph(&node.params);
                    curvature_from_normal::dispatch(
                        gpu, pool, &input_key, &key, resolution, &params,
                    );
                }
                other => return Err(EvalError::UnknownKind(id, other.to_string())),
            }
            Ok(())
        })();

        match result {
            Ok(()) => graph.mark_evaluated(id),
            Err(err) => {
                failed.insert(id);
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
    }

    match first_error {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// The texture pool key actually holding `node`'s visible output — the one
/// shown in its thumbnail, the 2D View, and the 3D viewport. For most nodes
/// that's just its own (single) output; an Output node is a pass-through
/// marker, so its "output" is whatever feeds its `input` port; a
/// multi-output node (currently just Thread) shows its *first* declared
/// output (Height, for Thread — deliberately kept first in its `NodeDef` so
/// this falls out for free rather than needing a port picker).
pub fn resolve_texture_key(graph: &mc_core::Graph, node: mc_core::NodeId) -> Option<String> {
    let kind = &graph.node(node)?.kind;
    if kind.starts_with("output_") {
        // Which *port* fed this Output node's input matters now that a
        // node can have more than one (Thread) — `upstream` alone would
        // discard that and fall through to whatever the upstream node's
        // own default/first output is, showing Height no matter which of
        // Thread's four outputs was actually wired in (confirmed bug: the
        // owner wired Ply ID, Fiber, and Mask into three separate Output
        // nodes and all three showed Height). An Output node is never
        // itself a multi-output node, so if the *upstream* one is, follow
        // its specific port directly instead of recursing into the
        // single-output branch below, which would re-derive the wrong
        // (first/default) port all over again.
        let (upstream, upstream_port) = graph.upstream_from(node, "input")?;
        let upstream_kind = &graph.node(upstream)?.kind;
        if upstream_kind.starts_with("output_") {
            resolve_texture_key(graph, upstream)
        } else {
            Some(texture_key(upstream, &upstream_port))
        }
    } else {
        let port = crate::defs::node_def(kind)
            .and_then(|def| def.outputs.first())
            .map(|p| p.name)
            .unwrap_or(DEFAULT_OUTPUT_PORT);
        Some(texture_key(node, port))
    }
}

#[cfg(test)]
mod tests {
    use super::{evaluate, resolve_texture_key, texture_key};

    /// Regression test for a real bug the owner hit: wiring Thread's Ply
    /// ID, Fiber, and Mask outputs each into their own Output node, and
    /// all three showing Height instead. Root cause: `resolve_texture_key`
    /// recursed into a multi-output upstream node without carrying along
    /// *which* of its outputs the connection actually used, silently
    /// falling back to that node's first/default output every time.
    #[test]
    fn an_output_node_shows_the_specific_upstream_port_it_is_wired_to() {
        let mut graph = mc_core::Graph::new();
        let thread = graph.add_node(mc_core::Node::new("thread_shape"));
        let height_out = graph.add_node(mc_core::Node::new("output_height"));
        let ply_id_out = graph.add_node(mc_core::Node::new("output_height"));
        let mask_out = graph.add_node(mc_core::Node::new("output_height"));
        graph
            .connect(thread, "height", height_out, "input")
            .unwrap();
        graph
            .connect(thread, "ply_id", ply_id_out, "input")
            .unwrap();
        graph.connect(thread, "mask", mask_out, "input").unwrap();
        graph.settings.resolution = 32;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let height_key = resolve_texture_key(&graph, height_out).unwrap();
        let ply_id_key = resolve_texture_key(&graph, ply_id_out).unwrap();
        let mask_key = resolve_texture_key(&graph, mask_out).unwrap();

        assert_eq!(height_key, texture_key(thread, "height"));
        assert_eq!(ply_id_key, texture_key(thread, "ply_id"));
        assert_eq!(mask_key, texture_key(thread, "mask"));

        let height_pixels = gpu.read_rgba8(pool.texture(&height_key).unwrap(), 32);
        let ply_id_pixels = gpu.read_rgba8(pool.texture(&ply_id_key).unwrap(), 32);
        let mask_pixels = gpu.read_rgba8(pool.texture(&mask_key).unwrap(), 32);
        assert_ne!(
            height_pixels, ply_id_pixels,
            "an Output node fed by Ply ID should not show Height's pixels"
        );
        assert_ne!(
            height_pixels, mask_pixels,
            "an Output node fed by Mask should not show Height's pixels"
        );
    }

    /// CLAUDE.md's M2 acceptance check: a 3-node graph (Noise → Levels →
    /// Output) evaluates headlessly.
    #[test]
    fn evaluates_noise_levels_output_graph() {
        let mut graph = mc_core::Graph::new();
        let noise = graph.add_node(mc_core::Node::new("perlin_noise"));
        let levels = graph.add_node(mc_core::Node::new("levels"));
        let output = graph.add_node(mc_core::Node::new("output_base_color"));
        graph.connect(noise, "output", levels, "input").unwrap();
        graph.connect(levels, "output", output, "input").unwrap();
        graph.settings.resolution = 64;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let key = resolve_texture_key(&graph, output).expect("output resolves to a texture");
        let pixels = gpu.read_rgba8(pool.texture(&key).expect("texture exists"), 64);
        assert_eq!(pixels.len(), 64 * 64 * 4);

        for id in [noise, levels, output] {
            assert!(
                !graph.needs_evaluation(id),
                "node {id:?} should be marked up to date after evaluate()"
            );
        }
    }

    /// Regression test for the Pattern Generator/Value Range/Flow Blur/Push
    /// Warp rename (`display_name` only, `kind` strings unchanged) — proves
    /// the registry's `"slope_blur"` dispatch arm (now showing as "Flow
    /// Blur" in the UI) still resolves and runs correctly end to end, not
    /// just the node's own isolated dispatch test. Both its inputs ("input"
    /// and "control") are required — the Output node is deliberately *not*
    /// fed through a single-input shortcut here, since the real-world
    /// report this guards against ("Flow Blur isn't visualizing") was most
    /// likely a missing second ("control") connection, not a dispatch bug.
    #[test]
    fn flow_blur_evaluates_through_the_registry_after_its_rename() {
        let mut graph = mc_core::Graph::new();
        let input_noise = graph.add_node(
            mc_core::Node::new("perlin_noise").with_param("seed", mc_core::ParamValue::Seed(1)),
        );
        let control_noise = graph.add_node(
            mc_core::Node::new("perlin_noise").with_param("seed", mc_core::ParamValue::Seed(2)),
        );
        let flow_blur = graph.add_node(mc_core::Node::new("slope_blur"));
        let output = graph.add_node(mc_core::Node::new("output_base_color"));

        graph
            .connect(input_noise, "output", flow_blur, "input")
            .unwrap();
        graph
            .connect(control_noise, "output", flow_blur, "control")
            .unwrap();
        graph.connect(flow_blur, "output", output, "input").unwrap();
        graph.settings.resolution = 64;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates with both inputs wired");

        let key = resolve_texture_key(&graph, output).expect("output resolves to a texture");
        let pixels = gpu.read_rgba8(pool.texture(&key).expect("texture exists"), 64);
        assert!(
            pixels.iter().any(|&b| b != pixels[0]),
            "Flow Blur's output is perfectly flat — looks like it isn't actually running"
        );
    }

    /// CLAUDE.md's M3 acceptance demo: a 4-node graph (Noise → Levels →
    /// Blend → Output) evaluates headlessly. Blend's second input comes from
    /// a second Perlin node so Blend has two genuinely different layers to
    /// combine.
    #[test]
    fn evaluates_noise_levels_blend_output_graph() {
        let mut graph = mc_core::Graph::new();
        let noise_a = graph.add_node(
            mc_core::Node::new("perlin_noise").with_param("seed", mc_core::ParamValue::Seed(1)),
        );
        let noise_b = graph.add_node(
            mc_core::Node::new("perlin_noise").with_param("seed", mc_core::ParamValue::Seed(2)),
        );
        let levels = graph.add_node(mc_core::Node::new("levels"));
        let blend = graph.add_node(mc_core::Node::new("blend"));
        let output = graph.add_node(mc_core::Node::new("output_base_color"));

        graph.connect(noise_a, "output", levels, "input").unwrap();
        graph.connect(levels, "output", blend, "base").unwrap();
        graph.connect(noise_b, "output", blend, "blend").unwrap();
        graph.connect(blend, "output", output, "input").unwrap();
        graph.settings.resolution = 64;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let key = resolve_texture_key(&graph, output).expect("output resolves to a texture");
        let pixels = gpu.read_rgba8(pool.texture(&key).expect("texture exists"), 64);
        assert_eq!(pixels.len(), 64 * 64 * 4);

        for id in [noise_a, noise_b, levels, blend, output] {
            assert!(
                !graph.needs_evaluation(id),
                "node {id:?} should be marked up to date after evaluate()"
            );
        }
    }

    #[test]
    fn re_evaluating_an_unchanged_graph_is_a_cache_hit() {
        let mut graph = mc_core::Graph::new();
        let noise = graph.add_node(mc_core::Node::new("perlin_noise"));
        graph.settings.resolution = 32;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).unwrap();
        assert!(!graph.needs_evaluation(noise));

        // Nothing changed, so a second evaluate() pass should be a no-op —
        // verified indirectly: it must not error and the node stays clean.
        evaluate(&mut gpu, &mut pool, &mut graph).unwrap();
        assert!(!graph.needs_evaluation(noise));
    }

    /// Regression test: a single unconnected filter node anywhere in the
    /// graph (completely normal mid-edit state — e.g. a just-added node
    /// nobody's wired up yet) used to abort the *entire* evaluation pass at
    /// the first `?`, silently leaving every other node after it in
    /// topological order un-evaluated too, including fully-wired,
    /// independent ones. A standalone generator with no inputs at all
    /// (Voronoi here) must still get its texture even when an unrelated,
    /// unconnected Blur node sorts before it.
    #[test]
    fn one_unconnected_node_does_not_block_the_rest_of_the_graph() {
        let mut graph = mc_core::Graph::new();
        let orphan_blur = graph.add_node(mc_core::Node::new("blur"));
        let voronoi = graph.add_node(mc_core::Node::new("voronoi"));
        graph.settings.resolution = 32;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let result = evaluate(&mut gpu, &mut pool, &mut graph);

        assert!(
            result.is_err(),
            "the orphan Blur's missing input should still be reported"
        );
        let key = resolve_texture_key(&graph, voronoi).expect("voronoi resolves to a texture");
        assert!(
            pool.texture(&key).is_some(),
            "an independent, fully-standalone node must still get evaluated even though \
             an unrelated node elsewhere in the graph is missing a required input"
        );
        assert!(
            graph.needs_evaluation(orphan_blur),
            "the actually-broken node should stay dirty, not get marked evaluated"
        );
        assert!(
            !graph.needs_evaluation(voronoi),
            "the independent node should be marked up to date"
        );
    }
}
