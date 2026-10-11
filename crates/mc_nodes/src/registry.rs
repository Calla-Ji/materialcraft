//! Bridges mc_core's generic graph model to actual GPU node dispatch.
//!
//! Given a [`mc_core::Graph`], [`evaluate`] walks it in topological order,
//! skips nodes the graph says are already up to date, and dispatches each
//! remaining node's GPU work by its `kind`. Each evaluated node's texture
//! lives in the [`mc_gpu::TexturePool`] under the key from [`texture_key`].

use crate::{
    blend, blur, checker, clamp, clouds, curvature_from_normal, curve, directional_blur,
    directional_warp, edge_detect, fractal_sum, gradient_linear, gradient_map, gradient_radial,
    grayscale_conversion, height_to_ao, height_to_normal, histogram_scan, hsl_adjust, invert,
    levels, perlin, shape, simplex_noise, slope_blur, threshold, tile_generator, transform2d,
    uniform_color, voronoi, warp, white_noise,
};

/// The texture pool key a node's own output is stored under.
pub fn texture_key(id: mc_core::NodeId) -> String {
    format!("node_{}", id.0)
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
pub fn evaluate(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    graph: &mut mc_core::Graph,
) -> Result<(), EvalError> {
    let order = graph.topological_order()?;
    let resolution = graph.settings.resolution;

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

        let key = texture_key(id);
        match node.kind.as_str() {
            "perlin_noise" => {
                let params = perlin::PerlinParams::from_graph(&node.params);
                perlin::dispatch(gpu, pool, &key, resolution, &params);
            }
            "levels" => {
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
                let params = levels::LevelsParams::from_graph(&node.params);
                levels::dispatch(gpu, pool, &input_key, &key, resolution, &params);
            }
            "blend" => {
                let base_id = graph
                    .upstream(id, "base")
                    .ok_or_else(|| EvalError::MissingInput(id, "base".to_string()))?;
                let blend_id = graph
                    .upstream(id, "blend")
                    .ok_or_else(|| EvalError::MissingInput(id, "blend".to_string()))?;
                let base_key = texture_key(base_id);
                let blend_key = texture_key(blend_id);
                let mask_key = graph.upstream(id, "mask").map(texture_key);
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
                let params = tile_generator::TileGeneratorParams::from_graph(&node.params);
                tile_generator::dispatch(gpu, pool, &key, resolution, &params);
            }
            "shape" => {
                let params = shape::ShapeParams::from_graph(&node.params);
                shape::dispatch(gpu, pool, &key, resolution, &params);
            }
            "gradient_linear" => {
                gradient_linear::dispatch(gpu, pool, &key, resolution);
            }
            "gradient_radial" => {
                let params = gradient_radial::GradientRadialParams::from_graph(&node.params);
                gradient_radial::dispatch(gpu, pool, &key, resolution, &params);
            }
            "uniform_color" => {
                let params = uniform_color::UniformColorParams::from_graph(&node.params);
                uniform_color::dispatch(gpu, pool, &key, resolution, &params);
            }
            "invert" => {
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
                invert::dispatch(gpu, pool, &input_key, &key, resolution);
            }
            "histogram_scan" => {
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
                let params = histogram_scan::HistogramScanParams::from_graph(&node.params);
                histogram_scan::dispatch(gpu, pool, &input_key, &key, resolution, &params);
            }
            "blur" => {
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
                let params = blur::BlurParams::from_graph(&node.params);
                blur::dispatch(gpu, pool, &input_key, &key, resolution, &params);
            }
            "warp" => {
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let intensity_id = graph
                    .upstream(id, "intensity")
                    .ok_or_else(|| EvalError::MissingInput(id, "intensity".to_string()))?;
                let input_key = texture_key(input_id);
                let intensity_key = texture_key(intensity_id);
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
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
                let params = transform2d::Transform2DParams::from_graph(&node.params);
                transform2d::dispatch(gpu, pool, &input_key, &key, resolution, &params);
            }
            "height_to_normal" => {
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
                let params = height_to_normal::HeightToNormalParams::from_graph(&node.params);
                height_to_normal::dispatch(gpu, pool, &input_key, &key, resolution, &params);
            }
            "clamp" => {
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
                let params = clamp::ClampParams::from_graph(&node.params);
                clamp::dispatch(gpu, pool, &input_key, &key, resolution, &params);
            }
            "threshold" => {
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
                let params = threshold::ThresholdParams::from_graph(&node.params);
                threshold::dispatch(gpu, pool, &input_key, &key, resolution, &params);
            }
            "grayscale_conversion" => {
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
                grayscale_conversion::dispatch(gpu, pool, &input_key, &key, resolution);
            }
            "checker" => {
                let params = checker::CheckerParams::from_graph(&node.params);
                checker::dispatch(gpu, pool, &key, resolution, &params);
            }
            "hsl_adjust" => {
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
                let params = hsl_adjust::HslAdjustParams::from_graph(&node.params);
                hsl_adjust::dispatch(gpu, pool, &input_key, &key, resolution, &params);
            }
            "curve" => {
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
                let params = curve::CurveParams::from_graph(&node.params);
                curve::dispatch(gpu, pool, &input_key, &key, resolution, &params);
            }
            "gradient_map" => {
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
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
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
                let params = directional_blur::DirectionalBlurParams::from_graph(&node.params);
                directional_blur::dispatch(gpu, pool, &input_key, &key, resolution, &params);
            }
            "directional_warp" => {
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let intensity_id = graph
                    .upstream(id, "intensity")
                    .ok_or_else(|| EvalError::MissingInput(id, "intensity".to_string()))?;
                let input_key = texture_key(input_id);
                let intensity_key = texture_key(intensity_id);
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
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
                let params = edge_detect::EdgeDetectParams::from_graph(&node.params);
                edge_detect::dispatch(gpu, pool, &input_key, &key, resolution, &params);
            }
            "slope_blur" => {
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let control_id = graph
                    .upstream(id, "control")
                    .ok_or_else(|| EvalError::MissingInput(id, "control".to_string()))?;
                let input_key = texture_key(input_id);
                let control_key = texture_key(control_id);
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
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
                let params = height_to_ao::HeightToAoParams::from_graph(&node.params);
                height_to_ao::dispatch(gpu, pool, &input_key, &key, resolution, &params);
            }
            "curvature_from_normal" => {
                let input_id = graph
                    .upstream(id, "input")
                    .ok_or_else(|| EvalError::MissingInput(id, "input".to_string()))?;
                let input_key = texture_key(input_id);
                let params =
                    curvature_from_normal::CurvatureFromNormalParams::from_graph(&node.params);
                curvature_from_normal::dispatch(gpu, pool, &input_key, &key, resolution, &params);
            }
            other => return Err(EvalError::UnknownKind(id, other.to_string())),
        }
        graph.mark_evaluated(id);
    }

    Ok(())
}

/// The texture pool key actually holding `node`'s visible output. For most
/// nodes that's just its own key; an Output node is a pass-through marker,
/// so its "output" is whatever feeds its `input` port.
pub fn resolve_texture_key(graph: &mc_core::Graph, node: mc_core::NodeId) -> Option<String> {
    let kind = &graph.node(node)?.kind;
    if kind.starts_with("output_") {
        let upstream = graph.upstream(node, "input")?;
        resolve_texture_key(graph, upstream)
    } else {
        Some(texture_key(node))
    }
}

#[cfg(test)]
mod tests {
    use super::{evaluate, resolve_texture_key};

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
}
