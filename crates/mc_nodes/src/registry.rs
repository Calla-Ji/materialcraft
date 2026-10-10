//! Bridges mc_core's generic graph model to actual GPU node dispatch.
//!
//! Given a [`mc_core::Graph`], [`evaluate`] walks it in topological order,
//! skips nodes the graph says are already up to date, and dispatches each
//! remaining node's GPU work by its `kind`. Each evaluated node's texture
//! lives in the [`mc_gpu::TexturePool`] under the key from [`texture_key`].

use crate::{blend, levels, perlin};

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
