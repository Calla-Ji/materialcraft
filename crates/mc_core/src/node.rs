//! A single node instance within a graph.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::param::ParamValue;

/// One node in the graph: what kind it is, its parameter values, and its
/// editor position. The node's *definition* — display name, what params it
/// has, defaults, its WGSL shader — lives in `mc_nodes`, keyed by `kind`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// The node type, e.g. `"perlin_noise"`. Looked up in `mc_nodes`'
    /// registry to know how to evaluate it.
    pub kind: String,
    /// A `BTreeMap` (not `HashMap`) so `.mcgraph` JSON serializes params in a
    /// stable, diffable order.
    #[serde(default)]
    pub params: BTreeMap<String, ParamValue>,
    /// Editor canvas position; purely cosmetic, ignored by evaluation.
    #[serde(default)]
    pub position: [f32; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl Node {
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            params: BTreeMap::new(),
            position: [0.0, 0.0],
            label: None,
        }
    }

    pub fn with_param(mut self, name: impl Into<String>, value: ParamValue) -> Self {
        self.params.insert(name.into(), value);
        self
    }
}
