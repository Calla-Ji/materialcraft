// SPDX-License-Identifier: MIT OR Apache-2.0
//! The graph model: nodes, connections, settings, evaluation order, and the
//! dirty/cache tracking that decides what needs re-evaluating.
//!
//! This crate is deliberately GPU- and UI-free (CLAUDE.md §5) so it can be
//! unit-tested on its own; actually running a node's shader lives in
//! `mc_nodes`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::connection::Connection;
use crate::id::NodeId;
use crate::node::Node;
use crate::param::ParamValue;

/// Current `.mcgraph` JSON schema version. Bump this and add a migration
/// whenever `Graph`'s shape changes in a way old files can't parse directly
/// — never break loading of existing example graphs.
pub const FORMAT_VERSION: u32 = 1;

/// Normal map convention: Blender expects OpenGL (Y+), Unreal expects
/// DirectX (Y-). A graph-wide default, overridable at export (CLAUDE.md §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalFormat {
    OpenGl,
    DirectX,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSettings {
    pub resolution: u32,
    pub seed: u32,
    pub normal_format: NormalFormat,
}

impl Default for GraphSettings {
    fn default() -> Self {
        Self {
            resolution: 1024,
            seed: 0,
            normal_format: NormalFormat::OpenGl,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GraphError {
    #[error("node {0:?} does not exist")]
    NoSuchNode(NodeId),
    #[error("connecting {from:?}:{from_port} to {to:?}:{to_port} would create a cycle")]
    WouldCreateCycle {
        from: NodeId,
        from_port: String,
        to: NodeId,
        to_port: String,
    },
    #[error("graph contains a cycle")]
    Cycle,
}

/// Failure loading a `.mcgraph` file.
#[derive(Debug, Error)]
pub enum LoadError {
    #[error("failed to read graph file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse graph JSON: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub format_version: u32,
    #[serde(default)]
    pub settings: GraphSettings,
    pub(crate) nodes: BTreeMap<NodeId, Node>,
    pub(crate) connections: Vec<Connection>,

    /// Derived from `nodes` after load, never persisted (see `from_json`).
    #[serde(skip, default)]
    pub(crate) next_id: u64,
    /// Which nodes have been explicitly invalidated (new, edited, or
    /// downstream of something that was) since they were last evaluated.
    #[serde(skip, default)]
    pub(crate) dirty: HashSet<NodeId>,
    /// The evaluation hash each node had the last time it was evaluated.
    #[serde(skip, default)]
    pub(crate) evaluated_hashes: HashMap<NodeId, u64>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            format_version: FORMAT_VERSION,
            settings: GraphSettings::default(),
            nodes: BTreeMap::new(),
            connections: Vec::new(),
            next_id: 0,
            dirty: HashSet::new(),
            evaluated_hashes: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.next_id);
        self.do_command(crate::command::Command::AddNode { id, node })
            .expect("adding a node at a fresh id never fails");
        id
    }

    /// Remove a node and any connections touching it (as a single undoable
    /// transaction — see `remove_node_transaction`). Anything downstream of
    /// it loses an input and is marked dirty (it'll fall back to that
    /// input's default).
    pub fn remove_node(&mut self, id: NodeId) {
        for command in self.remove_node_transaction(id) {
            let _ = self.do_command(command);
        }
    }

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes.iter().map(|(&id, node)| (id, node))
    }

    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// Update a node's editor canvas position. Purely cosmetic — it doesn't
    /// affect evaluation, so unlike other edits it isn't undo-tracked (a
    /// dragged node snapping back on undo would be surprising).
    pub fn set_node_position(&mut self, id: NodeId, position: [f32; 2]) -> Result<(), GraphError> {
        let node = self.nodes.get_mut(&id).ok_or(GraphError::NoSuchNode(id))?;
        node.position = position;
        Ok(())
    }

    pub fn set_param(
        &mut self,
        id: NodeId,
        name: &str,
        value: ParamValue,
    ) -> Result<(), GraphError> {
        if !self.nodes.contains_key(&id) {
            return Err(GraphError::NoSuchNode(id));
        }
        self.do_command(crate::command::Command::SetParam {
            id,
            name: name.to_string(),
            value,
        })
        .map(|_| ())
    }

    /// Connect `from_node:from_port` to `to_node:to_port`, replacing any
    /// existing connection into `to_node:to_port` (each input takes one
    /// connection; outputs can fan out). Rejected without changing the graph
    /// if it would create a cycle.
    pub fn connect(
        &mut self,
        from_node: NodeId,
        from_port: &str,
        to_node: NodeId,
        to_port: &str,
    ) -> Result<(), GraphError> {
        for command in self.connect_transaction(from_node, from_port, to_node, to_port)? {
            self.do_command(command)?;
        }
        Ok(())
    }

    pub fn disconnect(&mut self, to_node: NodeId, to_port: &str) {
        if let Some(existing) = self
            .connections
            .iter()
            .find(|c| c.to_node == to_node && c.to_port == to_port)
            .cloned()
        {
            let _ = self.do_command(crate::command::Command::Disconnect(existing));
        }
    }

    /// The node feeding `(to_node, to_port)`, if any.
    pub fn upstream(&self, to_node: NodeId, to_port: &str) -> Option<NodeId> {
        self.connections
            .iter()
            .find(|c| c.to_node == to_node && c.to_port == to_port)
            .map(|c| c.from_node)
    }

    /// A deterministic evaluation order (every node after everything it
    /// depends on). Graphs built through `connect` can never contain a
    /// cycle, but a hand-edited or corrupted `.mcgraph` file in principle
    /// could, hence this stays fallible.
    pub fn topological_order(&self) -> Result<Vec<NodeId>, GraphError> {
        Self::topo_order(&self.nodes, &self.connections).ok_or(GraphError::Cycle)
    }

    pub(crate) fn topo_order(
        nodes: &BTreeMap<NodeId, Node>,
        connections: &[Connection],
    ) -> Option<Vec<NodeId>> {
        let mut in_degree: BTreeMap<NodeId, usize> = nodes.keys().map(|&id| (id, 0)).collect();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for connection in connections {
            *in_degree.get_mut(&connection.to_node)? += 1;
            adjacency
                .entry(connection.from_node)
                .or_default()
                .push(connection.to_node);
        }

        // A `BTreeSet` frontier (rather than whatever order a `HashSet`
        // happens to give) makes the result deterministic for a given graph.
        let mut ready: BTreeSet<NodeId> = in_degree
            .iter()
            .filter(|&(_, &degree)| degree == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut order = Vec::with_capacity(nodes.len());
        while let Some(id) = ready.pop_first() {
            order.push(id);
            if let Some(children) = adjacency.get(&id) {
                for &child in children {
                    let degree = in_degree.get_mut(&child)?;
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert(child);
                    }
                }
            }
        }

        (order.len() == nodes.len()).then_some(order)
    }

    /// Every node reachable by following connections forward from `start`,
    /// not including `start` itself.
    pub(crate) fn downstream_of(&self, start: NodeId) -> HashSet<NodeId> {
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for connection in &self.connections {
            adjacency
                .entry(connection.from_node)
                .or_default()
                .push(connection.to_node);
        }

        let mut seen = HashSet::new();
        let mut stack = vec![start];
        while let Some(id) = stack.pop() {
            if seen.insert(id) {
                if let Some(children) = adjacency.get(&id) {
                    stack.extend(children.iter().copied());
                }
            }
        }
        seen.remove(&start);
        seen
    }

    pub(crate) fn mark_dirty_downstream(&mut self, start: NodeId) {
        self.dirty.insert(start);
        self.dirty.extend(self.downstream_of(start));
    }

    /// A hash of everything that determines this node's output: its kind,
    /// params, the graph's resolution/seed, and (recursively) its inputs'
    /// own hashes. Two evaluations with the same hash would produce the same
    /// pixels.
    pub fn evaluation_hash(&self, id: NodeId) -> Option<u64> {
        let mut memo = HashMap::new();
        self.evaluation_hash_memo(id, &mut memo)
    }

    fn evaluation_hash_memo(&self, id: NodeId, memo: &mut HashMap<NodeId, u64>) -> Option<u64> {
        if let Some(&hash) = memo.get(&id) {
            return Some(hash);
        }
        let node = self.nodes.get(&id)?;

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        node.kind.hash(&mut hasher);
        for (name, value) in &node.params {
            name.hash(&mut hasher);
            value.hash(&mut hasher);
        }
        self.settings.resolution.hash(&mut hasher);
        self.settings.seed.hash(&mut hasher);

        let mut inputs: Vec<&Connection> = self
            .connections
            .iter()
            .filter(|c| c.to_node == id)
            .collect();
        inputs.sort_by(|a, b| a.to_port.cmp(&b.to_port));
        for connection in inputs {
            connection.to_port.hash(&mut hasher);
            let input_hash = self
                .evaluation_hash_memo(connection.from_node, memo)
                .unwrap_or(0);
            input_hash.hash(&mut hasher);
        }

        let hash = hasher.finish();
        memo.insert(id, hash);
        Some(hash)
    }

    /// Whether `id` needs (re-)evaluating: it's explicitly dirty, has never
    /// been evaluated, or its evaluation hash changed since it last was.
    pub fn needs_evaluation(&self, id: NodeId) -> bool {
        if self.dirty.contains(&id) {
            return true;
        }
        match (self.evaluated_hashes.get(&id), self.evaluation_hash(id)) {
            (Some(&cached), Some(current)) => cached != current,
            _ => true,
        }
    }

    /// Record that `id` has been evaluated at its current hash and clear its
    /// dirty flag — the next `needs_evaluation` call is a cache hit unless
    /// something changes first.
    pub fn mark_evaluated(&mut self, id: NodeId) {
        if let Some(hash) = self.evaluation_hash(id) {
            self.evaluated_hashes.insert(id, hash);
        }
        self.dirty.remove(&id);
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// `next_id` is deliberately not persisted — it's derived from the
    /// loaded nodes' ids here, rather than trusted from the file, so a
    /// hand-edited `.mcgraph` can't desync it from reality.
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        let mut graph: Graph = serde_json::from_str(json)?;
        graph.next_id = graph.nodes.keys().map(|id| id.0 + 1).max().unwrap_or(0);
        Ok(graph)
    }

    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::write(
            path,
            self.to_json().expect("Graph always serializes to JSON"),
        )
    }

    pub fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<Self, LoadError> {
        let text = std::fs::read_to_string(path)?;
        Ok(Self::from_json(&text)?)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_graph() -> (Graph, NodeId, NodeId) {
        let mut graph = Graph::new();
        let a = graph.add_node(Node::new("perlin_noise"));
        let b = graph.add_node(Node::new("levels"));
        graph.connect(a, "output", b, "input").unwrap();
        (graph, a, b)
    }

    #[test]
    fn connect_rejects_cycle() {
        let (mut graph, a, b) = linear_graph();
        let err = graph.connect(b, "output", a, "input").unwrap_err();
        assert_eq!(
            err,
            GraphError::WouldCreateCycle {
                from: b,
                from_port: "output".to_string(),
                to: a,
                to_port: "input".to_string(),
            }
        );
        // Rejected connection must not have been applied.
        assert_eq!(graph.connections().len(), 1);
    }

    #[test]
    fn topological_order_respects_dependencies() {
        let (graph, a, b) = linear_graph();
        let order = graph.topological_order().unwrap();
        let pos_a = order.iter().position(|&id| id == a).unwrap();
        let pos_b = order.iter().position(|&id| id == b).unwrap();
        assert!(pos_a < pos_b, "producer must come before consumer");
    }

    #[test]
    fn connecting_to_an_occupied_input_replaces_it() {
        let mut graph = Graph::new();
        let a = graph.add_node(Node::new("perlin_noise"));
        let b = graph.add_node(Node::new("perlin_noise"));
        let c = graph.add_node(Node::new("levels"));
        graph.connect(a, "output", c, "input").unwrap();
        graph.connect(b, "output", c, "input").unwrap();
        assert_eq!(graph.connections().len(), 1);
        assert_eq!(graph.upstream(c, "input"), Some(b));
    }

    #[test]
    fn cache_hit_after_mark_evaluated() {
        let mut graph = Graph::new();
        let id = graph.add_node(Node::new("perlin_noise").with_param("seed", ParamValue::Seed(1)));

        assert!(graph.needs_evaluation(id), "never-evaluated node is dirty");
        graph.mark_evaluated(id);
        assert!(
            !graph.needs_evaluation(id),
            "cache hit right after evaluating"
        );

        graph.set_param(id, "seed", ParamValue::Seed(2)).unwrap();
        assert!(
            graph.needs_evaluation(id),
            "changed param must invalidate the cache"
        );

        graph.mark_evaluated(id);
        assert!(
            !graph.needs_evaluation(id),
            "cache hit again after re-evaluating"
        );
    }

    #[test]
    fn changing_an_upstream_param_dirties_downstream_nodes() {
        let (mut graph, a, b) = linear_graph();
        graph.mark_evaluated(a);
        graph.mark_evaluated(b);

        graph.set_param(a, "seed", ParamValue::Seed(99)).unwrap();
        assert!(graph.needs_evaluation(a));
        assert!(
            graph.needs_evaluation(b),
            "downstream node must be dirtied when its input changes"
        );
    }

    #[test]
    fn save_load_round_trip() {
        let mut original = Graph::new();
        let a = original.add_node(
            Node::new("perlin_noise")
                .with_param("scale", ParamValue::Int(4))
                .with_param("seed", ParamValue::Seed(7)),
        );
        let b = original.add_node(Node::new("levels"));
        original.connect(a, "output", b, "input").unwrap();
        original.settings.resolution = 2048;

        let json = original.to_json().unwrap();
        let loaded = Graph::from_json(&json).unwrap();

        assert_eq!(loaded.format_version, original.format_version);
        assert_eq!(loaded.settings.resolution, original.settings.resolution);
        assert_eq!(loaded.connections(), original.connections());
        for (id, node) in original.nodes() {
            let loaded_node = loaded.node(id).expect("node survives round trip");
            assert_eq!(loaded_node.kind, node.kind);
            assert_eq!(loaded_node.params, node.params);
        }

        // A freshly loaded graph must be able to add a node without
        // colliding with an id from the file.
        let mut loaded = loaded;
        let c = loaded.add_node(Node::new("output_base_color"));
        assert!(c.0 > a.0 && c.0 > b.0);
    }

    #[test]
    fn file_round_trip() {
        let mut graph = Graph::new();
        graph.add_node(Node::new("perlin_noise"));
        let path = std::env::temp_dir().join("mc_core_test_graph.mcgraph");

        graph.save_to_file(&path).unwrap();
        let loaded = Graph::load_from_file(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.nodes().count(), graph.nodes().count());
    }
}
