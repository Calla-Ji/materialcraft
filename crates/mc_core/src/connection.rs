//! An edge in the graph: one node's output port feeding another's input port.

use serde::{Deserialize, Serialize};

use crate::id::NodeId;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Connection {
    pub from_node: NodeId,
    pub from_port: String,
    pub to_node: NodeId,
    pub to_port: String,
}
