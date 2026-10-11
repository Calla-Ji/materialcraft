// SPDX-License-Identifier: MIT OR Apache-2.0
//! Node identifiers.

use serde::{Deserialize, Serialize};

/// A stable identifier for a node within a [`crate::Graph`]. Assigned when
/// the node is added and never reused, even if the node is later removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);
