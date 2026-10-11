// SPDX-License-Identifier: MIT OR Apache-2.0
//! Port types a node's inputs/outputs can carry (CLAUDE.md §5).
//!
//! `mc_core` stores this for completeness but doesn't enforce compatibility
//! itself — the UI layer (M3) blocks invalid connections using each node's
//! port-type metadata from `mc_nodes`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PortType {
    /// Single-channel (R16Float/R32Float on the GPU).
    Gray,
    /// RGBA (RGBA16Float on the GPU).
    Color,
}
