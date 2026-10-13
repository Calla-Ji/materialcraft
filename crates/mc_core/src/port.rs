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
    /// Vector path data (PATH_NODE_SPEC.md §1) — not a texture at all.
    /// Connections of this type bypass the GPU texture pool entirely: a
    /// downstream node reads the upstream Path node's own stored params
    /// directly (see `mc_nodes::registry`'s handling of the Stitch node's
    /// `path` input). Only ever compatible with another `Path` port — no
    /// broadcast to/from Gray or Color, unlike Gray->Color.
    Path,
}
