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
    /// Signed distance field, tile units (1.0 = tile width), negative
    /// inside / positive outside / 0 on the edge
    /// (`docs/hardsurface_nodes/HARDSURFACE_NODES.md` §2) — stored as an
    /// ordinary single-channel texture (R of Rgba16Float), same GPU
    /// footprint as `Gray`, but never silently interchangeable with it:
    /// only ever compatible with another `Sdf` port, no broadcast to/from
    /// Gray or Color — same strict rule `Path` uses above, for the same
    /// reason (this carries geometric distance, not plain brightness).
    /// `Sdf -> Greyscale` only through a future Bevel Profile node (or an
    /// explicit "SDF to Mask"); `Greyscale -> Sdf` only through a future
    /// Distance Field node — never an implicit conversion.
    Sdf,
}
