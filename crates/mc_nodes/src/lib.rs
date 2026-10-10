//! mc_nodes: node definitions — metadata, params, and the WGSL compute
//! shader that implements each one. [`registry`] bridges mc_core's generic
//! graph model to actually dispatching each node's GPU work.
//!
//! Each node's tests live alongside it in its own `#[cfg(test)] mod tests`,
//! sharing the helpers in `test_support`.

pub mod blend;
pub mod blur;
pub mod defs;
pub mod export;
pub mod fractal_sum;
pub mod gradient_linear;
pub mod gradient_radial;
pub mod height_to_normal;
pub mod histogram_scan;
pub mod invert;
pub mod levels;
pub mod perlin;
pub mod registry;
mod shaders;
pub mod shape;
#[cfg(test)]
mod test_support;
pub mod tile_generator;
pub mod transform2d;
pub mod uniform_color;
pub mod voronoi;
pub mod warp;
