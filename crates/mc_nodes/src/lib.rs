//! mc_nodes: node definitions — metadata, params, and the WGSL compute
//! shader that implements each one. [`registry`] bridges mc_core's generic
//! graph model to actually dispatching each node's GPU work.
//!
//! Each node's tests live alongside it in its own `#[cfg(test)] mod tests`,
//! sharing the helpers in `test_support`.

pub mod blend;
pub mod blur;
pub mod checker;
pub mod clamp;
pub mod clouds;
pub mod curvature_from_normal;
pub mod curve;
pub mod defs;
pub mod directional_blur;
pub mod directional_warp;
pub mod edge_detect;
pub mod export;
pub mod fractal_sum;
pub mod gradient_linear;
pub mod gradient_map;
pub mod gradient_radial;
pub mod grayscale_conversion;
pub mod height_to_ao;
pub mod height_to_normal;
pub mod histogram_scan;
pub mod hsl_adjust;
pub mod invert;
pub mod levels;
pub mod perlin;
pub mod registry;
mod shaders;
pub mod shape;
pub mod simplex_noise;
pub mod slope_blur;
#[cfg(test)]
mod test_support;
pub mod threshold;
pub mod tile_generator;
pub mod transform2d;
pub mod uniform_color;
pub mod voronoi;
pub mod warp;
pub mod white_noise;
