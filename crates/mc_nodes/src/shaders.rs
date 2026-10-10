//! Shader source loading, with hot-reload from disk in debug builds.
//!
//! WGSL has no `#include`, so a node's full shader is assembled by hand: the
//! shared helpers in `common.wgsl` are prepended to the node-specific source
//! before it's handed to `mc_gpu` for compilation.

use std::borrow::Cow;

/// The full WGSL source for a node: `common.wgsl` followed by `file_name`.
pub fn full_source(file_name: &'static str) -> String {
    format!("{}\n{}", node_source("common.wgsl"), node_source(file_name))
}

/// The WGSL source for a single shader file.
///
/// In debug builds this is read from disk on every call, so editing a
/// `.wgsl` file and re-evaluating the node picks up the change without
/// restarting the app. Release builds use the copy embedded at compile time.
fn node_source(file_name: &'static str) -> Cow<'static, str> {
    #[cfg(debug_assertions)]
    {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("shaders")
            .join(file_name);
        match std::fs::read_to_string(&path) {
            Ok(source) => return Cow::Owned(source),
            Err(err) => {
                log::warn!(
                    "could not hot-reload shader {}: {err}; using the embedded copy",
                    path.display()
                );
            }
        }
    }
    Cow::Borrowed(embedded_source(file_name))
}

fn embedded_source(file_name: &str) -> &'static str {
    match file_name {
        "common.wgsl" => include_str!("../shaders/common.wgsl"),
        "perlin_noise.wgsl" => include_str!("../shaders/perlin_noise.wgsl"),
        other => panic!("mc_nodes: unknown shader file {other:?}"),
    }
}
