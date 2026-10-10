//! Bridges the graph (`mc_core` + [`crate::registry`]) to `mc_export`: finds
//! each Output node present in `graph`, reads its texture back at the
//! requested export resolution, and writes files per the naming preset.

use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum ExportFlowError {
    #[error(transparent)]
    Eval(#[from] crate::registry::EvalError),
    #[error(transparent)]
    Export(#[from] mc_export::ExportError),
    #[error("failed to create output directory {0:?}: {1}")]
    CreateDir(PathBuf, std::io::Error),
}

fn read_output(
    kind: &str,
    resolution: u32,
    graph: &mc_core::Graph,
    pool: &mc_gpu::TexturePool,
    gpu: &mc_gpu::Context,
) -> Option<Vec<u8>> {
    let id = graph
        .nodes()
        .find(|(_, n)| n.kind == kind)
        .map(|(id, _)| id)?;
    let key = crate::registry::resolve_texture_key(graph, id)?;
    let texture = pool.texture(&key)?;
    Some(gpu.read_rgba8(texture, resolution))
}

/// Re-evaluates `graph` at `resolution` (restoring its original resolution
/// afterward, so the live preview isn't left at export resolution) and
/// writes every Output node present in it to `out_dir`, named per `preset`'s
/// convention. Returns the paths written; an Output kind missing from the
/// graph is silently skipped, not an error.
pub fn export_material(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    graph: &mut mc_core::Graph,
    preset: mc_export::naming::Preset,
    resolution: u32,
    graph_name: &str,
    out_dir: &Path,
) -> Result<Vec<PathBuf>, ExportFlowError> {
    std::fs::create_dir_all(out_dir)
        .map_err(|err| ExportFlowError::CreateDir(out_dir.to_path_buf(), err))?;

    let original_resolution = graph.settings.resolution;
    graph.settings.resolution = resolution;
    crate::registry::evaluate(gpu, pool, graph)?;

    let mut written = Vec::new();

    if let Some(pixels) = read_output("output_base_color", resolution, graph, pool, gpu) {
        let path = out_dir.join(preset.base_color_file_name(graph_name));
        mc_export::write_png8(
            &pixels,
            resolution,
            resolution,
            mc_export::ColorSpace::Srgb,
            &path,
        )?;
        written.push(path);
    }

    if let Some(mut pixels) = read_output("output_normal", resolution, graph, pool, gpu) {
        if preset.normal_format() != mc_core::NormalFormat::OpenGl {
            mc_export::flip_normal_green(&mut pixels);
        }
        let path = out_dir.join(preset.normal_file_name(graph_name));
        mc_export::write_png16(
            &pixels,
            resolution,
            resolution,
            mc_export::ColorSpace::Linear,
            &path,
        )?;
        written.push(path);
    }

    if let Some(pixels) = read_output("output_height", resolution, graph, pool, gpu) {
        let path = out_dir.join(preset.height_file_name(graph_name));
        mc_export::write_png16(
            &pixels,
            resolution,
            resolution,
            mc_export::ColorSpace::Linear,
            &path,
        )?;
        written.push(path);
    }

    let ambient_occlusion = read_output("output_ambient_occlusion", resolution, graph, pool, gpu);
    let roughness = read_output("output_roughness", resolution, graph, pool, gpu);
    let metallic = read_output("output_metallic", resolution, graph, pool, gpu);

    match preset {
        mc_export::naming::Preset::Blender => {
            for (pixels, file_name) in [
                (&roughness, preset.roughness_file_name(graph_name)),
                (&metallic, preset.metallic_file_name(graph_name)),
                (
                    &ambient_occlusion,
                    preset.ambient_occlusion_file_name(graph_name),
                ),
            ] {
                if let (Some(pixels), Some(file_name)) = (pixels, file_name) {
                    let path = out_dir.join(file_name);
                    mc_export::write_png8(
                        pixels,
                        resolution,
                        resolution,
                        mc_export::ColorSpace::Linear,
                        &path,
                    )?;
                    written.push(path);
                }
            }
        }
        mc_export::naming::Preset::Unreal => {
            if let (Some(ao), Some(roughness), Some(metallic)) =
                (&ambient_occlusion, &roughness, &metallic)
            {
                let packed = mc_export::pack_orm(ao, roughness, metallic, resolution, resolution)?;
                let path = out_dir.join(
                    preset
                        .orm_file_name(graph_name)
                        .expect("Unreal preset always has an ORM file name"),
                );
                mc_export::write_png8(
                    &packed,
                    resolution,
                    resolution,
                    mc_export::ColorSpace::Linear,
                    &path,
                )?;
                written.push(path);
            }
        }
    }

    if let Some(pixels) = read_output("output_emissive", resolution, graph, pool, gpu) {
        // CLAUDE.md §8's naming examples don't cover Emissive explicitly;
        // following the same `{name}_<Channel>` pattern as the others.
        let path = out_dir.join(format!("{graph_name}_Emissive.png"));
        mc_export::write_png8(
            &pixels,
            resolution,
            resolution,
            mc_export::ColorSpace::Srgb,
            &path,
        )?;
        written.push(path);
    }

    graph.settings.resolution = original_resolution;
    crate::registry::evaluate(gpu, pool, graph)?;

    Ok(written)
}
