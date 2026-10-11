// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Export… dialog: pick a preset, resolution, output folder, and graph
//! name, then write the material via `mc_nodes::export::export_material`.

const RESOLUTIONS: [u32; 5] = [256, 512, 1024, 2048, 4096];

pub struct ExportDialogState {
    pub open: bool,
    preset: mc_export::naming::Preset,
    resolution: u32,
    graph_name: String,
    out_dir: Option<std::path::PathBuf>,
    status: Option<Result<Vec<std::path::PathBuf>, String>>,
}

impl Default for ExportDialogState {
    fn default() -> Self {
        Self {
            open: false,
            preset: mc_export::naming::Preset::Blender,
            resolution: 2048,
            graph_name: "material".to_string(),
            out_dir: None,
            status: None,
        }
    }
}

/// Renders the dialog if `state.open`. Mutates `graph` transiently (it's
/// re-evaluated at the export resolution, then restored) via
/// `mc_nodes::export::export_material`.
pub fn show(
    ctx: &egui::Context,
    state: &mut ExportDialogState,
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    graph: &mut mc_core::Graph,
) {
    if !state.open {
        return;
    }

    let mut open = state.open;
    egui::Window::new("Export Material")
        .open(&mut open)
        .resizable(false)
        .show(ctx, |ui| {
            egui::ComboBox::from_label("Preset")
                .selected_text(state.preset.name())
                .show_ui(ui, |ui| {
                    for name in mc_export::naming::Preset::NAMES {
                        let preset = mc_export::naming::Preset::from_name(name)
                            .expect("NAMES and from_name must agree");
                        ui.selectable_value(&mut state.preset, preset, name);
                    }
                });

            egui::ComboBox::from_label("Resolution")
                .selected_text(format!("{0}x{0}", state.resolution))
                .show_ui(ui, |ui| {
                    for &resolution in &RESOLUTIONS {
                        ui.selectable_value(
                            &mut state.resolution,
                            resolution,
                            format!("{resolution}x{resolution}"),
                        );
                    }
                });

            ui.horizontal(|ui| {
                ui.label("Name:");
                ui.text_edit_singleline(&mut state.graph_name);
            });

            ui.horizontal(|ui| {
                ui.label("Folder:");
                let label = state
                    .out_dir
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "(none selected)".to_string());
                ui.weak(label);
                if ui.button("Choose…").clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        state.out_dir = Some(dir);
                        state.status = None;
                    }
                }
            });

            ui.separator();

            let can_export = state.out_dir.is_some() && !state.graph_name.trim().is_empty();
            if ui
                .add_enabled(can_export, egui::Button::new("Export"))
                .clicked()
            {
                let out_dir = state.out_dir.clone().expect("checked by can_export");
                let result = mc_nodes::export::export_material(
                    gpu,
                    pool,
                    graph,
                    state.preset,
                    state.resolution,
                    state.graph_name.trim(),
                    &out_dir,
                )
                .map_err(|err| err.to_string());
                state.status = Some(result);
            }

            if let Some(status) = &state.status {
                match status {
                    Ok(paths) => {
                        ui.colored_label(
                            egui::Color32::LIGHT_GREEN,
                            format!("Wrote {} file(s):", paths.len()),
                        );
                        for path in paths {
                            ui.weak(path.display().to_string());
                        }
                    }
                    Err(err) => {
                        ui.colored_label(egui::Color32::LIGHT_RED, format!("Export failed: {err}"));
                    }
                }
            }
        });
    state.open = open;
}
