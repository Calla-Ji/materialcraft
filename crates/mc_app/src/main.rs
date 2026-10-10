//! mc_app: the desktop application shell.
//!
//! Owns the window and the overall panel layout (menu, 2D/3D previews, node
//! graph, parameter panel). M3 wires the node editor (egui-snarl) to a real
//! `mc_core::Graph`: add nodes from a search menu, connect them (with Gray/
//! Color type checking), edit params in a generic panel, and undo/redo —
//! see `graph_view.rs` for how the UI mirrors the graph.

mod export_dialog;
mod graph_view;
mod params_panel;

use std::collections::HashMap;

use graph_view::{sync_snarl_from_graph, GraphViewer, NodeUi};

/// Root application state for the eframe event loop.
struct MaterialCraftApp {
    gpu: mc_gpu::Context,
    pool: mc_gpu::TexturePool,
    render_state: egui_wgpu::RenderState,

    graph: mc_core::Graph,
    history: mc_core::History,
    snarl: egui_snarl::Snarl<NodeUi>,
    core_to_ui: HashMap<mc_core::NodeId, egui_snarl::NodeId>,
    thumbnail_ids: HashMap<mc_core::NodeId, egui::TextureId>,

    selected: Option<mc_core::NodeId>,
    add_node_search: String,
    param_drag: params_panel::DragState,
    export_dialog: export_dialog::ExportDialogState,

    preview_texture_id: Option<egui::TextureId>,
}

impl eframe::App for MaterialCraftApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("menu_bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("MaterialCraft").strong());
                ui.separator();
                ui.label("New");
                ui.label("Open");
                ui.label("Save");
                ui.label("Save As");
                if ui.button("Export…").clicked() {
                    self.export_dialog.open = true;
                }
                ui.separator();
                if ui
                    .add_enabled(self.history.can_undo(), egui::Button::new("Undo"))
                    .clicked()
                {
                    self.history.undo(&mut self.graph);
                }
                if ui
                    .add_enabled(self.history.can_redo(), egui::Button::new("Redo"))
                    .clicked()
                {
                    self.history.redo(&mut self.graph);
                }
                ui.label("Examples");
            });
        });

        // Reflect the previous frame's edits (including anything undo/redo
        // just changed) into the editor widget, then re-evaluate whatever
        // that left dirty, before any panel reads from `self.pool`.
        sync_snarl_from_graph(&self.graph, &mut self.snarl, &mut self.core_to_ui);
        if let Err(err) =
            mc_nodes::registry::evaluate(&mut self.gpu, &mut self.pool, &mut self.graph)
        {
            log::warn!("graph evaluation error: {err}");
        }

        egui::Panel::right("parameters")
            .default_size(280.0)
            .show(ui, |ui| {
                params_panel::show(
                    ui,
                    &mut self.graph,
                    &mut self.history,
                    self.selected,
                    &mut self.param_drag,
                );
            });

        egui::Panel::top("previews")
            .default_size(260.0)
            .resizable(true)
            .show(ui, |ui| {
                ui.columns(2, |columns| {
                    columns[0].heading("2D View");
                    match self.preview_texture() {
                        Some(texture_id) => {
                            let size = columns[0].available_height().min(240.0);
                            columns[0].image(egui::load::SizedTexture::new(
                                texture_id,
                                egui::Vec2::splat(size),
                            ));
                        }
                        None => {
                            columns[0].weak("(select a node to preview it)");
                        }
                    }
                    columns[1].heading("3D View");
                    columns[1].weak("(sphere / cube / plane, PBR lit)");
                });
            });

        egui::CentralPanel::default().show(ui, |ui| {
            let style = egui_snarl::ui::SnarlStyle::new();
            let mut viewer = GraphViewer {
                graph: &mut self.graph,
                history: &mut self.history,
                selected: &mut self.selected,
                add_node_search: &mut self.add_node_search,
                gpu: &mut self.gpu,
                pool: &mut self.pool,
                render_state: &self.render_state,
                thumbnail_ids: &mut self.thumbnail_ids,
            };
            self.snarl.show(&mut viewer, &style, "material_graph", ui);
        });

        export_dialog::show(
            ui.ctx(),
            &mut self.export_dialog,
            &mut self.gpu,
            &mut self.pool,
            &mut self.graph,
        );
    }
}

impl MaterialCraftApp {
    /// The selected node's resolved output texture, registered/updated as an
    /// egui texture for the 2D view.
    fn preview_texture(&mut self) -> Option<egui::TextureId> {
        let id = self.selected?;
        let key = mc_nodes::registry::resolve_texture_key(&self.graph, id)?;
        let view = self.pool.view(&key)?.clone();

        Some(match self.preview_texture_id {
            Some(texture_id) => {
                self.render_state
                    .renderer
                    .write()
                    .update_egui_texture_from_wgpu_texture(
                        &self.gpu.device,
                        &view,
                        wgpu::FilterMode::Linear,
                        texture_id,
                    );
                texture_id
            }
            None => {
                let texture_id = self.render_state.renderer.write().register_native_texture(
                    &self.gpu.device,
                    &view,
                    wgpu::FilterMode::Linear,
                );
                self.preview_texture_id = Some(texture_id);
                texture_id
            }
        })
    }
}

fn main() -> eframe::Result<()> {
    env_logger::init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("MaterialCraft"),
        ..Default::default()
    };

    eframe::run_native(
        "MaterialCraft",
        native_options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());

            let render_state = cc
                .wgpu_render_state
                .clone()
                .expect("mc_app requires eframe's wgpu backend");
            let gpu = mc_gpu::Context::new(render_state.device.clone(), render_state.queue.clone());

            Ok(Box::new(MaterialCraftApp {
                gpu,
                pool: mc_gpu::TexturePool::default(),
                render_state,
                graph: mc_core::Graph::new(),
                history: mc_core::History::new(),
                snarl: egui_snarl::Snarl::new(),
                core_to_ui: HashMap::new(),
                thumbnail_ids: HashMap::new(),
                selected: None,
                add_node_search: String::new(),
                param_drag: None,
                export_dialog: export_dialog::ExportDialogState::default(),
                preview_texture_id: None,
            }))
        }),
    )
}
