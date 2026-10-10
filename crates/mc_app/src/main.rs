//! mc_app: the desktop application shell.
//!
//! Owns the window and the overall panel layout (menu, 2D/3D previews, node
//! graph, parameter panel). M3 wires the node editor (egui-snarl) to a real
//! `mc_core::Graph`: add nodes from a search menu, connect them (with Gray/
//! Color type checking), edit params in a generic panel, and undo/redo —
//! see `graph_view.rs` for how the UI mirrors the graph.

mod add_node_menu;
mod export_dialog;
mod graph_view;
mod hdri;
mod params_panel;
mod viewport3d;

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
    add_node_menu: Option<add_node_menu::AddNodeMenu>,
    param_drag: params_panel::DragState,
    export_dialog: export_dialog::ExportDialogState,
    viewport3d: viewport3d::Viewport3d,

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
                            let size = columns[0]
                                .available_height()
                                .min(columns[0].available_width());
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
                    columns[1].horizontal(|ui| {
                        egui::ComboBox::from_label("Mesh")
                            .selected_text(self.viewport3d.mesh_kind.label())
                            .show_ui(ui, |ui| {
                                for kind in viewport3d::MeshKind::ALL {
                                    ui.selectable_value(
                                        &mut self.viewport3d.mesh_kind,
                                        kind,
                                        kind.label(),
                                    );
                                }
                            });
                        ui.add(
                            egui::Slider::new(&mut self.viewport3d.height_scale, 0.0..=0.5)
                                .text("Height Scale"),
                        );
                    });
                    let desired =
                        egui::vec2(columns[1].available_width(), columns[1].available_height());
                    let response =
                        columns[1].allocate_response(desired, egui::Sense::click_and_drag());
                    self.viewport3d.handle_input(&columns[1], &response);
                    let texture_id = self.viewport3d.render(
                        &self.gpu,
                        &self.pool,
                        &self.graph,
                        &self.render_state,
                        response.rect.width().max(1.0) as u32,
                        response.rect.height().max(1.0) as u32,
                    );
                    columns[1].painter().image(
                        texture_id,
                        response.rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                });
            });

        let mut snarl_response = None;
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
            // `Snarl::show` itself discards the interaction `Response` we
            // need for `add_node_menu` below — `SnarlWidget::show` is the
            // lower-level entry point that returns it.
            snarl_response = Some(
                egui_snarl::ui::SnarlWidget::new()
                    .id_salt("material_graph")
                    .style(style)
                    .show(&mut self.snarl, &mut viewer, ui),
            );
        });
        let snarl_response = snarl_response.expect("CentralPanel::show always runs add_contents");

        // Right-click-on-empty-canvas now opens our own screen-space popup
        // (see add_node_menu.rs) instead of egui-snarl's built-in one, which
        // doesn't account for the canvas's pan/zoom transform.
        //
        // `snarl_response` lives inside egui-snarl's own transformed layer,
        // so egui already reports its `interact_pointer_pos()` in graph-space
        // (egui-snarl relies on this same behavior itself, e.g. for rect-
        // selection) — no manual pan/zoom math needed for `graph_pos`. The
        // popup itself is a plain `egui::Area` outside that layer, so it
        // needs the real, untransformed screen-space pointer position.
        if self.add_node_menu.is_none() && snarl_response.secondary_clicked() {
            if let (Some(graph_pos), Some(screen_pos)) = (
                snarl_response.interact_pointer_pos(),
                ui.ctx().input(|i| i.pointer.interact_pos()),
            ) {
                self.add_node_menu = Some(add_node_menu::AddNodeMenu::open(screen_pos, graph_pos));
            }
        }
        if let Some((def, graph_pos)) = add_node_menu::show(ui.ctx(), &mut self.add_node_menu) {
            let node = def.instantiate();
            let (core_id, transaction) = self.graph.add_node_transaction(node);
            if self
                .history
                .do_transaction(&mut self.graph, transaction)
                .is_ok()
            {
                let _ = self
                    .graph
                    .set_node_position(core_id, [graph_pos.x, graph_pos.y]);
                self.selected = Some(core_id);
            }
        }

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
            let viewport3d = viewport3d::Viewport3d::new(&gpu);

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
                add_node_menu: None,
                param_drag: None,
                export_dialog: export_dialog::ExportDialogState::default(),
                viewport3d,
                preview_texture_id: None,
            }))
        }),
    )
}
