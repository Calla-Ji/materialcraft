//! mc_app: the desktop application shell.
//!
//! Owns the window and the overall panel layout (menu, 2D/3D previews, node
//! graph, parameter panel). The node editor (`canvas.rs`) is a fully custom
//! pan/zoom/select/connect canvas built directly on egui, reading and
//! writing `mc_core::Graph` through the same `Command`/`Transaction` API as
//! every other part of the app: add nodes from a search menu, connect them
//! (with Gray/Color type checking), edit params in a generic panel, and
//! undo/redo.

mod add_node_menu;
mod autosave;
mod canvas;
mod display_encode;
mod dock;
mod export_dialog;
mod hdri;
mod mipgen;
mod node_icons;
mod params_panel;
mod theme;
mod toast;
mod viewport3d;
mod viewport3d_panels;

use std::collections::HashMap;

use canvas::CanvasState;

/// Root application state for the eframe event loop.
struct MaterialCraftApp {
    gpu: mc_gpu::Context,
    pool: mc_gpu::TexturePool,
    render_state: egui_wgpu::RenderState,

    graph: mc_core::Graph,
    history: mc_core::History,
    canvas: CanvasState,
    thumbnail_ids: HashMap<mc_core::NodeId, egui::TextureId>,

    selected: Option<mc_core::NodeId>,
    add_node_menu: Option<add_node_menu::AddNodeMenu>,
    pending_wire_connect: Option<canvas::DroppedWire>,
    param_drag: params_panel::DragState,
    export_dialog: export_dialog::ExportDialogState,
    viewport3d: viewport3d::Viewport3d,
    viewport_panels: viewport3d_panels::ViewportPanels,
    dock: dock::DockLayout,

    preview_texture_id: Option<egui::TextureId>,

    /// Where the current graph was last loaded from or saved to — `None`
    /// for a never-saved graph, in which case Save behaves like Save As.
    current_path: Option<std::path::PathBuf>,

    toasts: toast::Toasts,
    /// The last graph-evaluation error's message, if the graph is currently
    /// failing to evaluate — `evaluate()` runs every frame, so without this
    /// a persistent error would spam a fresh toast on every single frame
    /// instead of once when it first appears.
    last_eval_error: Option<String>,

    autosave: autosave::Autosave,
    /// Set at startup if a recovery file from a previous, uncleanly-ended
    /// session was found — shows the restore/discard prompt until the
    /// artist answers it.
    pending_recovery: bool,
}

impl eframe::App for MaterialCraftApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("menu_bar")
            .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(12, 8)))
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.x = 10.0;
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("MaterialCraft")
                            .strong()
                            .color(theme::SUPERNOVA_LIGHT),
                    );
                    ui.add_space(4.0);
                    ui.separator();
                    if ui.button("New").clicked() {
                        self.do_new();
                    }
                    if ui.button("Open").clicked() {
                        self.do_open();
                    }
                    if ui.button("Save").clicked() {
                        self.do_save();
                    }
                    if ui.button("Save As").clicked() {
                        self.do_save_as();
                    }
                    if ui.button("Export…").clicked() {
                        self.export_dialog.open = true;
                    }
                    ui.add_space(4.0);
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
                    ui.add_space(4.0);
                    ui.separator();
                    ui.label("Examples");
                });
            });

        self.toasts.show(ui.ctx());
        self.autosave.maybe_save(&self.graph);

        if self.pending_recovery {
            egui::Window::new("Recover unsaved work?")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ui.ctx(), |ui| {
                    ui.label(
                        "MaterialCraft didn't close cleanly last time — an autosave from \
                         that session is available.",
                    );
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Restore").clicked() {
                            match mc_core::Graph::load_from_file(autosave::recovery_path()) {
                                Ok(graph) => {
                                    // Clears the just-read recovery file too
                                    // (see `replace_graph`'s doc comment).
                                    self.replace_graph(graph, None);
                                    self.toasts.info("Restored autosave from previous session");
                                }
                                Err(err) => {
                                    log::warn!("failed to restore autosave: {err}");
                                    self.toasts
                                        .error(format!("Couldn't restore autosave: {err}"));
                                    autosave::Autosave::clear();
                                }
                            }
                            self.pending_recovery = false;
                        }
                        if ui.button("Discard").clicked() {
                            autosave::Autosave::clear();
                            self.pending_recovery = false;
                        }
                    });
                });
        }

        // Re-evaluate whatever the previous frame's edits (including
        // anything undo/redo just changed) left dirty, before any panel
        // reads from `self.pool`.
        match mc_nodes::registry::evaluate(&mut self.gpu, &mut self.pool, &mut self.graph) {
            Ok(()) => self.last_eval_error = None,
            Err(err) => {
                log::warn!("graph evaluation error: {err}");
                let message = err.to_string();
                if self.last_eval_error.as_deref() != Some(message.as_str()) {
                    self.toasts
                        .error(format!("Graph evaluation failed: {message}"));
                }
                self.last_eval_error = Some(message);
            }
        }

        // 2D View, 3D View, Node Graph, and Parameters are dock cells (see
        // dock.rs) laid out from raw rects we own, rather than
        // `egui::Window`/Panel/columns — none of those auto-size to content,
        // so nothing here can corrupt a sibling's measured size the way the
        // old nested layout did, and dragging a border always keeps both
        // neighbors flush since they read the same stored split ratio.
        let bounds = ui.available_rect_before_wrap();

        self.viewport_panels.update_anim(ui.ctx());
        let drawer_width = self.viewport_panels.current_width();
        let preview_texture_id = self.preview_texture();
        let mut canvas_output = None;

        // Disjoint borrows of the fields each dock cell's content needs, so
        // the closure below doesn't have to borrow `self` as a whole while
        // `self.dock` is already borrowed by the `show` call it's passed to.
        let MaterialCraftApp {
            dock,
            graph,
            history,
            canvas,
            thumbnail_ids,
            selected,
            param_drag,
            viewport3d,
            viewport_panels,
            gpu,
            pool,
            render_state,
            ..
        } = self;

        dock.show(
            ui,
            bounds,
            |id| match id {
                dock::CrateId::TwoDView => "2D View",
                dock::CrateId::ThreeDView => "3D View",
                dock::CrateId::NodeGraph => "Node Graph",
                dock::CrateId::Parameters => "Parameters",
                dock::CrateId::NodeLibrary => "Nodes",
            },
            &mut |ui, id| match id {
                dock::CrateId::Parameters => {
                    params_panel::show(ui, graph, history, *selected, param_drag);
                }
                dock::CrateId::TwoDView => match preview_texture_id {
                    Some(texture_id) => {
                        let size = ui.available_height().min(ui.available_width());
                        ui.image(egui::load::SizedTexture::new(
                            texture_id,
                            egui::Vec2::splat(size),
                        ));
                    }
                    None => {
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            ui.add_space(8.0);
                            ui.weak("(select a node to preview it)");
                        });
                    }
                },
                dock::CrateId::ThreeDView => {
                    const TOOLBAR_WIDTH: f32 = 38.0;
                    let full_rect = ui.available_rect_before_wrap();
                    let divider_stroke = ui.visuals().window_stroke();

                    let toolbar_rect = egui::Rect::from_min_size(
                        full_rect.min,
                        egui::vec2(TOOLBAR_WIDTH, full_rect.height()),
                    );
                    {
                        let mut toolbar_ui =
                            ui.new_child(egui::UiBuilder::new().max_rect(toolbar_rect));
                        toolbar_ui.set_clip_rect(toolbar_rect);
                        viewport_panels.toolbar_vertical(&mut toolbar_ui);
                    }
                    ui.painter().vline(
                        toolbar_rect.right(),
                        toolbar_rect.y_range(),
                        divider_stroke,
                    );

                    // The viewport always fills everything right of the
                    // toolbar — the side panel overlaps it instead of
                    // squeezing it narrower, matching the classic N-panel
                    // behavior (Blender) rather than a push-aside layout.
                    let viewport_rect = egui::Rect::from_min_max(
                        egui::pos2(toolbar_rect.right() + 1.0, full_rect.top()),
                        full_rect.max,
                    );
                    let response = ui.interact(
                        viewport_rect,
                        ui.id().with("viewport3d_interact"),
                        egui::Sense::click_and_drag(),
                    );
                    viewport3d.handle_input(ui, &response);
                    let texture_id = viewport3d.render(
                        gpu,
                        pool,
                        graph,
                        render_state,
                        viewport_rect.width().max(1.0) as u32,
                        viewport_rect.height().max(1.0) as u32,
                    );
                    ui.painter().image(
                        texture_id,
                        viewport_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );

                    if drawer_width > 0.5 {
                        let drawer_rect = egui::Rect::from_min_size(
                            egui::pos2(viewport_rect.right() - drawer_width, viewport_rect.top()),
                            egui::vec2(drawer_width, viewport_rect.height()),
                        );
                        ui.painter()
                            .rect_filled(drawer_rect, 0.0, theme::NEUTRON_STAR);
                        ui.painter().rect_stroke(
                            drawer_rect,
                            0.0,
                            divider_stroke,
                            egui::StrokeKind::Inside,
                        );
                        // Absorbs clicks/drags over empty space in the
                        // drawer so they don't fall through to the
                        // viewport's orbit-drag interact underneath it.
                        ui.interact(
                            drawer_rect,
                            ui.id().with("viewport3d_drawer_bg"),
                            egui::Sense::click_and_drag(),
                        );

                        let inner_rect = drawer_rect.shrink2(egui::vec2(10.0, 8.0));
                        let mut drawer_ui =
                            ui.new_child(egui::UiBuilder::new().max_rect(inner_rect));
                        drawer_ui.set_clip_rect(inner_rect);
                        let ui = &mut drawer_ui;

                        let kind = viewport_panels.visible_kind();
                        ui.heading(kind.title());
                        ui.weak(kind.description());
                        ui.separator();
                        match kind {
                            viewport3d_panels::PanelKind::Mesh => {
                                egui::ComboBox::from_label("Shape")
                                    .selected_text(viewport3d.mesh_kind.label())
                                    .show_ui(ui, |ui| {
                                        for mesh_kind in viewport3d::MeshKind::ALL {
                                            ui.selectable_value(
                                                &mut viewport3d.mesh_kind,
                                                mesh_kind,
                                                mesh_kind.label(),
                                            );
                                        }
                                    });
                            }
                            viewport3d_panels::PanelKind::Material => {
                                ui.spacing_mut().slider_width = 80.0;
                                ui.add(
                                    egui::Slider::new(&mut viewport3d.height_scale, 0.0..=0.5)
                                        .text("Height Scale"),
                                );
                                ui.add(
                                    egui::Slider::new(&mut viewport3d.tessellation, 0.0..=1.0)
                                        .text("Tessellation Density"),
                                );
                                ui.separator();
                                let mut unlit = viewport3d.shading_mode
                                    == viewport3d::ShadingMode::UnlitBaseColor;
                                if ui
                                    .checkbox(&mut unlit, "Unlit Base Color")
                                    .on_hover_text(
                                        "Shows Base Color directly, with no lighting, \
                                         exposure, or tonemapping \u{2014} for checking it \
                                         matches the 2D View and an exported PNG exactly.",
                                    )
                                    .changed()
                                {
                                    viewport3d.shading_mode = if unlit {
                                        viewport3d::ShadingMode::UnlitBaseColor
                                    } else {
                                        viewport3d::ShadingMode::Lit
                                    };
                                }
                                ui.add_enabled(
                                    !unlit,
                                    egui::Slider::new(&mut viewport3d.exposure, 0.1..=4.0)
                                        .text("Exposure"),
                                );
                                ui.separator();
                                let mut pixel_filter =
                                    viewport3d.texture_filter == viewport3d::TextureFilter::Pixel;
                                if ui
                                    .checkbox(&mut pixel_filter, "Pixel / Nearest Filtering")
                                    .on_hover_text(
                                        "Disables all texture smoothing (texel and mip) \u{2014} \
                                         for inspecting dither and pixel-art materials crisply \
                                         instead of blended.",
                                    )
                                    .changed()
                                {
                                    viewport3d.texture_filter = if pixel_filter {
                                        viewport3d::TextureFilter::Pixel
                                    } else {
                                        viewport3d::TextureFilter::Smooth
                                    };
                                }
                            }
                            viewport3d_panels::PanelKind::Camera
                            | viewport3d_panels::PanelKind::Environment => {}
                        }
                    }
                }
                dock::CrateId::NodeGraph => {
                    canvas_output = Some(canvas::show(
                        ui,
                        graph,
                        history,
                        canvas,
                        selected,
                        gpu,
                        pool,
                        render_state,
                        thumbnail_ids,
                    ));
                }
                dock::CrateId::NodeLibrary => {
                    show_node_library(ui, graph, history, canvas, selected);
                }
            },
        );

        let canvas_output = canvas_output.unwrap_or(canvas::CanvasOutput {
            open_add_menu_at: None,
            dropped_wire: None,
        });

        // Cmd+A (select all) / Delete, skipped while a text field (e.g. the
        // add-node search box, a parameter's numeric entry) has focus, so
        // those keys keep doing normal text editing instead of being
        // hijacked into graph-wide actions.
        if ui.memory(|m| m.focused()).is_none() {
            let (select_all, delete_pressed, new_pressed, open_pressed, save_pressed) =
                ui.ctx().input(|i| {
                    (
                        i.modifiers.command && i.key_pressed(egui::Key::A),
                        i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace),
                        i.modifiers.command && i.key_pressed(egui::Key::N),
                        i.modifiers.command && i.key_pressed(egui::Key::O),
                        i.modifiers.command && i.key_pressed(egui::Key::S),
                    )
                });
            // Shift+Cmd+S (Save As) vs. plain Cmd+S (Save): read `modifiers`
            // separately from the tuple above, since `key_pressed` already
            // consumed the event by the time `save_pressed` is true.
            let save_as = save_pressed && ui.ctx().input(|i| i.modifiers.shift);

            if select_all {
                self.canvas.selected = self.graph.nodes().map(|(id, _)| id).collect();
            }

            if delete_pressed && !self.canvas.selected.is_empty() {
                let ids: Vec<mc_core::NodeId> = self.canvas.selected.iter().copied().collect();
                let transaction = self.graph.remove_nodes_transaction(&ids);
                let _ = self.history.do_transaction(&mut self.graph, transaction);
                if self.selected.is_some_and(|id| ids.contains(&id)) {
                    self.selected = None;
                }
                self.canvas.selected.clear();
            }

            if new_pressed {
                self.do_new();
            }
            if open_pressed {
                self.do_open();
            }
            if save_pressed {
                if save_as {
                    self.do_save_as();
                } else {
                    self.do_save();
                }
            }
        }

        if self.add_node_menu.is_none() {
            if let Some(dropped) = canvas_output.dropped_wire {
                self.add_node_menu = Some(add_node_menu::AddNodeMenu::open_for_dropped_wire(
                    dropped.screen_pos,
                    dropped.graph_pos,
                    add_node_menu::DropFilter {
                        port_type: dropped.from_port_type,
                        need_input: dropped.from_is_output,
                    },
                ));
                self.pending_wire_connect = Some(dropped);
            } else if let Some((screen_pos, graph_pos)) = canvas_output.open_add_menu_at {
                self.add_node_menu = Some(add_node_menu::AddNodeMenu::open(screen_pos, graph_pos));
                self.pending_wire_connect = None;
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
                self.canvas.selected.clear();
                self.canvas.selected.insert(core_id);
                if let Some(dropped) = self.pending_wire_connect.take() {
                    canvas::connect_dropped_wire(
                        &mut self.graph,
                        &mut self.history,
                        &dropped,
                        core_id,
                    );
                }
            }
        }

        export_dialog::show(
            ui.ctx(),
            &mut self.export_dialog,
            &mut self.gpu,
            &mut self.pool,
            &mut self.graph,
        );

        // egui/eframe only repaints in response to new OS input (mouse
        // move, click, key) — it does not automatically notice that a graph
        // edit this frame (connect, param change, add/remove node,
        // undo/redo) left a node dirty. Without this, the node this frame's
        // edit affects sits stale/blank until some unrelated input event
        // happens to trigger another frame; this guarantees the follow-up
        // frame that actually runs `evaluate()` on it.
        if self
            .graph
            .nodes()
            .any(|(id, _)| self.graph.needs_evaluation(id))
        {
            ui.ctx().request_repaint();
        }
    }

    /// A graceful shutdown means there's nothing left for the recovery
    /// snapshot to protect against — leaving it behind is what signals an
    /// *unclean* exit (crash, force-quit) to the next launch.
    fn on_exit(&mut self) {
        autosave::Autosave::clear();
    }
}

impl MaterialCraftApp {
    /// Swaps in a different graph wholesale (New/Open) — resets every piece
    /// of state that's keyed off the *previous* graph's nodes (history,
    /// selection, cached thumbnails/preview texture, the GPU texture pool),
    /// not just the graph field itself, so nothing stale lingers (a node ID
    /// from the old graph reused by the new one would otherwise find an
    /// orphaned texture already sitting in the pool under that key).
    fn replace_graph(&mut self, graph: mc_core::Graph, path: Option<std::path::PathBuf>) {
        self.graph = graph;
        self.current_path = path;
        self.history = mc_core::History::new();
        self.canvas = CanvasState::default();
        self.thumbnail_ids.clear();
        self.selected = None;
        self.add_node_menu = None;
        self.pending_wire_connect = None;
        self.preview_texture_id = None;
        self.pool = mc_gpu::TexturePool::default();
        // A fresh or just-opened graph has no unsaved history of its own
        // yet, so any leftover crash-recovery snapshot from a previous
        // session is now stale.
        autosave::Autosave::clear();
    }

    fn do_new(&mut self) {
        self.replace_graph(mc_core::Graph::new(), None);
    }

    fn do_open(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("MaterialCraft Graph", &["mcgraph"])
            .pick_file()
        else {
            return;
        };
        match mc_core::Graph::load_from_file(&path) {
            Ok(graph) => self.replace_graph(graph, Some(path)),
            Err(err) => {
                log::warn!("failed to open {}: {err}", path.display());
                self.toasts
                    .error(format!("Couldn't open {}: {err}", path.display()));
            }
        }
    }

    fn do_save(&mut self) {
        match self.current_path.clone() {
            Some(path) => self.save_to(path),
            None => self.do_save_as(),
        }
    }

    fn do_save_as(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("MaterialCraft Graph", &["mcgraph"])
            .set_file_name("untitled.mcgraph")
            .save_file()
        else {
            return;
        };
        self.save_to(path);
    }

    fn save_to(&mut self, path: std::path::PathBuf) {
        match self.graph.save_to_file(&path) {
            Ok(()) => {
                self.toasts.info(format!("Saved {}", path.display()));
                self.current_path = Some(path);
                // The file just written is now the authoritative copy, so
                // any crash-recovery snapshot from before this save is
                // stale.
                autosave::Autosave::clear();
            }
            Err(err) => {
                log::warn!("failed to save {}: {err}", path.display());
                self.toasts
                    .error(format!("Couldn't save {}: {err}", path.display()));
            }
        }
    }

    /// The selected node's resolved output texture, registered/updated as an
    /// egui texture for the 2D view.
    fn preview_texture(&mut self) -> Option<egui::TextureId> {
        let id = self.selected?;
        let key = mc_nodes::registry::resolve_texture_key(&self.graph, id)?;
        // Node textures are linear; egui never decodes what it's given, so
        // this view has to already be sRGB-encoded or the preview reads too
        // dark/off relative to the 3D viewport and an exported PNG — see
        // `display_encode.rs`.
        let view = crate::display_encode::encode_for_display(&mut self.gpu, &mut self.pool, &key)?;

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

/// Display order for the Node Library panel's category groups — the
/// `mc_nodes::defs` category strings, just in a deliberate reading order
/// rather than their incidental declaration order in that file.
const NODE_CATEGORY_ORDER: [&str; 9] = [
    "Generators/Noise",
    "Generators/Patterns",
    "Filters/Adjustments",
    "Filters/Blending & Masking",
    "Filters/Spatial",
    "Filters/Stylize",
    "Filters/Glitch",
    "Converters",
    "Output",
];

/// A short label for a category group header — the panel is narrow, so the
/// full "Filters/Blending & Masking"-style string wouldn't fit on one line.
fn category_short_name(category: &'static str) -> &'static str {
    match category {
        "Generators/Noise" => "Noise",
        "Generators/Patterns" => "Patterns",
        "Filters/Adjustments" => "Adjust",
        "Filters/Blending & Masking" => "Blend/Mask",
        "Filters/Spatial" => "Spatial",
        "Filters/Stylize" => "Stylize",
        "Filters/Glitch" => "Glitch",
        "Converters" => "Convert",
        "Output" => "Output",
        _ => category,
    }
}

/// The Node Library panel's content: every node, grouped by category and
/// color-coded to match (`theme::category_color`), shown as a Material
/// Icons Round glyph (`node_icons::icon_for_kind`) — clicking one adds that
/// node to the graph, the same transaction/selection flow the add-node menu
/// uses, just without a specific click position to place it at.
fn show_node_library(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    canvas: &mut canvas::CanvasState,
    selected: &mut Option<mc_core::NodeId>,
) {
    const ICON_SIZE: f32 = 32.0;
    const ICON_FONT_SIZE: f32 = 18.0;

    egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(12, 8))
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for &category in &NODE_CATEGORY_ORDER {
                        ui.add_space(6.0);
                        ui.colored_label(
                            theme::category_color(category),
                            category_short_name(category),
                        );
                        ui.add_space(2.0);
                        ui.horizontal_wrapped(|ui| {
                            for def in mc_nodes::defs::all_node_defs()
                                .iter()
                                .filter(|def| def.category == category)
                            {
                                let (rect, response) = ui.allocate_exact_size(
                                    egui::Vec2::splat(ICON_SIZE),
                                    egui::Sense::click(),
                                );
                                if ui.is_rect_visible(rect) {
                                    let bg_alpha = if response.hovered() { 0.55 } else { 0.3 };
                                    ui.painter().rect_filled(
                                        rect,
                                        4.0,
                                        theme::category_color(category).gamma_multiply(bg_alpha),
                                    );
                                    ui.painter().text(
                                        rect.center(),
                                        egui::Align2::CENTER_CENTER,
                                        node_icons::icon_for_kind(def.kind).to_string(),
                                        egui::FontId::new(
                                            ICON_FONT_SIZE,
                                            egui::FontFamily::Name("material_icons_round".into()),
                                        ),
                                        theme::SUPERNOVA_LIGHT,
                                    );
                                }
                                let response = response.on_hover_text(def.display_name);
                                if response.clicked() {
                                    add_node_from_library(graph, history, canvas, selected, def);
                                }
                            }
                        });
                    }
                    // Trailing breathing room so the last category's icons
                    // don't sit flush against the scroll area's own bottom
                    // edge.
                    ui.add_space(10.0);
                });
        });
}

fn add_node_from_library(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    canvas: &mut canvas::CanvasState,
    selected: &mut Option<mc_core::NodeId>,
    def: &'static mc_nodes::defs::NodeDef,
) {
    let node = def.instantiate();
    let (core_id, transaction) = graph.add_node_transaction(node);
    if history.do_transaction(graph, transaction).is_ok() {
        let drop_pos = canvas.last_click_pos();
        let _ = graph.set_node_position(core_id, [drop_pos.x, drop_pos.y]);
        *selected = Some(core_id);
        canvas.selected.clear();
        canvas.selected.insert(core_id);
    }
}

/// Replaces egui's default fonts with Geist (body text) and registers
/// Material Icons Round under its own named family, used explicitly wherever
/// an icon glyph is needed (see `viewport3d_panels.rs`) rather than mixed
/// into the body font — both bundled under `assets/fonts/`, see ASSETS.md.
fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    fonts.font_data.insert(
        "geist".to_owned(),
        egui::FontData::from_static(include_bytes!("../../../assets/fonts/Geist-Regular.ttf"))
            .into(),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "geist".to_owned());

    fonts.font_data.insert(
        "material_icons_round".to_owned(),
        egui::FontData::from_static(include_bytes!(
            "../../../assets/fonts/MaterialIconsRound-Regular.otf"
        ))
        .into(),
    );
    fonts.families.insert(
        egui::FontFamily::Name("material_icons_round".into()),
        vec!["material_icons_round".to_owned()],
    );

    ctx.set_fonts(fonts);
}

/// The app's window/taskbar icon: the bundled Material Icons Round "texture"
/// glyph (Apache-2.0, see ASSETS.md) on Neutron Star, in Supernova Light —
/// generated once from the brand palette and committed as a PNG rather than
/// rendered at startup, so this has zero runtime cost beyond decoding it.
fn app_icon() -> egui::IconData {
    let bytes = include_bytes!("../../../assets/icon/icon_256.png");
    let (rgba, width, height) =
        mc_export::decode_png_rgba8(bytes).expect("bundled app icon PNG is well-formed");
    egui::IconData {
        rgba,
        width,
        height,
    }
}

fn main() -> eframe::Result<()> {
    env_logger::init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("MaterialCraft")
            .with_icon(app_icon()),
        ..Default::default()
    };

    eframe::run_native(
        "MaterialCraft",
        native_options,
        Box::new(|cc| {
            theme::apply(&cc.egui_ctx);
            install_fonts(&cc.egui_ctx);

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
                canvas: CanvasState::default(),
                thumbnail_ids: HashMap::new(),
                selected: None,
                add_node_menu: None,
                pending_wire_connect: None,
                param_drag: None,
                export_dialog: export_dialog::ExportDialogState::default(),
                viewport3d,
                viewport_panels: viewport3d_panels::ViewportPanels::default(),
                dock: dock::DockLayout::default(),
                preview_texture_id: None,
                current_path: None,
                toasts: toast::Toasts::default(),
                last_eval_error: None,
                autosave: autosave::Autosave::default(),
                pending_recovery: autosave::recovery_path().exists(),
            }))
        }),
    )
}
