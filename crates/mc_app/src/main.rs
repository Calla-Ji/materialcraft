//! mc_app: the desktop application shell.
//!
//! Owns the window and the overall panel layout (menu, 2D/3D previews, node
//! graph, parameter panel). M0 only lays out empty placeholders; real content
//! (the node editor, GPU-backed previews, etc.) arrives in later milestones.

/// Root application state for the eframe event loop.
struct MaterialCraftApp;

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
                ui.label("Export…");
                ui.label("Undo");
                ui.label("Redo");
                ui.label("Examples");
            });
        });

        egui::Panel::right("parameters")
            .default_size(260.0)
            .show(ui, |ui| {
                ui.heading("Parameters");
                ui.weak("(select a node to edit its parameters)");
            });

        egui::Panel::top("previews")
            .default_size(260.0)
            .resizable(true)
            .show(ui, |ui| {
                ui.columns(2, |columns| {
                    columns[0].heading("2D View");
                    columns[0].weak("(selected node's output, full resolution)");
                    columns[1].heading("3D View");
                    columns[1].weak("(sphere / cube / plane, PBR lit)");
                });
            });

        egui::CentralPanel::default().show(ui, |ui| {
            ui.heading("Node Graph");
            ui.weak("(right-click to add a node)");
        });
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
            Ok(Box::new(MaterialCraftApp))
        }),
    )
}
