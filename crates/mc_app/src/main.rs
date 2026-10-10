//! mc_app: the desktop application shell.
//!
//! Owns the window and the overall panel layout (menu, 2D/3D previews, node
//! graph, parameter panel). M1 adds the first real content: a single
//! hardcoded Perlin Noise node, dispatched on the GPU and shown in the 2D
//! view. There's no graph model or dirty-tracking yet (that's mc_core, M2),
//! so for now the node is simply re-evaluated every frame.

const PREVIEW_RESOLUTION: u32 = 512;
const PREVIEW_KEY: &str = "preview";

/// Root application state for the eframe event loop.
struct MaterialCraftApp {
    gpu: mc_gpu::Context,
    pool: mc_gpu::TexturePool,
    render_state: egui_wgpu::RenderState,
    perlin_params: mc_nodes::perlin::PerlinParams,
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
                ui.label("Perlin Noise");
                ui.add(egui::Slider::new(&mut self.perlin_params.scale, 1..=32).text("Scale"));
                ui.add(egui::Slider::new(&mut self.perlin_params.octaves, 1..=8).text("Octaves"));
                ui.add(
                    egui::Slider::new(&mut self.perlin_params.persistence, 0.1..=1.0)
                        .text("Persistence"),
                );
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.perlin_params.seed).prefix("Seed: "));
                    if ui.button("🎲").clicked() {
                        self.perlin_params.seed = self.perlin_params.seed.wrapping_add(1);
                    }
                });
            });

        // Re-evaluate the node and refresh its preview texture every frame.
        // Caching/dirty-tracking so only *changed* nodes re-run is mc_core's
        // job (M2) — for this single hardcoded node it's not needed yet.
        mc_nodes::perlin::dispatch(
            &mut self.gpu,
            &mut self.pool,
            PREVIEW_KEY,
            PREVIEW_RESOLUTION,
            &self.perlin_params,
        );
        let view = self.pool.view(PREVIEW_KEY).expect("just dispatched above");
        let texture_id = match self.preview_texture_id {
            Some(id) => {
                self.render_state
                    .renderer
                    .write()
                    .update_egui_texture_from_wgpu_texture(
                        &self.gpu.device,
                        view,
                        wgpu::FilterMode::Linear,
                        id,
                    );
                id
            }
            None => {
                let id = self.render_state.renderer.write().register_native_texture(
                    &self.gpu.device,
                    view,
                    wgpu::FilterMode::Linear,
                );
                self.preview_texture_id = Some(id);
                id
            }
        };

        egui::Panel::top("previews")
            .default_size(260.0)
            .resizable(true)
            .show(ui, |ui| {
                ui.columns(2, |columns| {
                    columns[0].heading("2D View");
                    let preview_size = columns[0].available_height().min(240.0);
                    columns[0].image(egui::load::SizedTexture::new(
                        texture_id,
                        egui::Vec2::splat(preview_size),
                    ));
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

            let render_state = cc
                .wgpu_render_state
                .clone()
                .expect("mc_app requires eframe's wgpu backend");
            let gpu = mc_gpu::Context::new(render_state.device.clone(), render_state.queue.clone());

            Ok(Box::new(MaterialCraftApp {
                gpu,
                pool: mc_gpu::TexturePool::default(),
                render_state,
                perlin_params: mc_nodes::perlin::PerlinParams::default(),
                preview_texture_id: None,
            }))
        }),
    )
}
