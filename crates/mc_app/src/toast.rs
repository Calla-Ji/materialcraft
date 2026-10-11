//! A small stack of auto-dismissing notification cards, anchored to the
//! window's bottom-right corner — CLAUDE.md's M6 "error toasts" item. Before
//! this existed, every fallible path in the app (failed Open/Save, a graph
//! evaluation error) only reached `log::warn!`, visible solely in the
//! terminal the app was launched from.

use std::time::{Duration, Instant};

enum ToastKind {
    Info,
    Error,
}

impl ToastKind {
    /// Material Icons Round glyphs, codepoints verified against the font's
    /// own published `.codepoints` file (same source as `node_icons.rs`).
    fn glyph(&self) -> char {
        match self {
            ToastKind::Info => '\u{e88e}',  // info
            ToastKind::Error => '\u{e000}', // error
        }
    }
}

struct Toast {
    message: String,
    kind: ToastKind,
    shown_at: Instant,
}

const LIFETIME: Duration = Duration::from_secs(6);
const WIDTH: f32 = 280.0;

/// Owned by the app; call [`Toasts::show`] once per frame regardless of
/// whether anything's queued.
#[derive(Default)]
pub struct Toasts {
    queue: Vec<Toast>,
}

impl Toasts {
    pub fn error(&mut self, message: impl Into<String>) {
        self.queue.push(Toast {
            message: message.into(),
            kind: ToastKind::Error,
            shown_at: Instant::now(),
        });
    }

    pub fn info(&mut self, message: impl Into<String>) {
        self.queue.push(Toast {
            message: message.into(),
            kind: ToastKind::Info,
            shown_at: Instant::now(),
        });
    }

    /// Draws every live toast stacked above the bottom-right corner, oldest
    /// on top, dropping any that have outlived `LIFETIME`.
    pub fn show(&mut self, ctx: &egui::Context) {
        self.queue.retain(|t| t.shown_at.elapsed() < LIFETIME);
        if self.queue.is_empty() {
            return;
        }
        // These age out on their own timer rather than in response to input,
        // so without this a toast could sit on screen well past its
        // lifetime until some unrelated event happened to trigger a repaint.
        ctx.request_repaint_after(Duration::from_millis(200));

        egui::Area::new(egui::Id::new("materialcreator_toasts"))
            .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-16.0, -16.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    for toast in &self.queue {
                        egui::Frame::popup(ui.style())
                            .fill(crate::theme::NEUTRON_STAR)
                            .show(ui, |ui| {
                                ui.set_max_width(WIDTH);
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(toast.kind.glyph().to_string())
                                            .family(egui::FontFamily::Name(
                                                "material_icons_round".into(),
                                            ))
                                            .color(crate::theme::SUPERNOVA_LIGHT),
                                    );
                                    ui.label(
                                        egui::RichText::new(&toast.message)
                                            .color(crate::theme::SUPERNOVA_LIGHT),
                                    );
                                });
                            });
                        ui.add_space(6.0);
                    }
                });
            });
    }
}
