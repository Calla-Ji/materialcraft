// SPDX-License-Identifier: MIT OR Apache-2.0
//! Icon-button toolbar + slide-out side panel for the 3D viewport, Blender
//! N-panel style: click an icon to slide a panel out from the right edge of
//! the viewport image; click it again (or a different icon) to switch or
//! close it. This module only owns which panel is open and the slide
//! animation — the actual panel contents are drawn by the caller
//! (`main.rs`), which already holds `&mut self.viewport3d` and would
//! otherwise have to fight the borrow checker to pass it through here too.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelKind {
    Mesh,
    Material,
    Camera,
    Environment,
}

/// Glyph codepoints from the bundled Material Icons Round font (see
/// ASSETS.md) — `view_in_ar`, `texture`, `photo_camera`, `panorama`.
impl PanelKind {
    fn glyph(self) -> char {
        match self {
            PanelKind::Mesh => '\u{e9fe}',
            PanelKind::Material => '\u{e421}',
            PanelKind::Camera => '\u{e412}',
            PanelKind::Environment => '\u{e40b}',
        }
    }

    fn tooltip(self) -> &'static str {
        match self {
            PanelKind::Mesh => "Mesh",
            PanelKind::Material => "Material",
            PanelKind::Camera => "Camera (coming soon)",
            PanelKind::Environment => "Environment",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            PanelKind::Mesh => "Mesh",
            PanelKind::Material => "Material",
            PanelKind::Camera => "Camera",
            PanelKind::Environment => "Environment",
        }
    }

    /// A one-line explanation shown at the top of the panel itself, above
    /// its controls — not just in the toolbar icon's hover tooltip, so the
    /// description and the functions it describes read as one panel instead
    /// of two separate pieces of UI.
    pub fn description(self) -> &'static str {
        match self {
            PanelKind::Mesh => "Choose the preview shape, or plug in your own mesh.",
            PanelKind::Material => "Fine-tune surface height and tessellation detail.",
            PanelKind::Camera => "Coming soon: render resolution and engine options.",
            PanelKind::Environment => "Pick a bundled HDRI, or load your own.",
        }
    }
}

pub const PANEL_WIDTH: f32 = 220.0;
const ANIM_SPEED: f32 = PANEL_WIDTH * 6.0; // px/second

/// Which panels have real content yet — the rest still get a toolbar icon
/// (so the final four-icon layout is visible now) but are disabled rather
/// than opening onto an empty drawer.
const ENABLED: [PanelKind; 3] = [PanelKind::Mesh, PanelKind::Material, PanelKind::Environment];

pub struct ViewportPanels {
    open: Option<PanelKind>,
    /// The panel to show in the drawer even while it's animating closed
    /// (once `open` goes back to `None`) — otherwise the drawer would go
    /// blank a frame before it finishes sliding away.
    last_kind: PanelKind,
    width: f32,
}

impl Default for ViewportPanels {
    fn default() -> Self {
        Self {
            open: None,
            last_kind: PanelKind::Mesh,
            width: 0.0,
        }
    }
}

const BUTTON_SIZE: f32 = 30.0;
const BUTTON_GAP: f32 = 4.0;
const ICON_FONT_SIZE: f32 = 20.0;
const TOP_PADDING: f32 = 8.0;

impl ViewportPanels {
    /// The icon rail; place to the left of the viewport image, inside a
    /// narrow fixed-width column so it doesn't grow/shrink with content.
    pub fn toolbar_vertical(&mut self, ui: &mut egui::Ui) {
        // Explicit, rather than relying on the caller's own layout: this is
        // nested inside the 3D View's `ui.horizontal(..)`, and a freshly
        // created child `Ui` inherits its *parent's* layout direction when
        // none is set — without forcing vertical here, the icons stacked
        // left-to-right instead of top-to-bottom and all but the first got
        // clipped out of the narrow toolbar column.
        ui.vertical(|ui| {
            ui.add_space(TOP_PADDING);
            for (i, kind) in [
                PanelKind::Mesh,
                PanelKind::Material,
                PanelKind::Camera,
                PanelKind::Environment,
            ]
            .into_iter()
            .enumerate()
            {
                if i > 0 {
                    ui.add_space(BUTTON_GAP);
                }
                self.icon_button(ui, kind);
            }
        });
    }

    /// A fixed-size square button drawn from raw primitives rather than
    /// `ui.selectable_label` — the Material Icons glyphs used here each have
    /// a different natural advance width, so sizing the button from the text
    /// widget itself made the four buttons inconsistent sizes. Owning the
    /// rect ourselves keeps every button identical regardless of which
    /// glyph it holds.
    fn icon_button(&mut self, ui: &mut egui::Ui, kind: PanelKind) {
        let x = ui.cursor().min.x + ((ui.available_width() - BUTTON_SIZE) / 2.0).max(0.0);
        let rect = egui::Rect::from_min_size(
            egui::pos2(x, ui.cursor().min.y),
            egui::vec2(BUTTON_SIZE, BUTTON_SIZE),
        );
        let enabled = ENABLED.contains(&kind);
        let id = ui.id().with(("viewport_panel_icon", kind as u8));
        let sense = if enabled {
            egui::Sense::click()
        } else {
            egui::Sense::hover()
        };
        let response = ui.interact(rect, id, sense);
        ui.advance_cursor_after_rect(rect);

        let is_open = self.open == Some(kind);
        let visuals = ui.style().interact_selectable(&response, is_open);
        let painter = ui.painter();
        if is_open || response.hovered() {
            painter.rect_filled(rect, 4.0, visuals.bg_fill);
        }
        let color = if enabled {
            visuals.fg_stroke.color
        } else {
            ui.visuals().weak_text_color()
        };
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            kind.glyph().to_string(),
            egui::FontId::new(
                ICON_FONT_SIZE,
                egui::FontFamily::Name("material_icons_round".into()),
            ),
            color,
        );

        let response = response.on_hover_text(kind.tooltip());
        if enabled && response.clicked() {
            self.open = if is_open { None } else { Some(kind) };
            if let Some(opened) = self.open {
                self.last_kind = opened;
            }
        }
    }

    /// Advances the slide animation by one frame — call this every frame
    /// regardless of whether a panel is open, so a just-closed panel keeps
    /// sliding shut instead of vanishing instantly.
    pub fn update_anim(&mut self, ctx: &egui::Context) {
        let target = if self.open.is_some() {
            PANEL_WIDTH
        } else {
            0.0
        };
        let step = ANIM_SPEED * ctx.input(|i| i.stable_dt).min(0.1);
        if self.width < target {
            self.width = (self.width + step).min(target);
        } else if self.width > target {
            self.width = (self.width - step).max(target);
        }
        if (self.width - target).abs() > 0.5 {
            ctx.request_repaint();
        }
    }

    /// Current drawer width in pixels, including while animating open/shut.
    /// Zero (or near it) means nothing needs to be drawn at all.
    pub fn current_width(&self) -> f32 {
        self.width
    }

    /// Which panel's content to draw — the open one, or (while closing) the
    /// last one that was open.
    pub fn visible_kind(&self) -> PanelKind {
        self.open.unwrap_or(self.last_kind)
    }
}
