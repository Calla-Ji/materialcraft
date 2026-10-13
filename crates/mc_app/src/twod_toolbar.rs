// SPDX-License-Identifier: MIT OR Apache-2.0
//! The 2D View's own left-edge icon rail — Tiling on top, a thin divider,
//! then Pen (greyed out with an explanatory tooltip when no Path node is
//! selected) — mirroring `viewport3d_panels`' own icon-button toolbar so
//! the 2D and 3D panels read as one consistent app rather than two
//! different UI styles. Room is left here for future Path-editor actions
//! (SVG import, etc.) to join Pen in the same rail.

const TOOLBAR_WIDTH: f32 = 38.0;
const BUTTON_SIZE: f32 = 30.0;
const ICON_FONT_SIZE: f32 = 20.0;
const TOP_PADDING: f32 = 8.0;

const GRID_3X3_GLYPH: char = '\u{f015}';
const EDIT_GLYPH: char = '\u{e3c9}'; // "edit" (pencil)

/// Draws the rail into a fixed-width column at the left edge of
/// `full_rect`, with a vertical divider matching the 3D viewport's own,
/// and returns the rect the 2D View's actual content should use instead of
/// `full_rect` — same split as `dock::CrateId::ThreeDView`'s own
/// toolbar/viewport division in `main.rs`.
#[allow(clippy::too_many_arguments)]
pub fn show(
    ui: &mut egui::Ui,
    full_rect: egui::Rect,
    tiling_preview: &mut bool,
    path_editor: &mut crate::path_editor::PathEditorState,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    selected: Option<mc_core::NodeId>,
) -> egui::Rect {
    let toolbar_rect =
        egui::Rect::from_min_size(full_rect.min, egui::vec2(TOOLBAR_WIDTH, full_rect.height()));
    let divider_stroke = ui.visuals().window_stroke();

    // The rail had no background fill of its own before — whatever was
    // directly behind it (even a stray point/handle dragged underneath,
    // see `path_editor`'s own drag clamping) showed straight through,
    // reading as a gap rather than a proper opaque panel strip like every
    // other toolbar in the app.
    ui.painter()
        .rect_filled(toolbar_rect, 0.0, crate::theme::NEUTRON_STAR);

    {
        let mut toolbar_ui = ui.new_child(egui::UiBuilder::new().max_rect(toolbar_rect));
        toolbar_ui.set_clip_rect(toolbar_rect);
        toolbar_ui.vertical(|ui| {
            ui.add_space(TOP_PADDING);
            tiling_button(ui, tiling_preview);
            // `ui.separator()` draws a horizontal rule inset from both
            // edges inside a vertical layout — the exact same widget, and
            // so the exact same look, as the menu bar's own separators
            // between e.g. Export and Undo.
            ui.separator();

            let path_node_id =
                selected.filter(|&id| graph.node(id).is_some_and(|n| n.kind == "path"));
            pen_button(ui, path_editor, graph, history, path_node_id);
        });
    }
    ui.painter()
        .vline(toolbar_rect.right(), toolbar_rect.y_range(), divider_stroke);

    egui::Rect::from_min_max(
        egui::pos2(toolbar_rect.right() + 1.0, full_rect.top()),
        full_rect.max,
    )
}

/// Square icon button, drawn from raw primitives (not `ui.selectable_label`)
/// so every button is identically sized regardless of its glyph's own
/// advance width — same reasoning as `viewport3d_panels::icon_button`.
fn icon_button(
    ui: &mut egui::Ui,
    id_source: &str,
    glyph: char,
    selected: bool,
    enabled: bool,
    tooltip: &str,
) -> egui::Response {
    let x = ui.cursor().min.x + ((ui.available_width() - BUTTON_SIZE) / 2.0).max(0.0);
    let rect = egui::Rect::from_min_size(
        egui::pos2(x, ui.cursor().min.y),
        egui::vec2(BUTTON_SIZE, BUTTON_SIZE),
    );
    let sense = if enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let response = ui.interact(rect, ui.id().with(id_source), sense);
    ui.advance_cursor_after_rect(rect);

    let visuals = ui.style().interact_selectable(&response, selected);
    let painter = ui.painter();
    if selected || response.hovered() {
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
        glyph.to_string(),
        egui::FontId::new(
            ICON_FONT_SIZE,
            egui::FontFamily::Name("material_icons_round".into()),
        ),
        color,
    );

    response.on_hover_text(tooltip)
}

fn tiling_button(ui: &mut egui::Ui, tiling_preview: &mut bool) {
    let response = icon_button(
        ui,
        "two_d_toolbar_tiling",
        GRID_3X3_GLYPH,
        *tiling_preview,
        true,
        "Show the pattern tiled across the whole view, to check how it seams.",
    );
    if response.clicked() {
        *tiling_preview = !*tiling_preview;
    }
}

fn pen_button(
    ui: &mut egui::Ui,
    path_editor: &mut crate::path_editor::PathEditorState,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    path_node_id: Option<mc_core::NodeId>,
) {
    let enabled = path_node_id.is_some();
    let tooltip = if enabled {
        "Draw new points on this curve — click empty space to append one, drag to pull out a \
         handle, click the first point again to close. P also toggles this while hovering the \
         view, Esc leaves it."
    } else {
        "Add a Path node and select it to draw on its curve."
    };
    let response = icon_button(
        ui,
        "two_d_toolbar_pen",
        EDIT_GLYPH,
        path_editor.pen_mode(),
        enabled,
        tooltip,
    );
    if let Some(id) = path_node_id {
        if response.clicked() {
            path_editor.toggle_pen_mode(graph, history, id);
        }
    }
}
