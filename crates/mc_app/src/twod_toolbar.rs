// SPDX-License-Identifier: MIT OR Apache-2.0
//! The 2D View's own left-edge icon rail — Tiling on top, a thin divider,
//! then Pen, Insert Point, Join, SVG Import and Paste (each greyed out with
//! an explanatory tooltip when they don't currently apply), then a second
//! divider and the RGB/R/G/B/A channel-view buttons (CLAUDE.md §9) —
//! mirroring `viewport3d_panels`' own icon-button toolbar so the 2D and 3D
//! panels read as one consistent app rather than two different UI styles.
//! Every glyph's codepoint below is confirmed against Google's own
//! `MaterialIconsRound-Regular.codepoints` file (shared across every style
//! variant — Filled/Outlined/Round/Sharp/Two-tone — only the drawn glyph
//! differs, so these render correctly against the bundled Round font)
//! rather than guessed, since a wrong one draws as a blank/tofu glyph.

const TOOLBAR_WIDTH: f32 = 38.0;
const BUTTON_SIZE: f32 = 30.0;
const ICON_FONT_SIZE: f32 = 20.0;
const TEXT_BUTTON_FONT_SIZE: f32 = 11.0;
const TOP_PADDING: f32 = 8.0;

const GRID_3X3_GLYPH: char = '\u{f015}';
// Codepoints confirmed against Google's own
// MaterialIcons-Regular.codepoints file — shared across every style
// variant (Filled/Outlined/Round/Sharp/Two-tone), only the drawn glyph
// differs, so these render correctly against the bundled Round font.
const ROUTE_GLYPH: char = '\u{eacd}'; // "route" — the owner's own pick over a plain pencil
const JOIN_GLYPH: char = '\u{eaeb}'; // "join_full"
const ADD_CIRCLE_GLYPH: char = '\u{e147}'; // "add_circle"

// Same file, confirmed again for these two — the owner's own pick from
// https://fonts.google.com/icons (Material Icons Round).
const UPLOAD_FILE_GLYPH: char = '\u{e9fc}'; // "upload_file", for SVG Import
const DATA_OBJECT_GLYPH: char = '\u{ead3}'; // "data_object", for Paste

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
    channel_view: &mut crate::display_encode::ChannelView,
    path_editor: &mut crate::path_editor::PathEditorState,
    svg_import_dialog: &mut crate::svg_import_dialog::SvgImportDialogState,
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
        // A fixed-width column with a growing number of buttons (Pen/Join/
        // Insert Point/SVG Import/Paste, then the channel-view buttons) can
        // run taller than a short docked 2D View panel — previously the
        // overflow was just clipped by `set_clip_rect` above, making those
        // buttons exist but be permanently unreachable. Scrolling instead
        // keeps every button reachable regardless of panel height.
        egui::ScrollArea::vertical()
            .id_salt("two_d_toolbar_scroll")
            .show(&mut toolbar_ui, |ui| {
                ui.vertical(|ui| {
                    ui.add_space(TOP_PADDING);
                    tiling_button(ui, tiling_preview);
                    // `ui.separator()` draws a horizontal rule inset from
                    // both edges inside a vertical layout — the exact same
                    // widget, and so the exact same look, as the menu bar's
                    // own separators between e.g. Export and Undo.
                    ui.separator();

                    let path_node_id =
                        selected.filter(|&id| graph.node(id).is_some_and(|n| n.kind == "path"));
                    pen_button(ui, path_editor, graph, history, path_node_id);
                    join_button(ui, path_editor, graph, history, path_node_id);
                    insert_point_button(ui, path_editor, path_node_id);
                    import_svg_button(ui, svg_import_dialog, path_node_id);
                    paste_button(ui, svg_import_dialog, path_node_id);

                    ui.separator();
                    channel_view_buttons(ui, channel_view);
                });
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

/// Same square-button layout/highlight logic as [`icon_button`], but with a
/// short text label instead of a Material Icons glyph — there's no single
/// glyph that reads as "just the R channel," so these spell it out.
fn text_button(
    ui: &mut egui::Ui,
    id_source: &str,
    label: &str,
    selected: bool,
    enabled: bool,
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
    if selected || (enabled && response.hovered()) {
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
        label,
        egui::FontId::proportional(TEXT_BUTTON_FONT_SIZE),
        color,
    );
    response
}

fn channel_view_buttons(ui: &mut egui::Ui, channel_view: &mut crate::display_encode::ChannelView) {
    for choice in crate::display_encode::ChannelView::ALL {
        let tooltip = match choice {
            crate::display_encode::ChannelView::Rgb => {
                "Show all channels combined, as normal.".to_string()
            }
            other => format!(
                "Show just the {} channel, broadcast to grey — the only way to judge a \
                 packed channel (e.g. one channel of an ORM texture) on its own, since the \
                 combined RGB view of unrelated channels isn't meant to look like a sensible \
                 color.",
                other.label()
            ),
        };
        let response = text_button(
            ui,
            &format!("two_d_toolbar_channel_{}", choice.label()),
            choice.label(),
            *channel_view == choice,
            true,
        )
        .on_hover_text(tooltip);
        if response.clicked() {
            *channel_view = choice;
        }
    }
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
        ROUTE_GLYPH,
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

/// "Glows" (the same selected-style highlight the Pen/Tiling buttons use
/// once active) the moment the current selection is a valid pair of
/// endpoints to join — more immediate than only offering this in the
/// Parameters panel, per the owner's own preference.
fn join_button(
    ui: &mut egui::Ui,
    path_editor: &mut crate::path_editor::PathEditorState,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    path_node_id: Option<mc_core::NodeId>,
) {
    let can_join = path_node_id.is_some_and(|id| path_editor.can_join(graph, id));
    let tooltip = if can_join {
        "Join the two selected endpoints with a new segment."
    } else {
        "Select two open-subpath endpoints (not already connected to anything on \
         that side) to join them."
    };
    let response = icon_button(
        ui,
        "two_d_toolbar_join",
        JOIN_GLYPH,
        can_join,
        can_join,
        tooltip,
    );
    if let Some(id) = path_node_id {
        if can_join && response.clicked() {
            path_editor.join_selected(graph, history, id);
        }
    }
}

/// A discoverability hint, not a toggle — clicking any segment of a Path
/// node's curve (outside Pen mode) always inserts a point there
/// (`path_editor::segment_hit_test`), with no mode to switch into first.
/// This icon just makes that fact visible; it has no click handler of its
/// own.
fn insert_point_button(
    ui: &mut egui::Ui,
    path_editor: &mut crate::path_editor::PathEditorState,
    path_node_id: Option<mc_core::NodeId>,
) {
    let enabled = path_node_id.is_some();
    let active = path_editor.insert_mode();
    let tooltip = if enabled {
        "Click a point on this curve's own line to insert a new point there. Click this icon \
         again to turn it off."
    } else {
        "Add a Path node and select it — then turn this on to insert points by clicking the \
         curve."
    };
    let response = icon_button(
        ui,
        "two_d_toolbar_insert_point",
        ADD_CIRCLE_GLYPH,
        active,
        enabled,
        tooltip,
    );
    if enabled && response.clicked() {
        path_editor.toggle_insert_mode();
    }
}

/// Opens the file picker immediately (same pattern as `main.rs`'s own
/// Open/Save As) — the SVG Import dialog itself only opens once a file's
/// actually been picked, so cancelling the picker is a no-op, not an
/// empty dialog.
fn import_svg_button(
    ui: &mut egui::Ui,
    svg_import_dialog: &mut crate::svg_import_dialog::SvgImportDialogState,
    path_node_id: Option<mc_core::NodeId>,
) {
    let enabled = path_node_id.is_some();
    let tooltip = if enabled {
        "Import an SVG file onto this curve."
    } else {
        "Add a Path node and select it to import an SVG onto it."
    };
    let response = icon_button(
        ui,
        "two_d_toolbar_import_svg",
        UPLOAD_FILE_GLYPH,
        false,
        enabled,
        tooltip,
    );
    if enabled && response.clicked() {
        if let Some(id) = path_node_id {
            svg_import_dialog.open_for_file_pick(id);
        }
    }
}

/// Opens the dialog directly, with an empty text box for the owner to
/// paste a whole `<svg>` or a bare `d` string into (plain OS paste into an
/// `egui::TextEdit` — no separate clipboard-reading dependency needed).
fn paste_button(
    ui: &mut egui::Ui,
    svg_import_dialog: &mut crate::svg_import_dialog::SvgImportDialogState,
    path_node_id: Option<mc_core::NodeId>,
) {
    let enabled = path_node_id.is_some();
    let tooltip = if enabled {
        "Paste an SVG or a path's own \"d\" string onto this curve."
    } else {
        "Add a Path node and select it to paste an SVG onto it."
    };
    let response = icon_button(
        ui,
        "two_d_toolbar_paste_svg",
        DATA_OBJECT_GLYPH,
        false,
        enabled,
        tooltip,
    );
    if enabled && response.clicked() {
        if let Some(id) = path_node_id {
            svg_import_dialog.open_for_paste(id);
        }
    }
}
