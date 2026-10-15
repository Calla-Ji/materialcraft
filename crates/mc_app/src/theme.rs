// SPDX-License-Identifier: MIT OR Apache-2.0
//! The app's color palette — adapted from the owner's personal brand
//! identity (astronomically-named colors) for a dark-mode tool, not a
//! literal copy of the brand guide's own light-mode usage: there, Supernova
//! Light is the main background and Pink Dwarf is the accent; here, Neutron
//! Star carries the big surfaces and Supernova Light is reserved for small
//! accent moments instead (separators, selection highlights, the app icon's
//! glyph). Pink Dwarf makes one appearance: Color-type ports/wires in the
//! node graph (`canvas.rs`'s `port_color`), replacing what used to be a
//! plain orange.

use egui::Color32;

/// Accent only: separators, selection highlights, active-state indicators,
/// the app icon's glyph. Never a large fill.
pub const SUPERNOVA_LIGHT: Color32 = Color32::from_rgb(0xFE, 0xE6, 0xEA);
/// Background of the Node Graph canvas and the 3D viewport — the two
/// "working surfaces" where the owner's own content (nodes, the rendered
/// material) is the subject, not the UI chrome around it.
pub const NEUTRON_STAR: Color32 = Color32::from_rgb(0x13, 0x17, 0x18);
/// Background for the rest of the UI chrome: menu bar, Parameters panel, 2D
/// View panel, node card fills.
pub const WORMHOLE_GREY: Color32 = Color32::from_rgb(0x4E, 0x54, 0x55);
/// Structural grey: the Node Graph's dot grid, node borders, low-contrast
/// dividers — matches the brand guide's own "dividers, input fields,
/// low-contrast indicators" role for this color.
pub const TWINKLING_GREY: Color32 = Color32::from_rgb(0xE0, 0xD1, 0xD4);
/// Color-type port pins and wires in the node graph — replaces the generic
/// orange those conventionally use elsewhere with a brand color instead.
pub const PINK_DWARF: Color32 = Color32::from_rgb(0xFF, 0x99, 0xD8);
/// Path-type port pins and wires (PATH_NODE_SPEC.md §1: "Path wires get
/// their own colour in the graph UI"). Not sourced from the owner's brand
/// guide, which has no entry for this — a clearly-distinct placeholder
/// (same "easy to swap for a real one later" status as `category_color`'s
/// own unassigned-category fallback below).
pub const PATH_CYAN: Color32 = Color32::from_rgb(0x6E, 0xE7, 0xE0);

/// One accent color per node category (`mc_nodes::defs`'s 10 category
/// strings), for the Node Library panel's grouping and each node card's
/// header-bar accent in the Node Graph canvas. An evenly hue-spaced
/// placeholder palette (not sourced from the owner's brand guide, which
/// doesn't define per-category colors) — easy to swap for real ones later.
/// `Generators/Path` is the one exception: rather than slot a 10th hue
/// into that wheel, it reuses `PATH_CYAN` directly — the same color
/// Path-type wires/ports already use everywhere else in the app, so the
/// category accent and "this is Path data" visual language line up
/// instead of introducing an unrelated second meaning for the same hue.
pub fn category_color(category: &str) -> Color32 {
    match category {
        "Generators/Noise" => Color32::from_rgb(0xD3, 0x68, 0x68),
        "Generators/Patterns" => Color32::from_rgb(0xD3, 0xAF, 0x68),
        "Generators/Path" => PATH_CYAN,
        "Filters/Adjustments" => Color32::from_rgb(0xAF, 0xD3, 0x68),
        "Filters/Blending & Masking" => Color32::from_rgb(0x68, 0xD3, 0x68),
        "Filters/Spatial" => Color32::from_rgb(0x68, 0xD3, 0xAF),
        "Filters/Stylize" => Color32::from_rgb(0x68, 0xAF, 0xD3),
        "Filters/Glitch" => Color32::from_rgb(0x68, 0x68, 0xD3),
        "Converters" => Color32::from_rgb(0xAF, 0x68, 0xD3),
        "Output" => Color32::from_rgb(0xD3, 0x68, 0xAF),
        _ => TWINKLING_GREY,
    }
}

/// `NEUTRON_STAR` as a linear `wgpu::Color`, for the 3D viewport's render
/// target clear value (a different color type than egui's `Color32`, same
/// source hex).
pub const NEUTRON_STAR_WGPU: wgpu::Color = wgpu::Color {
    r: 0x13 as f64 / 255.0,
    g: 0x17 as f64 / 255.0,
    b: 0x18 as f64 / 255.0,
    a: 1.0,
};

/// Applies the palette to egui's global style. Starts from `Visuals::dark()`
/// — only overriding what the palette actually has an opinion on, so
/// everything else (disabled-state dimming, etc.) keeps egui's own
/// well-tested dark-mode defaults.
pub fn apply(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();

    // "Rest of the UI mostly in Wormhole Grey" — the default panel/window
    // fill, which the Node Graph and 3D View dock cells override to Neutron
    // Star individually (`dock.rs`'s `draw_leaf`).
    visuals.panel_fill = WORMHOLE_GREY;
    visuals.window_fill = WORMHOLE_GREY;

    // Separation lines (dock dividers, cell borders) in Supernova Light —
    // thin, since it's an accent, not a fill.
    visuals.window_stroke = egui::Stroke::new(1.0, SUPERNOVA_LIGHT);

    // Selection highlight (e.g. selected text) — the one place this app's
    // dark theme uses Supernova Light the way the brand guide uses Pink
    // Dwarf: "reserved for interactive elements, active states."
    visuals.selection.bg_fill = SUPERNOVA_LIGHT;
    visuals.selection.stroke = egui::Stroke::new(1.0, NEUTRON_STAR);
    visuals.hyperlink_color = SUPERNOVA_LIGHT;

    // Text, everywhere, in Supernova Light — plain labels and normal
    // (not-currently-pressed) button text share `noninteractive`/`inactive`
    // in egui's style model, so both need setting. `hovered` too, so text
    // doesn't flicker to a different color just from the mouse passing
    // over it. Weak/hint text (`ui.weak(...)`, e.g. "select a node...")
    // keeps the same hue, just dimmer, rather than switching to a
    // different color family for "muted."
    let text_stroke = egui::Stroke::new(1.0, SUPERNOVA_LIGHT);
    visuals.widgets.noninteractive.fg_stroke = text_stroke;
    visuals.widgets.inactive.fg_stroke = text_stroke;
    visuals.widgets.hovered.fg_stroke = text_stroke;
    visuals.weak_text_color = Some(SUPERNOVA_LIGHT.gamma_multiply(0.6));

    // A clicked/pressed button inverts: Supernova Light fill, Neutron Star
    // text — egui's `active` widget state, which also happens to be what
    // `Style::strong_text_color()`/`RichText::strong()` reads from for
    // *non-button* emphasis text (e.g. the menu bar's app name, dock panel
    // titles). Those two uses would fight over the same field, so this app
    // doesn't use `.strong()` for emphasis anywhere — every "strong" text
    // element sets its `Supernova Light` color explicitly instead (see
    // `main.rs`'s app-name label, `dock.rs`'s panel title) precisely so
    // this can stay dedicated to real button-press feedback.
    visuals.widgets.active.bg_fill = SUPERNOVA_LIGHT;
    visuals.widgets.active.weak_bg_fill = SUPERNOVA_LIGHT;
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, NEUTRON_STAR);

    ctx.set_visuals(visuals);
}
