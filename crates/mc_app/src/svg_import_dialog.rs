// SPDX-License-Identifier: MIT OR Apache-2.0
//! The SVG Import dialog — "Import SVG…" (file) and "Paste" (a whole
//! `<svg>` or a bare `d` string) for the Path node, per
//! `SVG_IMPORT_SPEC.md`'s own "Node UI" section: fit (bbox/viewbox/none),
//! margin, wrap, and Replace-vs-Append, with the warnings list shown after
//! import. Opened by the 2D View toolbar's own SVG/Paste buttons
//! (`twod_toolbar.rs`), one per `mc_core::NodeId` at a time.
//!
//! "Paste" reads nothing from the OS clipboard itself — the dialog just
//! opens with an empty, focused `egui::TextEdit`, and the owner pastes
//! into it the normal OS way (Cmd+V), exactly like pasting into any other
//! text field in this app. That avoids a dedicated clipboard-reading
//! dependency entirely for what the spec only ever asks to *feed into the
//! importer*, not to query programmatically.
//!
//! Styled via `dialog_chrome::show` — the shared custom header/frame every
//! popup in this app uses — so this reads as part of the app rather than
//! a generic system window (the owner's own feedback on the previous
//! default-chrome version). The paste box is capped at
//! [`PASTE_BOX_MAX_HEIGHT`] inside a scroll area rather than left to grow
//! with the pasted text: `egui::TextEdit::multiline` always sizes to fit
//! every line of its content (it doesn't scroll internally on its own),
//! so a long paste used to grow the whole (unresizable, center-anchored)
//! window symmetrically off the top and bottom of the screen — tall
//! enough that its own header/close button scrolled out of view with it.

use mc_core::ParamValue;

/// Caps the paste box's own height — see this module's top doc comment for
/// why an uncapped `TextEdit::multiline` was growing the whole window
/// off-screen.
const PASTE_BOX_MAX_HEIGHT: f32 = 220.0;

pub struct SvgImportDialogState {
    pub open: bool,
    target_node: Option<mc_core::NodeId>,
    /// `Some(name)` when the source came from a picked file (shown
    /// read-only); `None` when pasting (an editable text box instead).
    file_name: Option<String>,
    source_text: String,
    fit: mc_nodes::svg_import::Fit,
    margin: f32,
    wrap: bool,
    append: bool,
    last_result: Option<mc_nodes::svg_import::ImportResult>,
}

impl Default for SvgImportDialogState {
    fn default() -> Self {
        Self {
            open: false,
            target_node: None,
            file_name: None,
            source_text: String::new(),
            fit: mc_nodes::svg_import::Fit::Bbox,
            margin: 0.05,
            wrap: true,
            append: false,
            last_result: None,
        }
    }
}

impl SvgImportDialogState {
    /// "Import SVG…": opens the file picker right away (same pattern as
    /// `main.rs`'s own Open/Save As) — cancelling it is a no-op, the
    /// dialog itself never opens empty.
    pub fn open_for_file_pick(&mut self, node_id: mc_core::NodeId) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("SVG", &["svg"])
            .pick_file()
        else {
            return;
        };
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(err) => {
                log::warn!("failed to read {}: {err}", path.display());
                return;
            }
        };
        *self = Self {
            open: true,
            target_node: Some(node_id),
            file_name: Some(
                path.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| path.display().to_string()),
            ),
            source_text: text,
            ..Self::default()
        };
    }

    /// "Paste": opens directly with an empty box to paste into.
    pub fn open_for_paste(&mut self, node_id: mc_core::NodeId) {
        *self = Self {
            open: true,
            target_node: Some(node_id),
            ..Self::default()
        };
    }
}

/// Runs the importer on `state.source_text` with the dialog's current
/// options. A bare `d` string (the common paste case per
/// `SVG_IMPORT_SPEC.md`'s own case 12) is told apart from a whole `<svg>`
/// the same way a human would glance at it: does it start with `<`.
fn run_import(state: &SvgImportDialogState) -> mc_nodes::svg_import::ImportResult {
    let text = state.source_text.trim();
    if text.starts_with('<') {
        mc_nodes::svg_import::import_svg(
            &state.source_text,
            state.fit,
            f64::from(state.margin),
            state.wrap,
        )
    } else {
        mc_nodes::svg_import::import_d(text, state.fit, f64::from(state.margin), state.wrap)
    }
}

pub fn show(
    ctx: &egui::Context,
    state: &mut SvgImportDialogState,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
) {
    if !state.open {
        return;
    }
    let Some(node_id) = state.target_node else {
        state.open = false;
        return;
    };

    if !state.source_text.trim().is_empty() {
        state.last_result = Some(run_import(state));
    } else {
        state.last_result = None;
    }

    let mut open = state.open;
    let mut should_close = false;

    crate::dialog_chrome::show(
        ctx,
        "svg_import_dialog",
        "Import SVG",
        &mut open,
        true,
        crate::dialog_chrome::DEFAULT_WIDTH,
        |ui| {
            match &state.file_name {
                Some(name) => {
                    ui.horizontal(|ui| {
                        ui.label("File:");
                        ui.weak(name);
                    });
                }
                None => {
                    ui.label("Paste a whole SVG, or just a path's own \"d\" string:");
                    egui::ScrollArea::vertical()
                        .max_height(PASTE_BOX_MAX_HEIGHT)
                        .show(ui, |ui| {
                            ui.add(
                                egui::TextEdit::multiline(&mut state.source_text)
                                    .desired_rows(6)
                                    .desired_width(f32::INFINITY)
                                    .hint_text("Cmd+V / Ctrl+V to paste here…"),
                            );
                        });
                }
            }

            ui.separator();

            egui::ComboBox::from_label("Fit")
                .selected_text(match state.fit {
                    mc_nodes::svg_import::Fit::Bbox => "Content bounds (bbox)",
                    mc_nodes::svg_import::Fit::Viewbox => "SVG viewBox",
                    mc_nodes::svg_import::Fit::None => "None (1 user unit = 1 tile)",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.fit,
                        mc_nodes::svg_import::Fit::Bbox,
                        "Content bounds (bbox)",
                    );
                    ui.selectable_value(
                        &mut state.fit,
                        mc_nodes::svg_import::Fit::Viewbox,
                        "SVG viewBox",
                    );
                    ui.selectable_value(
                        &mut state.fit,
                        mc_nodes::svg_import::Fit::None,
                        "None (1 user unit = 1 tile)",
                    );
                });

            ui.add_enabled_ui(state.fit == mc_nodes::svg_import::Fit::Bbox, |ui| {
                ui.add(
                    egui::Slider::new(&mut state.margin, 0.0..=0.3)
                        .text("Margin")
                        .fixed_decimals(2),
                );
            });

            ui.checkbox(&mut state.wrap, "Wrap");

            ui.separator();

            ui.horizontal(|ui| {
                ui.radio_value(&mut state.append, false, "Replace");
                ui.radio_value(&mut state.append, true, "Append");
            });

            ui.separator();

            let has_geometry = state
                .last_result
                .as_ref()
                .is_some_and(|r| !r.path.subpaths.is_empty());
            if ui
                .add_enabled(has_geometry, egui::Button::new("Import"))
                .clicked()
            {
                if let Some(result) = state.last_result.take() {
                    let old_path = mc_nodes::path::resolve_effective_path(graph, node_id);
                    let new_path = if state.append {
                        mc_core::Path {
                            subpaths: old_path
                                .subpaths
                                .iter()
                                .cloned()
                                .chain(result.path.subpaths.iter().cloned())
                                .collect(),
                            wrap: state.wrap,
                        }
                    } else {
                        result.path
                    };
                    if graph
                        .set_param(node_id, "path", ParamValue::Path(new_path))
                        .is_ok()
                    {
                        history.record_applied(vec![mc_core::Command::SetParam {
                            id: node_id,
                            name: "path".to_string(),
                            value: ParamValue::Path(old_path),
                        }]);
                    }
                    // A successful import closes the dialog outright
                    // rather than lingering on a "done" message — the
                    // owner's own request, and there's nothing left here
                    // worth a second look once the path's already
                    // applied to the graph.
                    should_close = true;
                }
            }

            if let Some(result) = &state.last_result {
                // `import_svg`/`import_d` already push a "no geometry"
                // warning themselves when there's nothing to import, so
                // the warnings list alone covers that case too — no
                // separate empty-result message needed.
                if !result.warnings.is_empty() {
                    ui.add_space(4.0);
                    for warning in &result.warnings {
                        ui.colored_label(egui::Color32::from_rgb(230, 180, 80), warning);
                    }
                }
            }
        },
    );

    state.open = open && !should_close;
}
