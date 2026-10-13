// SPDX-License-Identifier: MIT OR Apache-2.0
//! A generic parameter panel: every widget comes from the selected node's
//! `NodeDef` (CLAUDE.md §6.6) — no per-node-kind UI code here, so a new node
//! added in `mc_nodes` gets a working panel for free.

/// A param edit mid-drag: the value it had *before* the drag started, so
/// dragging a slider over many frames produces one undo step, not one per
/// frame (CLAUDE.md §9).
pub type DragState = Option<(mc_core::NodeId, String, mc_core::ParamValue)>;

/// A new seed for the dice button — previously just `value.wrapping_add(1)`,
/// which isn't random at all (confirmed bug report: it "just adds one more
/// number"). Mixes the current wall-clock time through a SplitMix64-style
/// finalizer for a well-distributed, visually "random" result each click, no
/// `rand` dependency needed for a UI convenience button that isn't
/// security-sensitive.
fn random_seed() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or_default();
    let mut z = nanos.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u32
}

/// An explanatory aside under a param — e.g. "this only matters once X is
/// set" — styled to actually read as a comment rather than another label:
/// three-quarters of the body text size (half read as too small to be
/// comfortable; this is the settled middle ground) and italic, matching how
/// comments look in code. The established convention for this kind of
/// per-param note going forward (first used for Thread's Fiber Depth/Fibers
/// per Ply hint below) — reach for this instead of a plain `ui.weak(...)`
/// whenever the text is explaining a param's behavior rather than its live
/// connection state (the `[x]`/`[ ] connected` status lines above stay
/// full-size, since those are closer to a label than an aside).
fn param_comment(ui: &mut egui::Ui, text: &str) {
    let body_size = ui
        .style()
        .text_styles
        .get(&egui::TextStyle::Body)
        .map(|font| font.size)
        .unwrap_or(14.0);
    ui.label(
        egui::RichText::new(text)
            .size(body_size * 0.75)
            .italics()
            .weak(),
    );
}

/// Fiber Depth carves relative to a *ply's own* radius, but Height
/// normalizes against the whole bundle's radius — so how much of Height's
/// own 0..1 range the groove can ever reach is capped at
/// `Fiber Depth * (ply radius / bundle radius)`, and that ratio is set by
/// Plies and Cable Groups (packing() 's own fixed ratios), *not* Threads —
/// confirmed by rendering the same comparison at Threads 1 and Threads 4
/// and getting an identical difference, after an earlier version of this
/// hint claimed Threads mattered here too. The full, undiluted groove
/// pattern is always in the Fiber output itself (its whole point, per the
/// spec — wire it in as a separate roughness/AO detail instead of judging
/// Fiber Twist from Height alone).
fn thread_fiber_param_hint(ui: &mut egui::Ui, node_kind: &str, param_name: &str) {
    if node_kind == "thread_shape"
        && matches!(param_name, "fibers_per_ply" | "fiber_depth" | "fiber_mode")
    {
        param_comment(
            ui,
            "How visible this is in Height (not in the Fiber output itself) is capped by \
             Plies/Cable Groups, not Threads \u{2014} for the full pattern, use the Fiber \
             output directly.",
        );
    }
}

/// Seed only ever shifts where a *ply* sits in its own orbit around the
/// thread axis — a single (unplied) yarn has no orbit to shift, so Seed is
/// a genuine no-op at Plies 1 (confirmed: `thread_shape`'s own
/// `seed_only_has_an_effect_with_two_or_more_plies` test). Explained here
/// rather than hiding the control outright — a user who already has a
/// seed dialed in on a Plies-2+ thread shouldn't lose that value from view
/// just because they're trying Plies 1.
fn thread_seed_hint(
    ui: &mut egui::Ui,
    node_kind: &str,
    param_name: &str,
    current_params: &std::collections::BTreeMap<String, mc_core::ParamValue>,
) {
    if node_kind != "thread_shape" || param_name != "seed" {
        return;
    }
    let plies = current_params
        .get("plies")
        .and_then(mc_core::ParamValue::as_int)
        .unwrap_or(2);
    if plies <= 1 {
        param_comment(
            ui,
            "No effect at Plies 1 \u{2014} a single yarn has no ply orbit for Seed to \
             shift.",
        );
    }
}

/// A dropdown of common named yarn constructions (values confirmed against
/// the owner's own screenshots of each one actually rendered, per
/// `mc_nodes::thread_shape::PRESETS`'s own doc comment) that set several of
/// Thread's params at once — the same one-click-starting-point idea as
/// Pattern Generator's own Shape input, just for an entire construction
/// instead of one silhouette. Every param a preset touches stays a normal,
/// live, editable slider afterward; this is a one-shot action, not a mode
/// — there's deliberately no "which preset is currently active" indicator,
/// since editing even one slider after applying a preset already makes
/// that concept meaningless (the result isn't "that preset" anymore, it's
/// just whatever the sliders currently say).
fn thread_preset_picker(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    id: mc_core::NodeId,
    node_kind: &str,
) {
    if node_kind != "thread_shape" {
        return;
    }
    let mut chosen = None;
    egui::ComboBox::from_label("Preset")
        .selected_text("Choose\u{2026}")
        .show_ui(ui, |ui| {
            for preset in mc_nodes::thread_shape::PRESETS {
                if ui.selectable_label(false, preset.name).clicked() {
                    chosen = Some(preset);
                }
            }
        });
    if let Some(preset) = chosen {
        commit_once_multi(
            graph,
            history,
            id,
            &[
                ("threads", mc_core::ParamValue::Int(preset.threads as i32)),
                ("thickness", mc_core::ParamValue::Float(preset.thickness)),
                ("plies", mc_core::ParamValue::Int(preset.plies as i32)),
                (
                    "ply_direction",
                    mc_core::ParamValue::Enum(preset.ply_direction.name().to_string()),
                ),
                (
                    "twist_angle",
                    mc_core::ParamValue::Float(preset.twist_angle),
                ),
                (
                    "fiber_mode",
                    mc_core::ParamValue::Enum(preset.fiber_mode.name().to_string()),
                ),
                (
                    "fibers_per_ply",
                    mc_core::ParamValue::Int(preset.fibers_per_ply as i32),
                ),
                (
                    "fiber_depth",
                    mc_core::ParamValue::Float(preset.fiber_depth),
                ),
                (
                    "cable_groups",
                    mc_core::ParamValue::Int(preset.cable_groups as i32),
                ),
                (
                    "cable_angle",
                    mc_core::ParamValue::Float(preset.cable_angle),
                ),
            ],
        );
    }
}

pub fn show(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    selected: Option<mc_core::NodeId>,
    dragging: &mut DragState,
) {
    egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(12, 8))
        .show(ui, |ui| {
            show_inner(ui, graph, history, selected, dragging);
        });
}

fn show_inner(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    selected: Option<mc_core::NodeId>,
    dragging: &mut DragState,
) {
    let Some(id) = selected else {
        ui.weak("(select a node to edit its parameters)");
        return;
    };
    let Some(node) = graph.node(id) else {
        ui.weak("(selected node no longer exists)");
        return;
    };
    let Some(def) = mc_nodes::defs::node_def(&node.kind) else {
        ui.weak(format!("(unknown node kind {:?})", node.kind));
        return;
    };
    // Snapshot params (and kind) as owned data so the loop below is free to
    // mutate `graph` without fighting the borrow checker over `node`.
    let current_params = node.params.clone();
    let node_kind = node.kind.clone();

    ui.heading(def.display_name);
    ui.weak(def.tooltip);

    thread_preset_picker(ui, graph, history, id, &node_kind);

    // A connected-vs-fallback status line for inputs where that's genuinely
    // ambiguous from the rendered result alone (e.g. a connected Fiber using
    // default params that happen to match the built-in fallback's own
    // defaults looks identical either way) — see
    // `mc_nodes::defs::input_is_optional`'s doc comment.
    for port in def.inputs {
        if !mc_nodes::defs::input_is_optional(&node_kind, port.name) {
            continue;
        }
        // Plain ASCII, not a Unicode check mark/circle glyph — the proportional
        // font (Geist) doesn't cover those, which rendered as a tofu box
        // instead of a status symbol.
        let status = if graph.upstream(id, port.name).is_some() {
            format!("[x] {}: connected", port.display_name)
        } else {
            format!(
                "[ ] {}: not connected \u{2014} using the built-in default below",
                port.display_name
            )
        };
        ui.weak(status);
    }
    ui.separator();

    // Pattern Generator: once a custom Shape is wired in, Rows and Columns
    // collapse into one linked "Tile Count" slider instead of their normal
    // independent ones below — a custom shape stamped into an uneven grid
    // would read as visibly stretched, unlike the built-in brick/grid shape
    // (which *wants* independent Rows/Columns, e.g. the classic 6x4 brick
    // layout). Gap/Edge Darkening only ever affected the built-in shape, so
    // they're hidden too once Shape is connected, replaced by Shape Size
    // (which conversely has no effect on the built-in shape, so it's hidden
    // the rest of the time). Only this one node/case; not a generic
    // ParamSpecKind or `param_is_relevant` rule, since both of those work
    // off param *values*, not port connection state.
    let shape_connected = node_kind == "tile_generator" && graph.upstream(id, "shape").is_some();
    if shape_connected {
        let rows = current_params
            .get("rows")
            .and_then(mc_core::ParamValue::as_int)
            .unwrap_or(4);
        let columns = current_params
            .get("columns")
            .and_then(mc_core::ParamValue::as_int)
            .unwrap_or(4);
        let mut tile_count = rows.max(columns);
        let response = ui
            .add(egui::Slider::new(&mut tile_count, 1..=32).text("Tile Count"))
            .on_hover_text(
                "Rows and Columns together, locked to the same value while a custom Shape \
                 is wired in, so it isn't stretched across an uneven grid.",
            );
        if response.changed() {
            commit_once_multi(
                graph,
                history,
                id,
                &[
                    ("rows", mc_core::ParamValue::Int(tile_count)),
                    ("columns", mc_core::ParamValue::Int(tile_count)),
                ],
            );
        }
    }

    // Fabric Generator: "Thread Width"/"Thread Roundness" only ever affect
    // the built-in fallback fiber this node renders when nothing's wired
    // into Thread — once a Thread node *is* connected, dragging them did
    // nothing, which read exactly like "the connection isn't working" (a
    // real bug report — the sliders were right there, visibly draggable,
    // producing zero visible change). Stockinette is the one exception:
    // it never samples Thread at all (see fabric_generator.wgsl's own doc
    // comment), so Thread Width keeps governing its loop width regardless
    // of whether something's connected.
    let fabric_thread_connected =
        node_kind == "fabric_generator" && graph.upstream(id, "fiber").is_some();
    let fabric_weave_type = current_params
        .get("weave_type")
        .and_then(|v| match v {
            mc_core::ParamValue::Enum(name) => Some(name.as_str()),
            _ => None,
        })
        .unwrap_or("Plain Weave");
    let fabric_is_stockinette = fabric_weave_type == "Stockinette";
    let hide_thread_width = fabric_thread_connected && !fabric_is_stockinette;

    for param in def.params {
        if shape_connected && matches!(param.name, "rows" | "columns" | "gap" | "bevel") {
            continue;
        }
        if !shape_connected && param.name == "shape_size" {
            continue;
        }
        if hide_thread_width && param.name == "thread_width" {
            continue;
        }
        if !mc_nodes::defs::param_is_relevant(&node_kind, param.name, &current_params) {
            continue;
        }
        let current = current_params.get(param.name).cloned();
        match &param.kind {
            mc_nodes::defs::ParamSpecKind::Float { min, max, default } => {
                let mut value = current.and_then(|v| v.as_float()).unwrap_or(*default);
                let response = ui
                    .add(egui::Slider::new(&mut value, *min..=*max).text(param.display_name))
                    .on_hover_text(param.tooltip);
                commit_live(
                    ui,
                    graph,
                    history,
                    dragging,
                    id,
                    param.name,
                    &response,
                    mc_core::ParamValue::Float(value),
                );
                thread_fiber_param_hint(ui, &node_kind, param.name);
            }
            mc_nodes::defs::ParamSpecKind::Int { min, max, default } => {
                let mut value = current.and_then(|v| v.as_int()).unwrap_or(*default);
                let response = ui
                    .add(egui::Slider::new(&mut value, *min..=*max).text(param.display_name))
                    .on_hover_text(param.tooltip);
                commit_live(
                    ui,
                    graph,
                    history,
                    dragging,
                    id,
                    param.name,
                    &response,
                    mc_core::ParamValue::Int(value),
                );
                thread_fiber_param_hint(ui, &node_kind, param.name);
            }
            mc_nodes::defs::ParamSpecKind::Seed { default } => {
                let mut value = current.and_then(|v| v.as_seed()).unwrap_or(*default);
                ui.horizontal(|ui| {
                    let response = ui
                        .add(
                            egui::DragValue::new(&mut value)
                                .prefix(format!("{}: ", param.display_name)),
                        )
                        .on_hover_text(param.tooltip);
                    commit_live(
                        ui,
                        graph,
                        history,
                        dragging,
                        id,
                        param.name,
                        &response,
                        mc_core::ParamValue::Seed(value),
                    );
                    if ui.button("🎲").clicked() {
                        commit_once(
                            graph,
                            history,
                            id,
                            param.name,
                            mc_core::ParamValue::Seed(random_seed()),
                        );
                    }
                });
                thread_seed_hint(ui, &node_kind, param.name, &current_params);
            }
            mc_nodes::defs::ParamSpecKind::Enum { options, default } => {
                let value = match &current {
                    Some(mc_core::ParamValue::Enum(s)) => s.clone(),
                    _ => (*default).to_string(),
                };
                let mut chosen = None;
                egui::ComboBox::from_label(param.display_name)
                    .selected_text(value.clone())
                    .show_ui(ui, |ui| {
                        for option in *options {
                            if ui.selectable_label(value == *option, *option).clicked() {
                                chosen = Some((*option).to_string());
                            }
                        }
                    });
                if let Some(new_value) = chosen {
                    commit_once(
                        graph,
                        history,
                        id,
                        param.name,
                        mc_core::ParamValue::Enum(new_value),
                    );
                }
                thread_fiber_param_hint(ui, &node_kind, param.name);
            }
            mc_nodes::defs::ParamSpecKind::Color { default } => {
                let mut value = match &current {
                    Some(mc_core::ParamValue::Color(c)) => *c,
                    _ => *default,
                };
                ui.horizontal(|ui| {
                    ui.label(param.display_name);
                    let response = ui.color_edit_button_rgba_unmultiplied(&mut value);
                    if response.changed() {
                        commit_once(
                            graph,
                            history,
                            id,
                            param.name,
                            mc_core::ParamValue::Color(value),
                        );
                    }
                });
            }
            mc_nodes::defs::ParamSpecKind::Bool { default } => {
                let mut value = current.and_then(|v| v.as_bool()).unwrap_or(*default);
                if ui
                    .checkbox(&mut value, param.display_name)
                    .on_hover_text(param.tooltip)
                    .changed()
                {
                    commit_once(
                        graph,
                        history,
                        id,
                        param.name,
                        mc_core::ParamValue::Bool(value),
                    );
                }
            }
            mc_nodes::defs::ParamSpecKind::Vec2 { min, max, default } => {
                let mut value = current.and_then(|v| v.as_vec2()).unwrap_or(*default);
                ui.horizontal(|ui| {
                    ui.label(param.display_name);
                    let x_response = ui
                        .add(egui::Slider::new(&mut value[0], *min..=*max).text("X"))
                        .on_hover_text(param.tooltip);
                    let y_response = ui
                        .add(egui::Slider::new(&mut value[1], *min..=*max).text("Y"))
                        .on_hover_text(param.tooltip);
                    // One slider's `changed()` already reflects both axes
                    // together, since `value` is read fresh each frame —
                    // committing on either response alone (not a merged
                    // one) just means only the axis actually dragged
                    // triggers a write.
                    commit_live(
                        ui,
                        graph,
                        history,
                        dragging,
                        id,
                        param.name,
                        &x_response,
                        mc_core::ParamValue::Vec2(value),
                    );
                    commit_live(
                        ui,
                        graph,
                        history,
                        dragging,
                        id,
                        param.name,
                        &y_response,
                        mc_core::ParamValue::Vec2(value),
                    );
                });
            }
            mc_nodes::defs::ParamSpecKind::Gradient { default } => {
                let mut stops = match &current {
                    Some(mc_core::ParamValue::Gradient(s)) if !s.is_empty() => s.clone(),
                    _ => default.to_vec(),
                };
                ui.label(param.display_name);
                let stop_count = stops.len();
                let mut changed = false;
                let mut remove_index = None;
                for (i, stop) in stops.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        if ui
                            .add(egui::Slider::new(&mut stop.position, 0.0..=1.0).text("pos"))
                            .changed()
                        {
                            changed = true;
                        }
                        if ui
                            .color_edit_button_rgba_unmultiplied(&mut stop.color)
                            .changed()
                        {
                            changed = true;
                        }
                        if stop_count > 2 && ui.small_button("✕").clicked() {
                            remove_index = Some(i);
                        }
                    });
                }
                if let Some(i) = remove_index {
                    stops.remove(i);
                    changed = true;
                }
                if ui.small_button("+ Add Stop").clicked() {
                    stops.push(mc_core::GradientStop {
                        position: 0.5,
                        color: [0.5, 0.5, 0.5, 1.0],
                    });
                    changed = true;
                }
                if changed {
                    commit_once(
                        graph,
                        history,
                        id,
                        param.name,
                        mc_core::ParamValue::Gradient(stops),
                    );
                }
            }
            mc_nodes::defs::ParamSpecKind::ImagePath => {
                let image_ref = current
                    .as_ref()
                    .and_then(mc_core::ParamValue::as_image)
                    .cloned()
                    .unwrap_or_default();
                // Whether the *next* load embeds the file's bytes into the
                // graph or just remembers its path — read from the sibling
                // "embed_in_graph" Bool param (its own checkbox renders via
                // the ordinary Bool case above, same loop).
                let embed_next_load = current_params
                    .get("embed_in_graph")
                    .and_then(mc_core::ParamValue::as_bool)
                    .unwrap_or(false);

                ui.horizontal(|ui| {
                    ui.label(param.display_name);
                    if ui.button("Load Image\u{2026}").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter(
                                "Image",
                                &["png", "jpg", "jpeg", "bmp", "tga", "gif", "webp"],
                            )
                            .pick_file()
                        {
                            let path_string = path.display().to_string();
                            let embedded_base64 = embed_next_load
                                .then(|| std::fs::read(&path).ok())
                                .flatten()
                                .map(|bytes| {
                                    use base64::Engine;
                                    base64::engine::general_purpose::STANDARD.encode(bytes)
                                });
                            commit_once(
                                graph,
                                history,
                                id,
                                param.name,
                                mc_core::ParamValue::Image(mc_core::ImageRef {
                                    path: Some(path_string),
                                    embedded_base64,
                                }),
                            );
                        }
                    }
                });
                let status = match (&image_ref.path, image_ref.embedded_base64.is_some()) {
                    (Some(path), true) => format!(
                        "Embedded: {}",
                        std::path::Path::new(path)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(path)
                    ),
                    (Some(path), false) => format!(
                        "Linked: {}",
                        std::path::Path::new(path)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(path)
                    ),
                    (None, _) => "(none loaded)".to_string(),
                };
                ui.weak(status).on_hover_text(param.tooltip);
            }
        }
    }
}

/// Applies a value live every frame (so the preview tracks the drag), and
/// records a single undo step spanning the whole drag gesture once it ends.
#[allow(clippy::too_many_arguments)]
fn commit_live(
    _ui: &egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    dragging: &mut DragState,
    id: mc_core::NodeId,
    name: &str,
    response: &egui::Response,
    new_value: mc_core::ParamValue,
) {
    if response.drag_started() {
        if let Some(old) = graph.node(id).and_then(|n| n.params.get(name).cloned()) {
            *dragging = Some((id, name.to_string(), old));
        }
    }
    if response.changed() {
        let _ = graph.set_param(id, name, new_value);
    }
    if response.drag_stopped() {
        if let Some((drag_id, drag_name, old_value)) = dragging.take() {
            if drag_id == id && drag_name == name {
                history.record_applied(vec![mc_core::Command::SetParam {
                    id,
                    name: name.to_string(),
                    value: old_value,
                }]);
            }
        }
    }
}

/// For instantaneous edits (a combo box choice, a dice-button click) that
/// aren't a drag gesture — each one is its own undo step.
fn commit_once(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    id: mc_core::NodeId,
    name: &str,
    new_value: mc_core::ParamValue,
) {
    let old = graph.node(id).and_then(|n| n.params.get(name).cloned());
    if graph.set_param(id, name, new_value).is_ok() {
        if let Some(old_value) = old {
            history.record_applied(vec![mc_core::Command::SetParam {
                id,
                name: name.to_string(),
                value: old_value,
            }]);
        }
    }
}

/// Like `commit_once`, but for several params that should feel like one
/// user action (e.g. Pattern Generator's linked Tile Count setting both
/// Rows and Columns) — one undo step for all of them together, rather than
/// needing Undo pressed once per param to fully revert the edit.
fn commit_once_multi(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    id: mc_core::NodeId,
    new_values: &[(&str, mc_core::ParamValue)],
) {
    let mut inverse = Vec::new();
    for (name, new_value) in new_values {
        let old = graph.node(id).and_then(|n| n.params.get(*name).cloned());
        if graph.set_param(id, name, new_value.clone()).is_ok() {
            if let Some(old_value) = old {
                inverse.push(mc_core::Command::SetParam {
                    id,
                    name: name.to_string(),
                    value: old_value,
                });
            }
        }
    }
    if !inverse.is_empty() {
        history.record_applied(inverse);
    }
}
