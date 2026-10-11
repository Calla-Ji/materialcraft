// SPDX-License-Identifier: MIT OR Apache-2.0
//! A generic parameter panel: every widget comes from the selected node's
//! `NodeDef` (CLAUDE.md §6.6) — no per-node-kind UI code here, so a new node
//! added in `mc_nodes` gets a working panel for free.

/// A param edit mid-drag: the value it had *before* the drag started, so
/// dragging a slider over many frames produces one undo step, not one per
/// frame (CLAUDE.md §9).
pub type DragState = Option<(mc_core::NodeId, String, mc_core::ParamValue)>;

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
    ui.separator();

    for param in def.params {
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
                            mc_core::ParamValue::Seed(value.wrapping_add(1)),
                        );
                    }
                });
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
