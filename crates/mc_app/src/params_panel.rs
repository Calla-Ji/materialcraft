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
    // Snapshot params as owned data so the loop below is free to mutate
    // `graph` without fighting the borrow checker over `node`.
    let current_params = node.params.clone();

    ui.heading(def.display_name);
    ui.weak(def.tooltip);
    ui.separator();

    for param in def.params {
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
