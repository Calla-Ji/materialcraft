//! Bridges `mc_core::Graph` (the source of truth) to `egui_snarl`'s node
//! editor widget.
//!
//! `mc_core::Graph` stays authoritative for everything that matters: node
//! kinds/params, connections, undo/redo. `Snarl<NodeUi>` is just a visual
//! index — each UI node wraps the `mc_core::NodeId` it represents, and the
//! only thing it owns that `Graph` doesn't is live drag position during
//! interaction. [`GraphViewer`]'s callbacks (`connect`, `disconnect`, the
//! add/delete-node menus) only ever mutate `Graph`/`History`, never `Snarl`
//! directly; [`sync_snarl_from_graph`] is the one place that reconciles
//! `Snarl`'s nodes and wires to match `Graph` — which is what makes undo/redo
//! (which only touches `Graph`) show up correctly without extra plumbing.

use std::collections::HashMap;

use egui_snarl::ui::{AnyPins, PinInfo, SnarlPin};
use egui_snarl::{InPin, InPinId, NodeId as UiNodeId, OutPin, OutPinId, Snarl};

/// A node's identity in the editor: just a pointer back into `mc_core::Graph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeUi {
    pub core_id: mc_core::NodeId,
}

fn port_color(port_type: mc_core::PortType) -> egui::Color32 {
    match port_type {
        mc_core::PortType::Gray => egui::Color32::from_gray(200),
        mc_core::PortType::Color => egui::Color32::from_rgb(230, 150, 60),
    }
}

/// Reconciles `snarl`'s nodes and wires to match `graph`, adding/removing UI
/// nodes as needed and fully rebuilding the wire set (cheap at the node
/// counts this app deals with, and far simpler than patching it
/// incrementally). Call this once per frame before `Snarl::show`.
pub fn sync_snarl_from_graph(
    graph: &mc_core::Graph,
    snarl: &mut Snarl<NodeUi>,
    core_to_ui: &mut HashMap<mc_core::NodeId, UiNodeId>,
) {
    let stale: Vec<(mc_core::NodeId, UiNodeId)> = core_to_ui
        .iter()
        .filter(|(&core_id, _)| graph.node(core_id).is_none())
        .map(|(&core_id, &ui_id)| (core_id, ui_id))
        .collect();
    for (core_id, ui_id) in stale {
        snarl.remove_node(ui_id);
        core_to_ui.remove(&core_id);
    }

    for (core_id, node) in graph.nodes() {
        if let std::collections::hash_map::Entry::Vacant(entry) = core_to_ui.entry(core_id) {
            let pos = egui::pos2(node.position[0], node.position[1]);
            entry.insert(snarl.insert_node(pos, NodeUi { core_id }));
        }
    }

    let existing_wires: Vec<(OutPinId, InPinId)> = snarl.wires().collect();
    for (out_pin, in_pin) in existing_wires {
        snarl.disconnect(out_pin, in_pin);
    }
    for connection in graph.connections() {
        let (Some(&from_ui), Some(&to_ui)) = (
            core_to_ui.get(&connection.from_node),
            core_to_ui.get(&connection.to_node),
        ) else {
            continue;
        };
        let from_index = graph
            .node(connection.from_node)
            .and_then(|n| mc_nodes::defs::node_def(&n.kind))
            .and_then(|def| {
                def.outputs
                    .iter()
                    .position(|p| p.name == connection.from_port)
            });
        let to_index = graph
            .node(connection.to_node)
            .and_then(|n| mc_nodes::defs::node_def(&n.kind))
            .and_then(|def| def.inputs.iter().position(|p| p.name == connection.to_port));
        if let (Some(output), Some(input)) = (from_index, to_index) {
            snarl.connect(
                OutPinId {
                    node: from_ui,
                    output,
                },
                InPinId { node: to_ui, input },
            );
        }
    }
}

/// A short-lived bundle of mutable references used to render and edit the
/// graph for a single frame. Everything it touches structurally
/// (nodes/params/connections) goes through `graph`/`history`; `gpu`/`pool`/
/// `render_state`/`thumbnail_ids` exist only to show each node's live
/// texture as a small preview in its body.
pub struct GraphViewer<'a> {
    pub graph: &'a mut mc_core::Graph,
    pub history: &'a mut mc_core::History,
    pub selected: &'a mut Option<mc_core::NodeId>,
    pub add_node_search: &'a mut String,
    pub gpu: &'a mut mc_gpu::Context,
    pub pool: &'a mut mc_gpu::TexturePool,
    pub render_state: &'a egui_wgpu::RenderState,
    pub thumbnail_ids: &'a mut HashMap<mc_core::NodeId, egui::TextureId>,
}

impl GraphViewer<'_> {
    fn node_def_of(
        &self,
        snarl: &Snarl<NodeUi>,
        node: UiNodeId,
    ) -> Option<&'static mc_nodes::defs::NodeDef> {
        let core_id = snarl[node].core_id;
        let kind = &self.graph.node(core_id)?.kind;
        mc_nodes::defs::node_def(kind)
    }
}

impl egui_snarl::ui::SnarlViewer<NodeUi> for GraphViewer<'_> {
    fn title(&mut self, node: &NodeUi) -> String {
        self.graph
            .node(node.core_id)
            .and_then(|n| mc_nodes::defs::node_def(&n.kind))
            .map(|def| def.display_name.to_string())
            .unwrap_or_else(|| "?".to_string())
    }

    fn inputs(&mut self, node: &NodeUi) -> usize {
        self.graph
            .node(node.core_id)
            .and_then(|n| mc_nodes::defs::node_def(&n.kind))
            .map_or(0, |def| def.inputs.len())
    }

    fn outputs(&mut self, node: &NodeUi) -> usize {
        self.graph
            .node(node.core_id)
            .and_then(|n| mc_nodes::defs::node_def(&n.kind))
            .map_or(0, |def| def.outputs.len())
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<NodeUi>,
    ) -> impl SnarlPin + 'static {
        let def = self.node_def_of(snarl, pin.id.node);
        let port = def.and_then(|def| def.inputs.get(pin.id.input));
        ui.label(port.map_or("?", |p| p.display_name));
        PinInfo::circle().with_fill(port_color(
            port.map_or(mc_core::PortType::Color, |p| p.port_type),
        ))
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<NodeUi>,
    ) -> impl SnarlPin + 'static {
        let def = self.node_def_of(snarl, pin.id.node);
        let port = def.and_then(|def| def.outputs.get(pin.id.output));
        ui.label(port.map_or("?", |p| p.display_name));
        PinInfo::circle().with_fill(port_color(
            port.map_or(mc_core::PortType::Color, |p| p.port_type),
        ))
    }

    fn has_body(&mut self, _node: &NodeUi) -> bool {
        true
    }

    fn show_body(
        &mut self,
        node: UiNodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<NodeUi>,
    ) {
        let core_id = snarl[node].core_id;
        if ui
            .selectable_label(*self.selected == Some(core_id), "Preview")
            .clicked()
        {
            *self.selected = Some(core_id);
        }

        let Some(key) = mc_nodes::registry::resolve_texture_key(self.graph, core_id) else {
            return;
        };
        let Some(view) = self.pool.view(&key) else {
            return;
        };
        let texture_id = match self.thumbnail_ids.get(&core_id).copied() {
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
                self.thumbnail_ids.insert(core_id, id);
                id
            }
        };
        ui.image(egui::load::SizedTexture::new(
            texture_id,
            egui::Vec2::splat(64.0),
        ));
    }

    fn has_graph_menu(&mut self, _pos: egui::Pos2, _snarl: &mut Snarl<NodeUi>) -> bool {
        true
    }

    fn show_graph_menu(&mut self, pos: egui::Pos2, ui: &mut egui::Ui, _snarl: &mut Snarl<NodeUi>) {
        ui.set_min_width(200.0);
        ui.text_edit_singleline(self.add_node_search);
        ui.separator();

        let query = self.add_node_search.to_lowercase();
        for def in mc_nodes::defs::all_node_defs() {
            if !query.is_empty() && !def.display_name.to_lowercase().contains(&query) {
                continue;
            }
            if ui.button(def.display_name).clicked() {
                let node = def.instantiate();
                let (core_id, transaction) = self.graph.add_node_transaction(node);
                if self.history.do_transaction(self.graph, transaction).is_ok() {
                    let _ = self.graph.set_node_position(core_id, [pos.x, pos.y]);
                    *self.selected = Some(core_id);
                }
                self.add_node_search.clear();
                ui.close();
            }
        }
    }

    fn has_dropped_wire_menu(&mut self, _src_pins: AnyPins, _snarl: &mut Snarl<NodeUi>) -> bool {
        true
    }

    /// Dragging a wire out from a pin and releasing it on empty canvas (the
    /// common "drag out → pick a node → it's already wired up" gesture)
    /// opens this menu instead of the plain add-node one: it's filtered to
    /// node kinds with a compatible port, and auto-connects the new node to
    /// whichever pin(s) were dragged from.
    fn show_dropped_wire_menu(
        &mut self,
        pos: egui::Pos2,
        ui: &mut egui::Ui,
        src_pins: AnyPins,
        snarl: &mut Snarl<NodeUi>,
    ) {
        ui.set_min_width(200.0);
        ui.text_edit_singleline(self.add_node_search);
        ui.separator();

        let query = self.add_node_search.to_lowercase();
        for def in mc_nodes::defs::all_node_defs() {
            if !query.is_empty() && !def.display_name.to_lowercase().contains(&query) {
                continue;
            }

            let compatible = match src_pins {
                AnyPins::Out(pins) => pins.iter().any(|pin| {
                    self.node_def_of(snarl, pin.node)
                        .and_then(|d| d.outputs.get(pin.output))
                        .is_some_and(|from| {
                            def.inputs.iter().any(|to| {
                                mc_nodes::defs::ports_compatible(from.port_type, to.port_type)
                            })
                        })
                }),
                AnyPins::In(pins) => pins.iter().any(|pin| {
                    self.node_def_of(snarl, pin.node)
                        .and_then(|d| d.inputs.get(pin.input))
                        .is_some_and(|to| {
                            def.outputs.iter().any(|from| {
                                mc_nodes::defs::ports_compatible(from.port_type, to.port_type)
                            })
                        })
                }),
            };
            if !compatible {
                continue;
            }

            if ui.button(def.display_name).clicked() {
                let node = def.instantiate();
                let (core_id, transaction) = self.graph.add_node_transaction(node);
                if self.history.do_transaction(self.graph, transaction).is_ok() {
                    let _ = self.graph.set_node_position(core_id, [pos.x, pos.y]);
                    *self.selected = Some(core_id);

                    match src_pins {
                        AnyPins::Out(pins) => {
                            for pin in pins {
                                let from_core = snarl[pin.node].core_id;
                                let Some(from_port) = self
                                    .node_def_of(snarl, pin.node)
                                    .and_then(|d| d.outputs.get(pin.output))
                                else {
                                    continue;
                                };
                                let Some(to_port) = def.inputs.iter().find(|to| {
                                    mc_nodes::defs::ports_compatible(
                                        from_port.port_type,
                                        to.port_type,
                                    )
                                }) else {
                                    continue;
                                };
                                if let Ok(tx) = self.graph.connect_transaction(
                                    from_core,
                                    from_port.name,
                                    core_id,
                                    to_port.name,
                                ) {
                                    let _ = self.history.do_transaction(self.graph, tx);
                                }
                            }
                        }
                        AnyPins::In(pins) => {
                            for pin in pins {
                                let to_core = snarl[pin.node].core_id;
                                let Some(to_port) = self
                                    .node_def_of(snarl, pin.node)
                                    .and_then(|d| d.inputs.get(pin.input))
                                else {
                                    continue;
                                };
                                let Some(from_port) = def.outputs.iter().find(|from| {
                                    mc_nodes::defs::ports_compatible(
                                        from.port_type,
                                        to_port.port_type,
                                    )
                                }) else {
                                    continue;
                                };
                                if let Ok(tx) = self.graph.connect_transaction(
                                    core_id,
                                    from_port.name,
                                    to_core,
                                    to_port.name,
                                ) {
                                    let _ = self.history.do_transaction(self.graph, tx);
                                }
                            }
                        }
                    }
                }
                self.add_node_search.clear();
                ui.close();
            }
        }
    }

    fn has_node_menu(&mut self, _node: &NodeUi) -> bool {
        true
    }

    fn show_node_menu(
        &mut self,
        node: UiNodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<NodeUi>,
    ) {
        let core_id = snarl[node].core_id;
        if ui.button("Delete Node").clicked() {
            let transaction = self.graph.remove_node_transaction(core_id);
            let _ = self.history.do_transaction(self.graph, transaction);
            if *self.selected == Some(core_id) {
                *self.selected = None;
            }
            ui.close();
        }
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<NodeUi>) {
        let from_core = snarl[from.id.node].core_id;
        let to_core = snarl[to.id.node].core_id;

        let Some(from_port) = self
            .node_def_of(snarl, from.id.node)
            .and_then(|def| def.outputs.get(from.id.output))
        else {
            return;
        };
        let Some(to_port) = self
            .node_def_of(snarl, to.id.node)
            .and_then(|def| def.inputs.get(to.id.input))
        else {
            return;
        };

        if !mc_nodes::defs::ports_compatible(from_port.port_type, to_port.port_type) {
            return;
        }

        if let Ok(transaction) =
            self.graph
                .connect_transaction(from_core, from_port.name, to_core, to_port.name)
        {
            let _ = self.history.do_transaction(self.graph, transaction);
        }
    }

    fn disconnect(&mut self, _from: &OutPin, to: &InPin, snarl: &mut Snarl<NodeUi>) {
        let to_core = snarl[to.id.node].core_id;
        let Some(to_port) = self
            .node_def_of(snarl, to.id.node)
            .and_then(|def| def.inputs.get(to.id.input))
        else {
            return;
        };
        if let Some(connection) = self
            .graph
            .connections()
            .iter()
            .find(|c| c.to_node == to_core && c.to_port == to_port.name)
            .cloned()
        {
            let _ = self
                .history
                .do_transaction(self.graph, vec![mc_core::Command::Disconnect(connection)]);
        }
    }

    fn final_node_rect(
        &mut self,
        node: UiNodeId,
        rect: egui::Rect,
        _ui: &mut egui::Ui,
        snarl: &mut Snarl<NodeUi>,
    ) {
        let core_id = snarl[node].core_id;
        let _ = self
            .graph
            .set_node_position(core_id, [rect.min.x, rect.min.y]);
    }
}
