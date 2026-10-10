//! A screen-space "add node" popup for right-clicking empty canvas, or for
//! dropping a dragged wire on empty canvas (filtered to node kinds with a
//! compatible port, auto-connected on pick — see `canvas.rs`'s
//! `connect_dropped_wire`).
//!
//! A plain `egui::Area` fixed at the real clicked *screen* position, rather
//! than egui's generic popup placement: since `canvas.rs` does its own pan/
//! zoom transform explicitly (there's no hidden transformed layer the way
//! egui-snarl had one), the caller just hands over the screen position it
//! already has, with no special-casing needed.

pub struct DropFilter {
    pub port_type: mc_core::PortType,
    /// Whether the dragged wire needs an *input* on the newly-added node
    /// (true — the wire started from an output) or an *output* (false).
    pub need_input: bool,
}

pub struct AddNodeMenu {
    screen_pos: egui::Pos2,
    graph_pos: egui::Pos2,
    search: String,
    filter: Option<DropFilter>,
    /// Suppresses the "clicked outside closes it" check for the one frame
    /// the menu was opened on — opening from a wire-drop happens on a
    /// primary-button release, and without this the same release would be
    /// seen as "a primary click outside the not-yet-drawn popup" and close
    /// the menu the instant it opened.
    just_opened: bool,
}

impl AddNodeMenu {
    pub fn open(screen_pos: egui::Pos2, graph_pos: egui::Pos2) -> Self {
        Self {
            screen_pos,
            graph_pos,
            search: String::new(),
            filter: None,
            just_opened: true,
        }
    }

    pub fn open_for_dropped_wire(
        screen_pos: egui::Pos2,
        graph_pos: egui::Pos2,
        filter: DropFilter,
    ) -> Self {
        Self {
            filter: Some(filter),
            ..Self::open(screen_pos, graph_pos)
        }
    }
}

/// Draws the menu if `state` is `Some`, closing it (setting `state` back to
/// `None`) when a node is picked, Escape is pressed, or the user clicks
/// outside the popup. Returns the picked node def and the graph-space
/// position it should be placed at, if one was picked this frame.
pub fn show(
    ctx: &egui::Context,
    state: &mut Option<AddNodeMenu>,
) -> Option<(&'static mc_nodes::defs::NodeDef, egui::Pos2)> {
    let menu = state.as_mut()?;
    let suppress_dismiss = menu.just_opened;
    menu.just_opened = false;

    let mut picked = None;
    let area_response = egui::Area::new(egui::Id::new("materialcraft_add_node_menu"))
        .fixed_pos(menu.screen_pos)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_width(220.0);
                ui.horizontal(|ui| {
                    ui.label("Search:");
                    ui.text_edit_singleline(&mut menu.search).request_focus();
                });
                ui.separator();

                let query = menu.search.to_lowercase();
                egui::ScrollArea::vertical()
                    .max_height(320.0)
                    .show(ui, |ui| {
                        for def in mc_nodes::defs::all_node_defs().iter().filter(|def| {
                            (query.is_empty() || def.display_name.to_lowercase().contains(&query))
                                && menu.filter.as_ref().is_none_or(|f| {
                                    if f.need_input {
                                        def.inputs.iter().any(|p| {
                                            mc_nodes::defs::ports_compatible(
                                                f.port_type,
                                                p.port_type,
                                            )
                                        })
                                    } else {
                                        def.outputs.iter().any(|p| {
                                            mc_nodes::defs::ports_compatible(
                                                p.port_type,
                                                f.port_type,
                                            )
                                        })
                                    }
                                })
                        }) {
                            if ui.button(def.display_name).clicked() {
                                picked = Some((def, menu.graph_pos));
                            }
                        }
                    });
            });
        });

    // Only a *left* click should be able to dismiss by landing outside —
    // otherwise the right-click that opens this menu (which lands outside
    // the not-yet-drawn popup) would immediately close it again.
    let clicked_outside = !suppress_dismiss
        && ctx.input(|i| i.pointer.primary_clicked())
        && ctx
            .input(|i| i.pointer.interact_pos())
            .is_some_and(|pos| !area_response.response.rect.contains(pos));
    let escape = ctx.input(|i| i.key_pressed(egui::Key::Escape));

    if picked.is_some() || clicked_outside || escape {
        *state = None;
    }

    picked
}
