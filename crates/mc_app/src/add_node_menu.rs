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
    /// `search`'s value as of the last frame — when it changes, the
    /// keyboard-highlighted row resets to the top of the (now different)
    /// filtered list rather than keeping a stale index into it.
    last_search: String,
    /// Index into the current filtered/sorted list that Up/Down move and
    /// Enter picks; also follows the mouse on hover, so keyboard and mouse
    /// selection never disagree.
    highlighted: usize,
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
            last_search: String::new(),
            highlighted: 0,
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

    let (move_down, move_up, confirm) = ctx.input(|i| {
        (
            i.key_pressed(egui::Key::ArrowDown),
            i.key_pressed(egui::Key::ArrowUp),
            i.key_pressed(egui::Key::Enter),
        )
    });

    let mut picked = None;
    let area_response = egui::Area::new(egui::Id::new("materialcraft_add_node_menu"))
        .fixed_pos(menu.screen_pos)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style())
                .fill(crate::theme::NEUTRON_STAR)
                .show(ui, |ui| {
                    // Wide enough for the longest current display name ("ON
                    // Ambient Occlusion" — this list isn't the header's "ON "-
                    // shortened form) plus the scrollbar reserved below, with
                    // room to spare for names added later.
                    ui.set_width(260.0);
                    ui.horizontal(|ui| {
                        ui.label("Search:");
                        ui.text_edit_singleline(&mut menu.search).request_focus();
                    });
                    ui.separator();

                    if menu.search != menu.last_search {
                        menu.last_search = menu.search.clone();
                        menu.highlighted = 0;
                    }

                    let query = menu.search.to_lowercase();
                    // Alphabetical, not declaration order — the sidebar's
                    // Node Library panel already groups by category, so this
                    // search list is the place to find a node by name
                    // instead.
                    let mut defs: Vec<_> = mc_nodes::defs::all_node_defs()
                        .iter()
                        .filter(|def| {
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
                        })
                        .collect();
                    defs.sort_by_key(|def| def.display_name);

                    if !defs.is_empty() {
                        menu.highlighted = menu.highlighted.min(defs.len() - 1);
                        if move_down {
                            menu.highlighted = (menu.highlighted + 1).min(defs.len() - 1);
                        }
                        if move_up {
                            menu.highlighted = menu.highlighted.saturating_sub(1);
                        }
                        if confirm {
                            picked = Some((defs[menu.highlighted], menu.graph_pos));
                        }
                    }

                    egui::ScrollArea::vertical()
                        .max_height(320.0)
                        // `ScrollArea` defaults to shrinking itself to fit its
                        // content on *both* axes — so with buttons narrower
                        // than the 260px frame, it was shrinking its own width
                        // down to roughly the widest button, and drawing the
                        // scrollbar at the edge of *that* narrower width
                        // instead of the popup's actual right edge: a dead gap
                        // on both sides of the bar (wide empty space, bar, more
                        // empty space to the real edge), not a misplaced-but-
                        // adjacent-to-content bar. `auto_shrink([false, ...])`
                        // makes it fill the full 260px width the frame actually
                        // has, so the bar sits flush against the real edge.
                        .auto_shrink([false, true])
                        // Always reserved, not just when a scrollbar happens to
                        // be showing — otherwise a button's width is measured
                        // against the *full* unreserved width right up until
                        // the list becomes tall enough to need scrolling, at
                        // which point the bar pops in and immediately overlaps
                        // text that was already laid out assuming it had that
                        // space.
                        .scroll_bar_visibility(
                            egui::scroll_area::ScrollBarVisibility::AlwaysVisible,
                        )
                        .show(ui, |ui| {
                            for (i, def) in defs.iter().enumerate() {
                                let label = egui::RichText::new(def.display_name)
                                    .color(crate::theme::category_color(def.category));
                                let mut button = egui::Button::new(label);
                                if i == menu.highlighted {
                                    button = button.fill(crate::theme::WORMHOLE_GREY);
                                }
                                let response = ui.add(button);
                                if response.hovered() {
                                    menu.highlighted = i;
                                }
                                if response.clicked() {
                                    picked = Some((*def, menu.graph_pos));
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
