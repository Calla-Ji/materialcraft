//! A screen-space "add node" popup for right-clicking empty canvas, used
//! instead of egui-snarl's own built-in graph-menu mechanism.
//!
//! egui-snarl positions its built-in context menu using egui's generic popup
//! placement, which has no idea the node canvas renders through its own pan/
//! zoom transform — so the menu (and therefore any node added from it) lands
//! away from the actual click, worse the further the canvas has scrolled or
//! the more its containing panel has moved on screen. A plain `egui::Area`
//! fixed at the real clicked *screen* position sidesteps all of that: the
//! caller (`main.rs`) hands us both a screen-space position (straight from
//! `egui::Context`'s raw pointer input, for drawing the popup) and a
//! graph-space position (from the canvas response's own
//! `interact_pointer_pos()`, which egui-snarl's transformed layer already
//! reports in graph space — no manual pan/zoom math needed, for the same
//! reason egui-snarl itself doesn't do any for e.g. rect-selection). Neither
//! one goes through egui's confused popup-placement path at all.
//!
//! The drag-a-wire-and-drop-it-on-empty-space menu still uses egui-snarl's
//! built-in mechanism (see graph_view.rs's `show_dropped_wire_menu`) and so
//! still has the old positioning quirk — replacing that one too would mean
//! reimplementing egui-snarl's internal wire-drop tracking ourselves, which
//! is a larger job than this fix; see PROGRESS.md.

pub struct AddNodeMenu {
    screen_pos: egui::Pos2,
    graph_pos: egui::Pos2,
    search: String,
}

impl AddNodeMenu {
    pub fn open(screen_pos: egui::Pos2, graph_pos: egui::Pos2) -> Self {
        Self {
            screen_pos,
            graph_pos,
            search: String::new(),
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
                            query.is_empty() || def.display_name.to_lowercase().contains(&query)
                        }) {
                            if ui.button(def.display_name).clicked() {
                                picked = Some((def, menu.graph_pos));
                            }
                        }
                    });
            });
        });

    // Opened by a right-click; only a *left* click should be able to dismiss
    // it by landing outside — otherwise the very right-click that opens the
    // menu (which lands outside the not-yet-drawn popup) would immediately
    // close it again on the same frame.
    let clicked_outside = ctx.input(|i| i.pointer.primary_clicked())
        && ctx
            .input(|i| i.pointer.interact_pos())
            .is_some_and(|pos| !area_response.response.rect.contains(pos));
    let escape = ctx.input(|i| i.key_pressed(egui::Key::Escape));

    if picked.is_some() || clicked_outside || escape {
        *state = None;
    }

    picked
}
