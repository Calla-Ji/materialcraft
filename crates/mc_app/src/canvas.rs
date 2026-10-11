//! A fully custom node-graph canvas: pan, zoom, node dragging, click/shift/
//! box selection, and drag-to-connect wires — built directly on egui
//! primitives instead of egui-snarl.
//!
//! `mc_core::Graph` stays the only source of truth for node kind/params/
//! connections, exactly as before; this module owns purely-visual state
//! (camera pan/zoom, multi-selection, in-progress drag/wire gestures) and
//! edits `Graph` through the same `Command`/`Transaction` API the rest of
//! the app already uses, so undo/redo keeps working with no extra plumbing.
//! Node position is intentionally *not* undo-tracked (CLAUDE.md: "purely
//! cosmetic, ignored by evaluation"), matching the old behavior.
//!
//! Built to replace egui-snarl after its selection model turned out to have
//! no public way to clear or replace its own internal (and visually
//! hardcoded) selection — a plain click elsewhere could never deselect, and
//! that limitation sat underneath both rendering *and* input handling with
//! no seam to patch around. Owning both here means every gesture (what
//! clears a selection, what a drag does, how box-select works) is ours to
//! define and change going forward.

use std::collections::{HashMap, HashSet};

use egui::{Color32, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};
use mc_core::{NodeId, PortType};

const NODE_WIDTH: f32 = 170.0;
const HEADER_HEIGHT: f32 = 24.0;
const PORT_ROW_HEIGHT: f32 = 20.0;
const BODY_PADDING: f32 = 6.0;
/// Fills the node's full width (minus the usual side margin) rather than
/// being a small fixed square with dead space around it — stays square, so
/// the node ends up a bit taller too.
const THUMBNAIL_SIZE: f32 = NODE_WIDTH - 2.0 * BODY_PADDING;
const PIN_RADIUS: f32 = 5.0;
const PIN_HIT_RADIUS: f32 = 10.0;
const WIRE_HIT_DISTANCE: f32 = 12.0;
const PREVIEW_DOT_RADIUS: f32 = 4.0;
const PREVIEW_DOT_MARGIN: f32 = 12.0;
const DOT_SPACING: f32 = 16.0;
const HEADER_TEXT_LEFT: f32 = 8.0;
/// Space reserved on the header's right side for the preview/delete dots, so
/// the title text never runs underneath them.
const HEADER_TEXT_RIGHT_RESERVED: f32 = PREVIEW_DOT_MARGIN + DOT_SPACING + 6.0;

/// Which pin a wire-drag gesture started from, or that a drawn wire ends at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PinRef {
    Output { node: NodeId, index: usize },
    Input { node: NodeId, index: usize },
}

struct PinInfo {
    r: PinRef,
    pos: Pos2,
    port_type: PortType,
}

/// Visual/interaction state for the node canvas: camera, multi-selection,
/// and any in-progress drag gesture. Everything structural lives in
/// `mc_core::Graph` instead.
pub struct CanvasState {
    pan: Vec2,
    zoom: f32,
    pub selected: HashSet<NodeId>,
    wire_drag: Option<PinRef>,
    /// Screen-space drag-start position, while a Shift+drag box-select is in
    /// progress.
    box_select_start: Option<Pos2>,
}

impl Default for CanvasState {
    fn default() -> Self {
        Self {
            pan: Vec2::ZERO,
            zoom: 1.0,
            selected: HashSet::new(),
            wire_drag: None,
            box_select_start: None,
        }
    }
}

/// What the caller still needs to do after `show` returns — `main.rs` drives
/// `add_node_menu` the same way it already did before this rewrite.
pub struct CanvasOutput {
    pub open_add_menu_at: Option<(Pos2, Pos2)>,
    /// A wire was dragged out from a pin and dropped on empty canvas (no
    /// compatible pin under the release point) — `main.rs` should open the
    /// add-node menu filtered to the given port type/direction, then connect
    /// whatever gets picked back to this pin.
    pub dropped_wire: Option<DroppedWire>,
}

pub struct DroppedWire {
    pub screen_pos: Pos2,
    pub graph_pos: Pos2,
    pub from_node: NodeId,
    pub from_port_index: usize,
    pub from_is_output: bool,
    pub from_port_type: PortType,
}

fn graph_to_screen(origin: Pos2, pan: Vec2, zoom: f32, p: Pos2) -> Pos2 {
    origin + (p.to_vec2() - pan) * zoom
}

fn screen_to_graph(origin: Pos2, pan: Vec2, zoom: f32, s: Pos2) -> Pos2 {
    (pan + (s - origin) / zoom).to_pos2()
}

/// A faint grid of dots over the canvas background ("a light dot grid in
/// twinkling grey") — drawn in graph space so it pans/zooms with the camera
/// like every other node editor's background grid, rather than staying
/// fixed to the screen.
const GRID_BASE_SPACING: f32 = 24.0;

fn draw_dot_grid(painter: &egui::Painter, canvas_rect: Rect, origin: Pos2, pan: Vec2, zoom: f32) {
    // Keeps on-screen dot spacing within a sane range regardless of zoom: at
    // a fixed graph-space spacing, zooming out would either draw an
    // impractical number of dots or (if simply capped) show visibly uneven
    // density. Doubling the spacing in graph space instead keeps the
    // on-screen density roughly constant.
    let mut spacing = GRID_BASE_SPACING;
    while spacing * zoom < 16.0 {
        spacing *= 2.0;
    }

    let top_left = screen_to_graph(origin, pan, zoom, canvas_rect.min);
    let bottom_right = screen_to_graph(origin, pan, zoom, canvas_rect.max);
    let start_x = (top_left.x / spacing).floor() as i64;
    let end_x = (bottom_right.x / spacing).ceil() as i64;
    let start_y = (top_left.y / spacing).floor() as i64;
    let end_y = (bottom_right.y / spacing).ceil() as i64;

    let color = crate::theme::TWINKLING_GREY.gamma_multiply(0.15);
    for gy in start_y..=end_y {
        for gx in start_x..=end_x {
            let p = graph_to_screen(
                origin,
                pan,
                zoom,
                Pos2::new(gx as f32 * spacing, gy as f32 * spacing),
            );
            if canvas_rect.contains(p) {
                painter.circle_filled(p, 1.0, color);
            }
        }
    }
}

fn port_color(port_type: PortType) -> Color32 {
    match port_type {
        PortType::Gray => Color32::from_gray(200),
        PortType::Color => crate::theme::PINK_DWARF,
    }
}

/// The node header's title text: the generic `"Output: "` prefix (every
/// Output node def has one — see `mc_nodes::defs`) is swapped for a much
/// shorter `"ON "`, and whatever still doesn't fit the node's fixed width
/// after that is truncated with `"..."` — long display names were bleeding
/// right past the node's own edge, over its neighbors. Applies to any node,
/// not just Output ones, since nothing stops a future node def from having
/// just as long a name.
fn node_title_text(
    painter: &egui::Painter,
    display_name: &str,
    font_id: egui::FontId,
    max_width: f32,
) -> String {
    let shortened = match display_name.strip_prefix("Output: ") {
        Some(rest) => format!("ON {rest}"),
        None => display_name.to_string(),
    };

    let fits = |s: &str| -> bool {
        painter
            .layout_no_wrap(s.to_string(), font_id.clone(), Color32::WHITE)
            .size()
            .x
            <= max_width
    };

    if fits(&shortened) {
        return shortened;
    }

    let mut truncated = shortened;
    while !truncated.is_empty() {
        let candidate = format!("{truncated}...");
        if fits(&candidate) {
            return candidate;
        }
        truncated.pop();
    }
    "...".to_string()
}

/// Node layout constants are defined at `zoom == 1` and scaled by `zoom`
/// everywhere they're used for screen-space geometry — a real proportional
/// zoom, matching every other node editor, rather than only scaling node
/// *positions* and leaving each node's own size fixed (which reads as nodes
/// sliding closer/further apart instead of the view zooming).
fn node_size(def: &mc_nodes::defs::NodeDef, zoom: f32) -> Vec2 {
    let port_rows = def.inputs.len().max(def.outputs.len()).max(1) as f32;
    let height = HEADER_HEIGHT
        + BODY_PADDING
        + port_rows * PORT_ROW_HEIGHT
        + BODY_PADDING
        + THUMBNAIL_SIZE
        + BODY_PADDING;
    Vec2::new(NODE_WIDTH, height) * zoom
}

fn pin_pos(rect: Rect, is_output: bool, index: usize, zoom: f32) -> Pos2 {
    let y =
        rect.min.y + (HEADER_HEIGHT + BODY_PADDING + (index as f32 + 0.5) * PORT_ROW_HEIGHT) * zoom;
    let x = if is_output { rect.max.x } else { rect.min.x };
    Pos2::new(x, y)
}

fn cubic_bezier(p0: Pos2, p1: Pos2, p2: Pos2, p3: Pos2, t: f32) -> Pos2 {
    let mt = 1.0 - t;
    let a = mt * mt * mt;
    let b = 3.0 * mt * mt * t;
    let c = 3.0 * mt * t * t;
    let d = t * t * t;
    Pos2::new(
        a * p0.x + b * p1.x + c * p2.x + d * p3.x,
        a * p0.y + b * p1.y + c * p2.y + d * p3.y,
    )
}

fn wire_points(from: Pos2, to: Pos2) -> Vec<Pos2> {
    let control_offset = ((to.x - from.x).abs() * 0.5).max(40.0);
    let c1 = from + Vec2::new(control_offset, 0.0);
    let c2 = to - Vec2::new(control_offset, 0.0);
    (0..=24)
        .map(|i| cubic_bezier(from, c1, c2, to, i as f32 / 24.0))
        .collect()
}

fn draw_wire(painter: &egui::Painter, from: Pos2, to: Pos2, color: Color32) {
    painter.line(wire_points(from, to), Stroke::new(2.0, color));
}

/// Distance from `p` to the actual (curved) wire, not the straight line
/// between its endpoints — a right-click near a strongly curved wire could
/// be nowhere near that straight chord, which is why disconnect-by-right-
/// click could silently miss.
fn dist_to_wire(p: Pos2, from: Pos2, to: Pos2) -> f32 {
    let points = wire_points(from, to);
    points
        .windows(2)
        .map(|w| dist_to_segment(p, w[0], w[1]))
        .fold(f32::INFINITY, f32::min)
}

fn dist_to_segment(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_sq();
    if len2 <= f32::EPSILON {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    let proj = a + ab * t;
    (p - proj).length()
}

fn node_def_of(graph: &mc_core::Graph, id: NodeId) -> Option<&'static mc_nodes::defs::NodeDef> {
    mc_nodes::defs::node_def(&graph.node(id)?.kind)
}

/// If `node`'s `port_name` input currently has a connection, disconnects it
/// (so the old wire disappears the instant it's grabbed, not lingering until
/// the new connection lands) and returns the *output* pin it was fed from.
///
/// Rewiring an existing connection should keep its output end anchored and
/// let you drag the input end to a new home — not the other way around. So
/// picking up a connected input re-anchors the drag at that fixed output,
/// via the same `PinRef::Output` a fresh drag from that output would use,
/// rather than anchoring at the input pin that was actually clicked.
fn pick_up_connected_input(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    node: NodeId,
    port_name: &str,
) -> Option<PinRef> {
    let connection = graph
        .connections()
        .iter()
        .find(|c| c.to_node == node && c.to_port == port_name)
        .cloned()?;
    let from_index = node_def_of(graph, connection.from_node).and_then(|d| {
        d.outputs
            .iter()
            .position(|p| p.name == connection.from_port)
    })?;
    let _ = history.do_transaction(
        graph,
        vec![mc_core::Command::Disconnect(connection.clone())],
    );
    Some(PinRef::Output {
        node: connection.from_node,
        index: from_index,
    })
}

/// Connects two pins if exactly one is an output and one is an input and
/// their port types are compatible; a no-op otherwise (e.g. dragging between
/// two inputs, or onto nothing).
fn try_connect(graph: &mut mc_core::Graph, history: &mut mc_core::History, a: PinRef, b: PinRef) {
    let (
        PinRef::Output {
            node: from_node,
            index: from_index,
        },
        PinRef::Input {
            node: to_node,
            index: to_index,
        },
    ) = (match (a, b) {
        (PinRef::Output { .. }, PinRef::Input { .. }) => (a, b),
        (PinRef::Input { .. }, PinRef::Output { .. }) => (b, a),
        _ => return,
    })
    else {
        return;
    };

    let Some(from_port) = node_def_of(graph, from_node).and_then(|d| d.outputs.get(from_index))
    else {
        return;
    };
    let Some(to_port) = node_def_of(graph, to_node).and_then(|d| d.inputs.get(to_index)) else {
        return;
    };
    let from_kind = graph
        .node(from_node)
        .map(|n| n.kind.clone())
        .unwrap_or_default();
    let from_type =
        mc_nodes::defs::effective_output_type(graph, from_node, &from_kind, from_port.port_type);
    if !mc_nodes::defs::ports_compatible(from_type, to_port.port_type) {
        return;
    }
    if let Ok(tx) = graph.connect_transaction(from_node, from_port.name, to_node, to_port.name) {
        let _ = history.do_transaction(graph, tx);
    }
}

#[expect(clippy::too_many_arguments)]
pub fn show(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    canvas: &mut CanvasState,
    selected: &mut Option<NodeId>,
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    render_state: &egui_wgpu::RenderState,
    thumbnail_ids: &mut HashMap<NodeId, egui::TextureId>,
) -> CanvasOutput {
    let canvas_rect = ui.available_rect_before_wrap();
    let origin = canvas_rect.min;
    let modifiers = ui.input(|i| i.modifiers);
    let mut output = CanvasOutput {
        open_add_menu_at: None,
        dropped_wire: None,
    };

    // Background interact region first, so it sits "under" the per-node/
    // per-pin regions added later this frame — same z-order-by-add-order
    // egui-snarl itself relied on, and the reason this works without any
    // special-casing: whichever interactive region was added *last* for a
    // given screen position wins.
    let bg_id = ui.id().with("mc_canvas_bg");
    let bg_response = ui.interact(canvas_rect, bg_id, Sense::click_and_drag());

    // Trackpad two-finger scroll and a mouse wheel both land in
    // `smooth_scroll_delta` and pan; pinch (and ctrl/cmd-scroll, which egui
    // already folds into `zoom_delta()` separately from the scroll delta)
    // zooms. Keeping these on separate input signals is what makes a plain
    // trackpad swipe pan instead of zooming — they used to be the same
    // scroll-wheel signal, which is why a swipe looked like "nodes pushing
    // closer together" instead of the view sliding.
    //
    // Gated on the pointer being anywhere inside the canvas *and* this
    // canvas's own layer being the topmost one there — not on
    // `bg_response.hovered()` (a node's own interact region, added later,
    // z-order-wins the hover over the background's, which used to make
    // pinch-zoom and pan silently stop working the moment the cursor
    // crossed over a node), but also not on `canvas_rect.contains(pointer)`
    // alone (that doesn't know about a foreground popup like the add-node
    // menu sitting on top — its own scroll area would otherwise fight the
    // canvas for the same scroll gesture, with the canvas panning *behind*
    // the popup at the same time the popup scrolled its list).
    let canvas_layer = ui.layer_id();
    if let Some(pointer) = ui.input(|i| i.pointer.hover_pos()) {
        if canvas_rect.contains(pointer) && ui.ctx().layer_id_at(pointer) == Some(canvas_layer) {
            let zoom_delta = ui.input(|i| i.zoom_delta());
            if zoom_delta != 1.0 {
                let graph_pt = screen_to_graph(origin, canvas.pan, canvas.zoom, pointer);
                canvas.zoom = (canvas.zoom * zoom_delta).clamp(0.2, 3.0);
                canvas.pan = graph_pt.to_vec2() - (pointer - origin) / canvas.zoom;
            }
            let scroll = ui.input(|i| i.smooth_scroll_delta);
            if scroll != Vec2::ZERO {
                canvas.pan -= scroll / canvas.zoom;
            }
        }
    }

    let painter = ui.painter_at(canvas_rect);
    draw_dot_grid(&painter, canvas_rect, origin, canvas.pan, canvas.zoom);

    // Pass 1: node screen rects (read-only — needed before any drawing so
    // wires can be drawn *behind* nodes).
    let mut node_rects: HashMap<NodeId, Rect> = HashMap::new();
    for (id, node) in graph.nodes() {
        let Some(def) = mc_nodes::defs::node_def(&node.kind) else {
            continue;
        };
        let pos = graph_to_screen(
            origin,
            canvas.pan,
            canvas.zoom,
            Pos2::new(node.position[0], node.position[1]),
        );
        node_rects.insert(id, Rect::from_min_size(pos, node_size(def, canvas.zoom)));
    }

    // Pass 2: pin positions (read-only), used for both wire endpoints and
    // wire-drag hit-testing.
    let mut pins: Vec<PinInfo> = Vec::new();
    for (id, node) in graph.nodes() {
        let Some(def) = mc_nodes::defs::node_def(&node.kind) else {
            continue;
        };
        let Some(&rect) = node_rects.get(&id) else {
            continue;
        };
        for (i, port) in def.outputs.iter().enumerate() {
            let port_type =
                mc_nodes::defs::effective_output_type(graph, id, &node.kind, port.port_type);
            pins.push(PinInfo {
                r: PinRef::Output { node: id, index: i },
                pos: pin_pos(rect, true, i, canvas.zoom),
                port_type,
            });
        }
        for (i, port) in def.inputs.iter().enumerate() {
            pins.push(PinInfo {
                r: PinRef::Input { node: id, index: i },
                pos: pin_pos(rect, false, i, canvas.zoom),
                port_type: port.port_type,
            });
        }
    }

    // Pass 3: wires (drawn before nodes so node bodies paint on top of the
    // wire ends, not the other way around).
    let mut wires: Vec<(mc_core::Connection, Pos2, Pos2)> = Vec::new();
    for connection in graph.connections() {
        let from_index = node_def_of(graph, connection.from_node).and_then(|d| {
            d.outputs
                .iter()
                .position(|p| p.name == connection.from_port)
        });
        let to_index = node_def_of(graph, connection.to_node)
            .and_then(|d| d.inputs.iter().position(|p| p.name == connection.to_port));
        let (Some(fi), Some(ti)) = (from_index, to_index) else {
            continue;
        };
        let from_pos = pins
            .iter()
            .find(|p| {
                p.r == PinRef::Output {
                    node: connection.from_node,
                    index: fi,
                }
            })
            .map(|p| p.pos);
        let to_pos = pins
            .iter()
            .find(|p| {
                p.r == PinRef::Input {
                    node: connection.to_node,
                    index: ti,
                }
            })
            .map(|p| p.pos);
        if let (Some(fp), Some(tp)) = (from_pos, to_pos) {
            wires.push((connection.clone(), fp, tp));
        }
    }
    for (connection, from_pos, to_pos) in &wires {
        let color = node_def_of(graph, connection.from_node)
            .and_then(|d| d.output(&connection.from_port))
            .map_or(Color32::from_gray(160), |p| port_color(p.port_type));
        draw_wire(&painter, *from_pos, *to_pos, color);
    }

    // Pass 4: nodes themselves — drawing, selection, dragging, pins.
    let node_ids: Vec<NodeId> = graph.nodes().map(|(id, _)| id).collect();
    for id in node_ids {
        let Some(def) = node_def_of(graph, id) else {
            continue;
        };
        let Some(&rect) = node_rects.get(&id) else {
            continue;
        };
        let is_selected = canvas.selected.contains(&id);

        let node_id = ui.id().with(("mc_node", id));
        let node_response = ui.interact(rect, node_id, Sense::click_and_drag());

        if node_response.drag_started_by(egui::PointerButton::Primary) && !is_selected {
            canvas.selected.clear();
            canvas.selected.insert(id);
            *selected = Some(id);
        }
        if node_response.dragged_by(egui::PointerButton::Primary) {
            let delta = node_response.drag_delta() / canvas.zoom;
            let targets: Vec<NodeId> = if canvas.selected.contains(&id) {
                canvas.selected.iter().copied().collect()
            } else {
                vec![id]
            };
            for target in targets {
                if let Some(pos) = graph.node(target).map(|n| n.position) {
                    let _ = graph.set_node_position(target, [pos[0] + delta.x, pos[1] + delta.y]);
                }
            }
        }
        if node_response.clicked_by(egui::PointerButton::Primary) {
            if modifiers.shift {
                if !canvas.selected.remove(&id) {
                    canvas.selected.insert(id);
                }
            } else {
                canvas.selected.clear();
                canvas.selected.insert(id);
                *selected = Some(id);
            }
        }
        node_response.context_menu(|ui| {
            if ui.button("Delete Node").clicked() {
                let transaction = graph.remove_node_transaction(id);
                let _ = history.do_transaction(graph, transaction);
                canvas.selected.remove(&id);
                if *selected == Some(id) {
                    *selected = None;
                }
                ui.close();
            }
        });

        let zoom = canvas.zoom;
        let is_selected = canvas.selected.contains(&id);
        let fill = crate::theme::WORMHOLE_GREY;
        let stroke = if is_selected {
            Stroke::new(1.0, crate::theme::SUPERNOVA_LIGHT)
        } else {
            Stroke::new(1.0, crate::theme::TWINKLING_GREY)
        };
        painter.rect(rect, 4.0 * zoom, fill, stroke, StrokeKind::Inside);
        painter.line_segment(
            [
                Pos2::new(rect.min.x, rect.min.y + HEADER_HEIGHT * zoom),
                Pos2::new(rect.max.x, rect.min.y + HEADER_HEIGHT * zoom),
            ],
            Stroke::new(1.0, crate::theme::TWINKLING_GREY.gamma_multiply(0.5)),
        );
        let title_font = egui::FontId::proportional(14.0 * zoom);
        let title_max_width = (NODE_WIDTH - HEADER_TEXT_LEFT - HEADER_TEXT_RIGHT_RESERVED) * zoom;
        let title_text = node_title_text(
            &painter,
            def.display_name,
            title_font.clone(),
            title_max_width,
        );
        painter.text(
            Pos2::new(
                rect.min.x + HEADER_TEXT_LEFT * zoom,
                rect.min.y + HEADER_HEIGHT * zoom / 2.0,
            ),
            egui::Align2::LEFT_CENTER,
            title_text,
            title_font,
            crate::theme::SUPERNOVA_LIGHT,
        );

        // Delete indicator: same idea as the preview dot below, for the
        // right-click-menu "Delete Node" action — a one-click way to remove
        // a single node without having to select it first. Rightmost of the
        // two header dots.
        let delete_dot_pos = Pos2::new(
            rect.max.x - PREVIEW_DOT_MARGIN * zoom,
            rect.min.y + HEADER_HEIGHT * zoom / 2.0,
        );
        painter.circle_filled(
            delete_dot_pos,
            PREVIEW_DOT_RADIUS * zoom,
            Color32::from_rgb(210, 70, 70),
        );
        let delete_dot_id = ui.id().with(("mc_delete_dot", id));
        let delete_dot_rect =
            Rect::from_center_size(delete_dot_pos, Vec2::splat(PIN_HIT_RADIUS * zoom * 2.0));
        if ui
            .interact(delete_dot_rect, delete_dot_id, Sense::click())
            .clicked()
        {
            let transaction = graph.remove_node_transaction(id);
            let _ = history.do_transaction(graph, transaction);
            canvas.selected.remove(&id);
            if *selected == Some(id) {
                *selected = None;
            }
        }

        // Preview-target indicator: a small dot in the header, rather than a
        // full "Preview" button in the body — frees up the whole body for
        // the thumbnail, and sidesteps real egui widgets (which don't scale
        // with our zoom) inside the node entirely. Left of the delete dot.
        let preview_dot_pos = delete_dot_pos - Vec2::new(DOT_SPACING * zoom, 0.0);
        painter.circle_filled(
            preview_dot_pos,
            PREVIEW_DOT_RADIUS * zoom,
            if *selected == Some(id) {
                crate::theme::SUPERNOVA_LIGHT
            } else {
                crate::theme::TWINKLING_GREY.gamma_multiply(0.6)
            },
        );
        let preview_dot_id = ui.id().with(("mc_preview_dot", id));
        let preview_dot_rect =
            Rect::from_center_size(preview_dot_pos, Vec2::splat(PIN_HIT_RADIUS * zoom * 2.0));
        if ui
            .interact(preview_dot_rect, preview_dot_id, Sense::click())
            .clicked()
        {
            *selected = Some(id);
        }

        for (i, port) in def.inputs.iter().enumerate() {
            let p = pin_pos(rect, false, i, zoom);
            draw_pin(
                ui,
                &painter,
                p,
                port.port_type,
                port.display_name,
                false,
                zoom,
            );
            let pin_id = ui.id().with(("mc_pin_in", id, i));
            let pin_rect = Rect::from_center_size(p, Vec2::splat(PIN_HIT_RADIUS * zoom * 2.0));
            let pin_resp = ui.interact(pin_rect, pin_id, Sense::click_and_drag());
            if pin_resp.drag_started_by(egui::PointerButton::Primary) {
                // If this input is already wired, re-anchor the drag at the
                // output it came from (see `pick_up_connected_input`) so
                // that end stays fixed while this one follows the cursor to
                // a new input. An empty input just starts a fresh drag from
                // itself, looking for an output, as before.
                canvas.wire_drag = pick_up_connected_input(graph, history, id, port.name)
                    .or(Some(PinRef::Input { node: id, index: i }));
            }
        }
        let node_kind = graph.node(id).map(|n| n.kind.clone()).unwrap_or_default();
        for (i, port) in def.outputs.iter().enumerate() {
            let p = pin_pos(rect, true, i, zoom);
            let port_type =
                mc_nodes::defs::effective_output_type(graph, id, &node_kind, port.port_type);
            draw_pin(ui, &painter, p, port_type, port.display_name, true, zoom);
            let pin_id = ui.id().with(("mc_pin_out", id, i));
            let pin_rect = Rect::from_center_size(p, Vec2::splat(PIN_HIT_RADIUS * zoom * 2.0));
            let pin_resp = ui.interact(pin_rect, pin_id, Sense::click_and_drag());
            if pin_resp.drag_started_by(egui::PointerButton::Primary) {
                canvas.wire_drag = Some(PinRef::Output { node: id, index: i });
            }
        }

        // Thumbnail: drawn straight onto the canvas via the painter, not a
        // real egui `Image` widget — a plain widget's layout doesn't know
        // about our zoom, which is exactly what made the old "Preview"
        // button (and, before that, this same image) overflow or get
        // clipped at low zoom.
        let port_rows = def.inputs.len().max(def.outputs.len()).max(1) as f32;
        let body_top = rect.min.y
            + (HEADER_HEIGHT + BODY_PADDING + port_rows * PORT_ROW_HEIGHT + BODY_PADDING) * zoom;
        let thumbnail_rect = Rect::from_min_size(
            Pos2::new(rect.min.x + BODY_PADDING * zoom, body_top),
            Vec2::splat(THUMBNAIL_SIZE * zoom),
        );
        if let Some(key) = mc_nodes::registry::resolve_texture_key(graph, id) {
            if let Some(view) = pool.view(&key).cloned() {
                let texture_id = match thumbnail_ids.get(&id).copied() {
                    Some(tex_id) => {
                        render_state
                            .renderer
                            .write()
                            .update_egui_texture_from_wgpu_texture(
                                &gpu.device,
                                &view,
                                wgpu::FilterMode::Linear,
                                tex_id,
                            );
                        tex_id
                    }
                    None => {
                        let tex_id = render_state.renderer.write().register_native_texture(
                            &gpu.device,
                            &view,
                            wgpu::FilterMode::Linear,
                        );
                        thumbnail_ids.insert(id, tex_id);
                        tex_id
                    }
                };
                painter.image(
                    texture_id,
                    thumbnail_rect,
                    Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            }
        }
    }

    // Pass 5: in-progress wire-drag — live preview plus release handling.
    if let Some(from) = canvas.wire_drag {
        let current = ui.input(|i| i.pointer.interact_pos().or(i.pointer.hover_pos()));
        if let (Some(from_pin), Some(cur)) = (pins.iter().find(|p| p.r == from), current) {
            draw_wire(&painter, from_pin.pos, cur, port_color(from_pin.port_type));
        }
        if ui.input(|i| i.pointer.primary_released()) {
            if let Some(cur) = current {
                let target = pins
                    .iter()
                    .filter(|p| p.r != from)
                    .min_by(|a, b| a.pos.distance(cur).total_cmp(&b.pos.distance(cur)));
                let connected =
                    target.is_some_and(|t| t.pos.distance(cur) <= PIN_HIT_RADIUS * canvas.zoom);
                if let Some(target) = target.filter(|_| connected) {
                    try_connect(graph, history, from, target.r);
                } else if let Some(from_pin) = pins.iter().find(|p| p.r == from) {
                    let (from_node, from_port_index, from_is_output) = match from {
                        PinRef::Output { node, index } => (node, index, true),
                        PinRef::Input { node, index } => (node, index, false),
                    };
                    output.dropped_wire = Some(DroppedWire {
                        screen_pos: cur,
                        graph_pos: screen_to_graph(origin, canvas.pan, canvas.zoom, cur),
                        from_node,
                        from_port_index,
                        from_is_output,
                        from_port_type: from_pin.port_type,
                    });
                }
            }
            canvas.wire_drag = None;
        }
    }

    // Pass 6: background gestures — pan, box-select, click-to-clear,
    // Cmd+drag-to-rewire, right-click (wire disconnect or add-node menu).

    // Cmd+drag starting on an existing wire picks its input end back up —
    // an alternative to grabbing the (small) input pin directly, for
    // relocating a connection without having to aim precisely at the pin.
    if modifiers.command
        && canvas.wire_drag.is_none()
        && bg_response.drag_started_by(egui::PointerButton::Primary)
    {
        if let Some(start) = bg_response.interact_pointer_pos() {
            if let Some((connection, _, _)) = wires
                .iter()
                .find(|(_, a, b)| dist_to_wire(start, *a, *b) <= WIRE_HIT_DISTANCE)
            {
                let to_node = connection.to_node;
                let to_port = connection.to_port.clone();
                canvas.wire_drag = pick_up_connected_input(graph, history, to_node, &to_port);
            }
        }
    }

    if modifiers.shift {
        if bg_response.drag_started_by(egui::PointerButton::Primary) {
            canvas.box_select_start = bg_response.interact_pointer_pos();
        }
    } else if canvas.wire_drag.is_none() && bg_response.dragged_by(egui::PointerButton::Primary) {
        canvas.pan -= bg_response.drag_delta() / canvas.zoom;
    }

    if let Some(start) = canvas.box_select_start {
        if let Some(cur) = bg_response.interact_pointer_pos() {
            let select_rect = Rect::from_two_pos(start, cur);
            let accent = crate::theme::SUPERNOVA_LIGHT;
            painter.rect(
                select_rect,
                0.0,
                Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 40),
                Stroke::new(1.0, accent),
                StrokeKind::Inside,
            );
        }
        if bg_response.drag_stopped_by(egui::PointerButton::Primary) {
            if let Some(cur) = bg_response.interact_pointer_pos() {
                let select_rect = Rect::from_two_pos(start, cur);
                for (&id, &rect) in &node_rects {
                    if select_rect.intersects(rect) {
                        canvas.selected.insert(id);
                    }
                }
            }
            canvas.box_select_start = None;
        }
    }

    // A plain left click (no drag, no modifiers) disconnects a wire if it
    // landed on one, or clears the selection otherwise. Previously this was
    // on right-click, which turned out not to register reliably; left-click
    // reuses the exact same `clicked_by(Primary)` path that click-to-clear
    // already used and was confirmed working, and reserves right-click
    // purely for the add-node menu so it can never compete with the
    // Cmd+drag-to-rewire gesture below.
    if bg_response.clicked_by(egui::PointerButton::Primary) && !modifiers.shift {
        let hit_wire = bg_response.interact_pointer_pos().and_then(|screen_pos| {
            wires
                .iter()
                .find(|(_, a, b)| dist_to_wire(screen_pos, *a, *b) <= WIRE_HIT_DISTANCE)
        });
        if let Some((connection, _, _)) = hit_wire {
            let _ = history.do_transaction(
                graph,
                vec![mc_core::Command::Disconnect(connection.clone())],
            );
        } else {
            canvas.selected.clear();
        }
    }

    if bg_response.secondary_clicked() {
        if let Some(screen_pos) = bg_response.interact_pointer_pos() {
            let graph_pos = screen_to_graph(origin, canvas.pan, canvas.zoom, screen_pos);
            output.open_add_menu_at = Some((screen_pos, graph_pos));
        }
    }

    output
}

fn draw_pin(
    ui: &egui::Ui,
    painter: &egui::Painter,
    pos: Pos2,
    port_type: PortType,
    label: &str,
    is_output: bool,
    zoom: f32,
) {
    painter.circle_filled(pos, PIN_RADIUS * zoom, port_color(port_type));
    let offset = (PIN_RADIUS + 4.0) * zoom;
    let (anchor, text_pos) = if is_output {
        (egui::Align2::RIGHT_CENTER, pos - Vec2::new(offset, 0.0))
    } else {
        (egui::Align2::LEFT_CENTER, pos + Vec2::new(offset, 0.0))
    };
    painter.text(
        text_pos,
        anchor,
        label,
        egui::FontId::proportional(12.0 * zoom),
        ui.visuals().text_color(),
    );
}

/// Connects a freshly-added `new_node` back to the pin a [`DroppedWire`]
/// started from, picking the first port on `new_node` compatible with it —
/// the "drag a wire out, drop it on nothing, pick a node, it's already
/// wired up" gesture.
pub fn connect_dropped_wire(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    dropped: &DroppedWire,
    new_node: NodeId,
) {
    let Some(new_def) = node_def_of(graph, new_node) else {
        return;
    };
    if dropped.from_is_output {
        let Some(from_port) = node_def_of(graph, dropped.from_node)
            .and_then(|d| d.outputs.get(dropped.from_port_index))
        else {
            return;
        };
        let Some(to_port) = new_def
            .inputs
            .iter()
            .find(|p| mc_nodes::defs::ports_compatible(dropped.from_port_type, p.port_type))
        else {
            return;
        };
        if let Ok(tx) =
            graph.connect_transaction(dropped.from_node, from_port.name, new_node, to_port.name)
        {
            let _ = history.do_transaction(graph, tx);
        }
    } else {
        let Some(to_port) = node_def_of(graph, dropped.from_node)
            .and_then(|d| d.inputs.get(dropped.from_port_index))
        else {
            return;
        };
        let Some(from_port) = new_def
            .outputs
            .iter()
            .find(|p| mc_nodes::defs::ports_compatible(p.port_type, dropped.from_port_type))
        else {
            return;
        };
        if let Ok(tx) =
            graph.connect_transaction(new_node, from_port.name, dropped.from_node, to_port.name)
        {
            let _ = history.do_transaction(graph, tx);
        }
    }
}
