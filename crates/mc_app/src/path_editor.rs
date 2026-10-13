// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Path node's own interactive overlay in the 2D View — drawing the
//! curve on top of its Preview Mask, and letting the artist select and
//! drag its points and handles directly, or draw brand new ones with Pen
//! mode. PATH_NODE_SPEC.md §4 is the source of truth for the full editor;
//! this covers: click/shift-click select, drag a point, drag a handle
//! (respecting its kind's constraint via `mc_nodes::path::drag_handle`),
//! Delete/Backspace to remove the selection, and Pen mode (P to enter,
//! click empty space to append a point, click-and-drag to append one with
//! a dragged-out symmetric handle, click the first point to close, Esc to
//! leave). Double-click kind toggling, the right-click menu, Alt-drag
//! break-link, click-on-segment insert, SVG paste, and wrap ghost copies
//! are still to come.
//!
//! Shares the 2D View's existing pan-drag `egui::Response` rather than
//! opening a second `Sense::click_and_drag()` region over the same rect —
//! two independent interact regions on one area would race for which one
//! "wins" a given frame's pointer input.

use std::collections::{HashMap, HashSet};

use mc_core::{ParamValue, Point, PointKind, SubPath};

/// Constant on-screen size for every point/handle marker and the pick
/// radius around them, regardless of the 2D View's own zoom (it doesn't
/// have one — just a small nudge-pan — so this never needs to divide by a
/// zoom factor the way the Node Graph's own picking does).
const POINT_RADIUS: f32 = 4.0;
const HANDLE_RADIUS: f32 = 3.0;
const PICK_RADIUS: f32 = 8.0;
/// How densely to sample each curve segment for drawing — cheap and
/// smooth-looking is all this needs, same reasoning as
/// `mc_nodes::path::dispatch_preview_mask`'s own sample count.
const DRAW_SAMPLES_PER_SEGMENT: usize = 24;
/// Closing a Pen-mode subpath needs at least a triangle — closing a
/// 2-point subpath on itself would just be the same single segment drawn
/// twice.
const MIN_POINTS_TO_CLOSE: usize = 3;

/// `(subpath index, point index)` — identifies one point within whichever
/// Path node is currently selected.
type PointKey = (usize, usize);

#[derive(Clone, Copy)]
enum DragTarget {
    Point(PointKey),
    Handle(PointKey, mc_nodes::path::HandleSide),
}

impl DragTarget {
    fn key(self) -> PointKey {
        match self {
            DragTarget::Point(k) | DragTarget::Handle(k, _) => k,
        }
    }
}

struct ActiveDrag {
    target: DragTarget,
    /// What live-dragging re-applies the raw pointer delta onto every
    /// frame (fresh each time, not incrementally, so there's no drift).
    /// Equal to `old_path` except right after Pen mode's click-and-drag
    /// gesture, which has already inserted the brand new point here (so
    /// there's something for `target` to refer to) but *not* in
    /// `old_path` — so Undo still reverts the whole gesture, insertion
    /// included.
    base_path: mc_core::Path,
    /// Snapshot of the whole `Path` value from just before this drag
    /// gesture started — restored as one undo step on `drag_stopped`, same
    /// pattern as `params_panel::commit_live`'s `DragState`, just keyed by
    /// a point/handle instead of a `(NodeId, param name)` pair since this
    /// editor only ever touches one node's one `"path"` param.
    old_path: mc_core::Path,
}

/// Per-session editor state for whichever Path node is currently selected
/// — selection and an in-progress drag, not part of the graph itself (the
/// same split `TwoDViewState` already makes for pan/tiling).
#[derive(Default)]
pub struct PathEditorState {
    selected: HashSet<PointKey>,
    drag: Option<ActiveDrag>,
    pen_mode: bool,
    /// Which subpath Pen mode is currently appending to — `None` means the
    /// next empty-space click/drag starts a brand new subpath.
    pen_subpath: Option<usize>,
}

impl PathEditorState {
    pub fn pen_mode(&self) -> bool {
        self.pen_mode
    }

    /// Enter or leave Pen mode. Entering with exactly one point selected —
    /// and it's the last point of an open subpath — resumes drawing that
    /// subpath instead of starting a fresh one on the next click
    /// (PATH_NODE_SPEC.md §4). Leaving always clears the selection too,
    /// same as pressing Esc.
    fn set_pen_mode(&mut self, path: &mc_core::Path, enabled: bool) {
        self.pen_mode = enabled;
        if enabled {
            self.pen_subpath = single_selected_point(&self.selected)
                .filter(|&key| is_last_point_of_an_open_subpath(path, key))
                .map(|(sp_i, _)| sp_i);
        } else {
            self.pen_subpath = None;
            self.selected.clear();
        }
    }

    /// Flips Pen mode on/off for whichever Path node `node_id` refers to —
    /// used by the 2D View's own Pen toggle button, which (unlike `show`'s
    /// own P-key handler) doesn't already have that frame's `Path` value in
    /// hand.
    pub fn toggle_pen_mode(&mut self, graph: &mc_core::Graph, node_id: mc_core::NodeId) {
        let Some(path) = graph
            .node(node_id)
            .and_then(|n| n.params.get("path"))
            .and_then(ParamValue::as_path)
            .cloned()
        else {
            return;
        };
        let enable = !self.pen_mode;
        self.set_pen_mode(&path, enable);
    }
}

fn uv_to_screen(uv: [f64; 2], image_rect: egui::Rect) -> egui::Pos2 {
    egui::pos2(
        image_rect.min.x + uv[0] as f32 * image_rect.width(),
        image_rect.min.y + uv[1] as f32 * image_rect.height(),
    )
}

fn screen_to_uv(p: egui::Pos2, image_rect: egui::Rect) -> [f64; 2] {
    [
        ((p.x - image_rect.min.x) / image_rect.width()) as f64,
        ((p.y - image_rect.min.y) / image_rect.height()) as f64,
    ]
}

fn single_selected_point(selected: &HashSet<PointKey>) -> Option<PointKey> {
    let mut it = selected.iter();
    let only = *it.next()?;
    if it.next().is_some() {
        None
    } else {
        Some(only)
    }
}

/// Nearest handle (of an already-selected point) or point within
/// `PICK_RADIUS` of `screen_pos`, handles taking priority per
/// PATH_NODE_SPEC.md §4's own stated hit-test order. A handle sitting
/// exactly on its own point (zero offset — the common case for a freshly
/// placed corner point) is skipped entirely, since hit-testing it there
/// would make that point impossible to grab on its own.
fn hit_test(
    path: &mc_core::Path,
    image_rect: egui::Rect,
    screen_pos: egui::Pos2,
    selected: &HashSet<PointKey>,
) -> Option<DragTarget> {
    fn consider(best: &mut Option<(f32, DragTarget)>, dist: f32, target: DragTarget) {
        if dist <= PICK_RADIUS && best.is_none_or(|(best_dist, _)| dist < best_dist) {
            *best = Some((dist, target));
        }
    }

    let mut best: Option<(f32, DragTarget)> = None;
    for &(sp_i, pt_i) in selected {
        let Some(pt) = path.subpaths.get(sp_i).and_then(|sp| sp.points.get(pt_i)) else {
            continue;
        };
        for (side, offset) in [
            (mc_nodes::path::HandleSide::In, pt.h_in),
            (mc_nodes::path::HandleSide::Out, pt.h_out),
        ] {
            if offset == [0.0, 0.0] {
                continue;
            }
            let world = [pt.pos[0] + offset[0], pt.pos[1] + offset[1]];
            let dist = uv_to_screen(world, image_rect).distance(screen_pos);
            consider(&mut best, dist, DragTarget::Handle((sp_i, pt_i), side));
        }
    }
    if best.is_some() {
        return best.map(|(_, t)| t);
    }

    for (sp_i, sp) in path.subpaths.iter().enumerate() {
        for (pt_i, pt) in sp.points.iter().enumerate() {
            let dist = uv_to_screen(pt.pos, image_rect).distance(screen_pos);
            consider(&mut best, dist, DragTarget::Point((sp_i, pt_i)));
        }
    }
    best.map(|(_, t)| t)
}

fn apply_selection(ui: &egui::Ui, state: &mut PathEditorState, target: Option<DragTarget>) {
    let shift = ui.input(|i| i.modifiers.shift);
    match target.map(DragTarget::key) {
        Some(key) => {
            if shift {
                if !state.selected.remove(&key) {
                    state.selected.insert(key);
                }
            } else if !state.selected.contains(&key) {
                state.selected.clear();
                state.selected.insert(key);
            }
            // Clicking an already-selected point with no modifier keeps
            // the rest of the selection, so a follow-up Delete still
            // removes the whole group.
        }
        None => {
            if !shift {
                state.selected.clear();
            }
        }
    }
}

fn apply_drag(path: &mut mc_core::Path, target: DragTarget, new_uv: [f64; 2]) {
    match target {
        DragTarget::Point((sp_i, pt_i)) => {
            if let Some(pt) = path
                .subpaths
                .get_mut(sp_i)
                .and_then(|sp| sp.points.get_mut(pt_i))
            {
                pt.pos = new_uv;
            }
        }
        DragTarget::Handle((sp_i, pt_i), side) => {
            let Some(sp) = path.subpaths.get_mut(sp_i) else {
                return;
            };
            let Some(pt) = sp.points.get(pt_i).copied() else {
                return;
            };
            let offset = [new_uv[0] - pt.pos[0], new_uv[1] - pt.pos[1]];
            // Alt/Option-drag break-link (the `break_link` arg below) is
            // part of the deferred right-click-menu-adjacent work, not
            // this slice.
            sp.points[pt_i] = mc_nodes::path::drag_handle(&pt, side, offset, false);
        }
    }
}

fn commit_path_change(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    node_id: mc_core::NodeId,
    old_path: mc_core::Path,
    new_path: mc_core::Path,
) {
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
}

/// Live-applies the in-progress drag every frame it moves, re-deriving
/// from `active.base_path` each time rather than mutating incrementally.
fn update_dragged(
    graph: &mut mc_core::Graph,
    node_id: mc_core::NodeId,
    active: &ActiveDrag,
    image_rect: egui::Rect,
    cur: egui::Pos2,
) {
    let mut new_path = active.base_path.clone();
    apply_drag(&mut new_path, active.target, screen_to_uv(cur, image_rect));
    let _ = graph.set_param(node_id, "path", ParamValue::Path(new_path));
}

/// Ends the current drag gesture, recording one undo step back to
/// `active.old_path` — shared by both ordinary handle/point dragging and
/// Pen mode's click-and-drag point creation.
fn finish_drag(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    node_id: mc_core::NodeId,
    state: &mut PathEditorState,
) {
    if let Some(active) = state.drag.take() {
        if let Some(current) = graph
            .node(node_id)
            .and_then(|n| n.params.get("path"))
            .and_then(ParamValue::as_path)
            .cloned()
        {
            commit_path_change(graph, history, node_id, active.old_path, current);
        }
    }
}

fn delete_selected(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    node_id: mc_core::NodeId,
    old_path: &mc_core::Path,
    state: &mut PathEditorState,
) {
    let mut by_subpath: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(sp_i, pt_i) in &state.selected {
        by_subpath.entry(sp_i).or_default().push(pt_i);
    }

    let mut new_path = old_path.clone();
    for (sp_i, mut point_indices) in by_subpath {
        // Highest index first so deleting one doesn't shift the next
        // index-to-delete out from under it.
        point_indices.sort_unstable_by(|a, b| b.cmp(a));
        let Some(mut sp) = new_path.subpaths.get(sp_i).cloned() else {
            continue;
        };
        for pt_i in point_indices {
            if pt_i < sp.points.len() {
                sp = mc_nodes::path::delete_point(&sp, pt_i);
            }
        }
        new_path.subpaths[sp_i] = sp;
    }
    // A subpath with fewer than 2 points left isn't a usable curve —
    // `mc_nodes::path::delete_point`'s own doc comment calls this the
    // caller's concern.
    new_path.subpaths.retain(|sp| sp.points.len() >= 2);

    state.selected.clear();
    commit_path_change(graph, history, node_id, old_path.clone(), new_path);
}

/// True if `key` is the last point of an open subpath in `path` — the
/// condition under which entering Pen mode resumes drawing that subpath
/// instead of starting a fresh one (PATH_NODE_SPEC.md §4: "append... to
/// the end of the selected open subpath, or start a new subpath").
fn is_last_point_of_an_open_subpath(path: &mc_core::Path, key: PointKey) -> bool {
    let (sp_i, pt_i) = key;
    path.subpaths
        .get(sp_i)
        .is_some_and(|sp| !sp.closed && pt_i + 1 == sp.points.len())
}

/// Pen mode's plain click (no drag): close the active subpath if this
/// landed on its first point, otherwise append a zero-handle corner point
/// — starting a brand new subpath first if there's no active one.
fn pen_click(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    node_id: mc_core::NodeId,
    state: &mut PathEditorState,
    path: &mc_core::Path,
    image_rect: egui::Rect,
    screen_pos: egui::Pos2,
) {
    let uv = screen_to_uv(screen_pos, image_rect);

    if let Some(sp_i) = state.pen_subpath {
        if path.subpaths.get(sp_i).is_some() {
            if closes_active_subpath(path, sp_i, image_rect, screen_pos) {
                let mut new_path = path.clone();
                new_path.subpaths[sp_i].closed = true;
                state.pen_subpath = None;
                state.selected.clear();
                commit_path_change(graph, history, node_id, path.clone(), new_path);
                return;
            }
            let mut new_path = path.clone();
            new_path.subpaths[sp_i].points.push(Point::new(uv));
            let new_index = new_path.subpaths[sp_i].points.len() - 1;
            state.selected = HashSet::from([(sp_i, new_index)]);
            commit_path_change(graph, history, node_id, path.clone(), new_path);
            return;
        }
        // The active subpath vanished from under us (e.g. an Undo while
        // Pen mode stayed open) — fall through and start a fresh one.
        state.pen_subpath = None;
    }

    let mut new_path = path.clone();
    new_path
        .subpaths
        .push(SubPath::new(vec![Point::new(uv)], false));
    let sp_i = new_path.subpaths.len() - 1;
    state.pen_subpath = Some(sp_i);
    state.selected = HashSet::from([(sp_i, 0)]);
    commit_path_change(graph, history, node_id, path.clone(), new_path);
}

fn closes_active_subpath(
    path: &mc_core::Path,
    sp_i: usize,
    image_rect: egui::Rect,
    screen_pos: egui::Pos2,
) -> bool {
    let Some(sp) = path.subpaths.get(sp_i) else {
        return false;
    };
    if sp.points.len() < MIN_POINTS_TO_CLOSE {
        return false;
    }
    let Some(first) = sp.points.first() else {
        return false;
    };
    uv_to_screen(first.pos, image_rect).distance(screen_pos) <= PICK_RADIUS
}

fn draw(
    ui: &egui::Ui,
    image_rect: egui::Rect,
    path: &mc_core::Path,
    selected: &HashSet<PointKey>,
    pen_mode: bool,
) {
    let painter = ui.painter();
    let curve_stroke = egui::Stroke::new(1.5, crate::theme::PATH_CYAN);
    let handle_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(190));
    let point_outline = egui::Stroke::new(1.0, egui::Color32::BLACK);

    for sp in &path.subpaths {
        let samples = mc_nodes::path::preview_polyline(sp, DRAW_SAMPLES_PER_SEGMENT);
        let screen_points: Vec<egui::Pos2> = samples
            .iter()
            .map(|&uv| uv_to_screen(uv, image_rect))
            .collect();
        if screen_points.len() >= 2 {
            painter.add(egui::Shape::line(screen_points, curve_stroke));
        }
    }

    for (sp_i, sp) in path.subpaths.iter().enumerate() {
        for (pt_i, pt) in sp.points.iter().enumerate() {
            let is_selected = selected.contains(&(sp_i, pt_i));
            let screen = uv_to_screen(pt.pos, image_rect);

            if is_selected {
                for offset in [pt.h_in, pt.h_out] {
                    if offset == [0.0, 0.0] {
                        continue;
                    }
                    let handle_screen =
                        uv_to_screen([pt.pos[0] + offset[0], pt.pos[1] + offset[1]], image_rect);
                    painter.line_segment([screen, handle_screen], handle_stroke);
                    painter.circle_filled(handle_screen, HANDLE_RADIUS, handle_stroke.color);
                }
            }

            let fill = if is_selected {
                crate::theme::SUPERNOVA_LIGHT
            } else {
                egui::Color32::WHITE
            };
            painter.circle_filled(screen, POINT_RADIUS, fill);
            painter.circle_stroke(screen, POINT_RADIUS, point_outline);
        }
    }

    if pen_mode {
        painter.text(
            image_rect.left_top() + egui::vec2(6.0, 6.0),
            egui::Align2::LEFT_TOP,
            "Pen mode — click to add points, click the first point to close, Esc to exit",
            egui::FontId::proportional(12.0),
            crate::theme::SUPERNOVA_LIGHT,
        );
    }
}

/// Draws the current Path and handles select/drag/delete/Pen-mode for it.
/// Called once per frame the 2D View is showing a selected Path node's
/// single (non-tiled) preview — `response` is the same drag-sense response
/// the 2D View already created for panning, reused rather than duplicated
/// so the two don't race for the same pointer input; the caller is
/// responsible for *not* also applying it as a pan while a path node is
/// selected.
pub fn show(
    ui: &egui::Ui,
    image_rect: egui::Rect,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    node_id: mc_core::NodeId,
    state: &mut PathEditorState,
    response: &egui::Response,
) {
    let Some(path) = graph
        .node(node_id)
        .and_then(|n| n.params.get("path"))
        .and_then(ParamValue::as_path)
        .cloned()
    else {
        return;
    };

    if response.hovered() {
        if ui.input(|i| i.key_pressed(egui::Key::P)) {
            let enable = !state.pen_mode;
            state.set_pen_mode(&path, enable);
        }
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            state.set_pen_mode(&path, false);
        }
    }

    if state.pen_mode {
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                pen_click(graph, history, node_id, state, &path, image_rect, pos);
            }
        }
        if response.drag_started() {
            if let Some(start) = response.interact_pointer_pos() {
                if let Some(sp_i) = state.pen_subpath {
                    if closes_active_subpath(&path, sp_i, image_rect, start) {
                        let mut new_path = path.clone();
                        new_path.subpaths[sp_i].closed = true;
                        state.pen_subpath = None;
                        state.selected.clear();
                        commit_path_change(graph, history, node_id, path.clone(), new_path);
                    } else {
                        start_pen_drag(graph, node_id, state, &path, image_rect, start);
                    }
                } else {
                    start_pen_drag(graph, node_id, state, &path, image_rect, start);
                }
            }
        }
    } else {
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let target = hit_test(&path, image_rect, pos, &state.selected);
                apply_selection(ui, state, target);
            }
        }
        if response.drag_started() {
            if let Some(start) = response.interact_pointer_pos() {
                let target = hit_test(&path, image_rect, start, &state.selected);
                apply_selection(ui, state, target);
                state.drag = target.map(|target| ActiveDrag {
                    target,
                    base_path: path.clone(),
                    old_path: path.clone(),
                });
            }
        }
        if response.hovered()
            && !state.selected.is_empty()
            && ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace))
        {
            delete_selected(graph, history, node_id, &path, state);
        }
    }

    if response.dragged() {
        if let (Some(active), Some(cur)) = (&state.drag, response.interact_pointer_pos()) {
            update_dragged(graph, node_id, active, image_rect, cur);
        }
    }
    if response.drag_stopped() {
        finish_drag(graph, history, node_id, state);
    }

    // Re-read for drawing — a drag, click, or delete just above may have
    // changed it this same frame.
    if let Some(current) = graph
        .node(node_id)
        .and_then(|n| n.params.get("path"))
        .and_then(ParamValue::as_path)
    {
        draw(ui, image_rect, current, &state.selected, state.pen_mode);
    }
}

/// Appends a new `Symmetric` point (starting a fresh subpath if Pen mode
/// has none active yet) and hands off to the ordinary drag machinery so
/// its Out handle drags live for the rest of this gesture — `old_path` is
/// `path` itself (pre-insertion), so if the gesture ends with no real
/// movement the undo step still correctly reverts the insertion, not just
/// the handle.
fn start_pen_drag(
    graph: &mut mc_core::Graph,
    node_id: mc_core::NodeId,
    state: &mut PathEditorState,
    path: &mc_core::Path,
    image_rect: egui::Rect,
    start: egui::Pos2,
) {
    let uv = screen_to_uv(start, image_rect);
    let new_point = Point {
        kind: PointKind::Symmetric,
        ..Point::new(uv)
    };

    let mut base_path = path.clone();
    let (sp_i, pt_i) = match state
        .pen_subpath
        .filter(|&i| base_path.subpaths.get(i).is_some())
    {
        Some(sp_i) => {
            base_path.subpaths[sp_i].points.push(new_point);
            (sp_i, base_path.subpaths[sp_i].points.len() - 1)
        }
        None => {
            base_path
                .subpaths
                .push(SubPath::new(vec![new_point], false));
            let sp_i = base_path.subpaths.len() - 1;
            state.pen_subpath = Some(sp_i);
            (sp_i, 0)
        }
    };
    state.selected = HashSet::from([(sp_i, pt_i)]);

    let _ = graph.set_param(node_id, "path", ParamValue::Path(base_path.clone()));
    state.drag = Some(ActiveDrag {
        target: DragTarget::Handle((sp_i, pt_i), mc_nodes::path::HandleSide::Out),
        base_path,
        old_path: path.clone(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_core::SubPath;

    fn square_rect() -> egui::Rect {
        egui::Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(200.0, 200.0))
    }

    fn triangle_path() -> mc_core::Path {
        mc_core::Path {
            subpaths: vec![SubPath::new(
                vec![
                    Point::new([0.1, 0.1]),
                    Point {
                        pos: [0.5, 0.5],
                        h_in: [-0.05, 0.0],
                        h_out: [0.05, 0.0],
                        kind: PointKind::Smooth,
                        width: 1.0,
                    },
                    Point::new([0.1, 0.9]),
                ],
                false,
            )],
            wrap: true,
        }
    }

    fn graph_with_path(path: mc_core::Path) -> (mc_core::Graph, mc_core::NodeId) {
        let mut graph = mc_core::Graph::new();
        let id = graph.add_node(mc_core::Node::new("path"));
        graph.set_param(id, "path", ParamValue::Path(path)).unwrap();
        (graph, id)
    }

    /// `uv_to_screen`/`screen_to_uv` round-trip through `f32` screen
    /// coordinates, so a uv fed in via `uv_to_screen` for a test fixture
    /// comes back slightly off in the last few bits — this tolerance is
    /// generous enough to absorb that without masking a real bug.
    fn assert_uv_close(actual: [f64; 2], expected: [f64; 2]) {
        assert!(
            (actual[0] - expected[0]).abs() < 1e-5 && (actual[1] - expected[1]).abs() < 1e-5,
            "expected {expected:?}, got {actual:?}"
        );
    }

    fn path_of(graph: &mc_core::Graph, id: mc_core::NodeId) -> mc_core::Path {
        graph
            .node(id)
            .and_then(|n| n.params.get("path"))
            .and_then(ParamValue::as_path)
            .cloned()
            .unwrap()
    }

    #[test]
    fn uv_screen_round_trips() {
        let rect = square_rect();
        for uv in [[0.0, 0.0], [1.0, 1.0], [0.37, 0.82]] {
            let screen = uv_to_screen(uv, rect);
            let back = screen_to_uv(screen, rect);
            assert!((back[0] - uv[0]).abs() < 1e-5);
            assert!((back[1] - uv[1]).abs() < 1e-5);
        }
    }

    #[test]
    fn hit_test_finds_the_nearest_point_within_pick_radius() {
        let path = triangle_path();
        let rect = square_rect();
        let near_first = uv_to_screen([0.1, 0.1], rect);
        let target = hit_test(&path, rect, near_first, &HashSet::new());
        assert!(matches!(target, Some(DragTarget::Point((0, 0)))));
    }

    #[test]
    fn hit_test_finds_nothing_far_from_any_point() {
        let path = triangle_path();
        let rect = square_rect();
        let far = uv_to_screen([0.9, 0.1], rect);
        assert!(hit_test(&path, rect, far, &HashSet::new()).is_none());
    }

    #[test]
    fn hit_test_prefers_a_selected_points_handle_over_the_point_itself() {
        let path = triangle_path();
        let rect = square_rect();
        // Point 1's h_out sits at uv [0.55, 0.5] — close enough to also be
        // within the point's own pick radius were it not for handle
        // priority.
        let mut selected = HashSet::new();
        selected.insert((0usize, 1usize));
        let near_handle = uv_to_screen([0.55, 0.5], rect);
        let target = hit_test(&path, rect, near_handle, &selected);
        assert!(matches!(
            target,
            Some(DragTarget::Handle((0, 1), mc_nodes::path::HandleSide::Out))
        ));
    }

    #[test]
    fn hit_test_skips_a_zero_length_handle_even_when_selected() {
        let path = triangle_path();
        let rect = square_rect();
        let mut selected = HashSet::new();
        selected.insert((0usize, 0usize)); // point 0 has zero handles
        let at_point = uv_to_screen([0.1, 0.1], rect);
        let target = hit_test(&path, rect, at_point, &selected);
        assert!(matches!(target, Some(DragTarget::Point((0, 0)))));
    }

    #[test]
    fn apply_drag_moves_a_points_position() {
        let mut path = triangle_path();
        apply_drag(&mut path, DragTarget::Point((0, 0)), [0.2, 0.3]);
        assert_eq!(path.subpaths[0].points[0].pos, [0.2, 0.3]);
    }

    #[test]
    fn apply_drag_on_a_handle_goes_through_drag_handle_not_a_raw_assignment() {
        let mut path = triangle_path();
        // Point 1 is Smooth — dragging h_out should rotate h_in to stay
        // opposite, not leave it untouched the way a raw assignment would.
        let original_h_in = path.subpaths[0].points[1].h_in;
        apply_drag(
            &mut path,
            DragTarget::Handle((0, 1), mc_nodes::path::HandleSide::Out),
            [0.5, 0.58], // new h_out points straight up
        );
        let moved = &path.subpaths[0].points[1];
        assert_ne!(moved.h_in, original_h_in);
        assert!(
            moved.h_in[1] < 0.0,
            "h_in should now point the opposite way"
        );
    }

    #[test]
    fn pen_click_on_empty_space_starts_a_new_subpath() {
        let (mut graph, id) = graph_with_path(mc_core::Path::default());
        let mut history = mc_core::History::new();
        let mut state = PathEditorState::default();
        let rect = square_rect();

        pen_click(
            &mut graph,
            &mut history,
            id,
            &mut state,
            &mc_core::Path::default(),
            rect,
            uv_to_screen([0.2, 0.3], rect),
        );

        let path = path_of(&graph, id);
        assert_eq!(path.subpaths.len(), 1);
        assert_eq!(path.subpaths[0].points.len(), 1);
        assert_uv_close(path.subpaths[0].points[0].pos, [0.2, 0.3]);
        assert_eq!(path.subpaths[0].points[0].kind, PointKind::Corner);
        assert_eq!(state.pen_subpath, Some(0));
    }

    #[test]
    fn pen_click_appends_to_the_active_subpath() {
        let initial = {
            let mut p = mc_core::Path::default();
            p.subpaths
                .push(SubPath::new(vec![Point::new([0.1, 0.1])], false));
            p
        };
        let (mut graph, id) = graph_with_path(initial.clone());
        let mut history = mc_core::History::new();
        let mut state = PathEditorState {
            pen_subpath: Some(0),
            ..Default::default()
        };
        let rect = square_rect();

        pen_click(
            &mut graph,
            &mut history,
            id,
            &mut state,
            &initial,
            rect,
            uv_to_screen([0.6, 0.6], rect),
        );

        let path = path_of(&graph, id);
        assert_eq!(path.subpaths[0].points.len(), 2);
        assert_uv_close(path.subpaths[0].points[1].pos, [0.6, 0.6]);
        assert!(!path.subpaths[0].closed);
    }

    #[test]
    fn pen_click_on_the_first_point_closes_the_subpath() {
        let initial = {
            let mut p = mc_core::Path::default();
            p.subpaths.push(SubPath::new(
                vec![
                    Point::new([0.1, 0.1]),
                    Point::new([0.5, 0.1]),
                    Point::new([0.3, 0.5]),
                ],
                false,
            ));
            p
        };
        let (mut graph, id) = graph_with_path(initial.clone());
        let mut history = mc_core::History::new();
        let mut state = PathEditorState {
            pen_subpath: Some(0),
            ..Default::default()
        };
        let rect = square_rect();

        pen_click(
            &mut graph,
            &mut history,
            id,
            &mut state,
            &initial,
            rect,
            uv_to_screen([0.1, 0.1], rect),
        );

        let path = path_of(&graph, id);
        assert!(path.subpaths[0].closed);
        assert_eq!(
            path.subpaths[0].points.len(),
            3,
            "closing shouldn't add a point"
        );
        assert_eq!(state.pen_subpath, None);
    }

    #[test]
    fn pen_click_refuses_to_close_a_two_point_subpath() {
        let initial = {
            let mut p = mc_core::Path::default();
            p.subpaths.push(SubPath::new(
                vec![Point::new([0.1, 0.1]), Point::new([0.5, 0.1])],
                false,
            ));
            p
        };
        let (mut graph, id) = graph_with_path(initial.clone());
        let mut history = mc_core::History::new();
        let mut state = PathEditorState {
            pen_subpath: Some(0),
            ..Default::default()
        };
        let rect = square_rect();

        // Clicking back on the first point of a 2-point subpath should
        // append a third point, not close it.
        pen_click(
            &mut graph,
            &mut history,
            id,
            &mut state,
            &initial,
            rect,
            uv_to_screen([0.1, 0.1], rect),
        );

        let path = path_of(&graph, id);
        assert!(!path.subpaths[0].closed);
        assert_eq!(path.subpaths[0].points.len(), 3);
    }

    #[test]
    fn start_pen_drag_creates_a_symmetric_point_and_arms_its_handle_drag() {
        let path = mc_core::Path::default();
        let (mut graph, id) = graph_with_path(path.clone());
        let mut state = PathEditorState::default();
        let rect = square_rect();

        start_pen_drag(
            &mut graph,
            id,
            &mut state,
            &path,
            rect,
            uv_to_screen([0.4, 0.4], rect),
        );

        let current = path_of(&graph, id);
        assert_eq!(current.subpaths[0].points.len(), 1);
        assert_eq!(current.subpaths[0].points[0].kind, PointKind::Symmetric);
        assert!(matches!(
            state.drag.as_ref().map(|d| d.target),
            Some(DragTarget::Handle((0, 0), mc_nodes::path::HandleSide::Out))
        ));
        // Undo should revert the insertion too, not just the handle.
        assert!(state.drag.as_ref().unwrap().old_path.subpaths.is_empty());
    }

    #[test]
    fn is_last_point_of_an_open_subpath_rejects_closed_or_non_last() {
        let path = triangle_path();
        assert!(is_last_point_of_an_open_subpath(&path, (0, 2)));
        assert!(!is_last_point_of_an_open_subpath(&path, (0, 0)));
        let mut closed = path.clone();
        closed.subpaths[0].closed = true;
        assert!(!is_last_point_of_an_open_subpath(&closed, (0, 2)));
    }
}
