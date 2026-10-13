// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Path node's own interactive overlay in the 2D View — drawing the
//! curve on top of its Preview Mask, and letting the artist select and
//! drag its points and handles directly. PATH_NODE_SPEC.md §4 is the
//! source of truth for the full editor; this is its first slice:
//! click/shift-click select, drag a point, drag a handle (respecting its
//! kind's constraint via `mc_nodes::path::drag_handle`), and
//! Delete/Backspace to remove the selection. Double-click kind toggling,
//! the right-click menu, Alt-drag break-link, click-on-segment insert, Pen
//! mode, SVG paste, and wrap ghost copies are still to come — deliberately
//! held back so this core slice can be tested hands-on first.
//!
//! Shares the 2D View's existing pan-drag `egui::Response` rather than
//! opening a second `Sense::click_and_drag()` region over the same rect —
//! two independent interact regions on one area would race for which one
//! "wins" a given frame's pointer input.

use std::collections::{HashMap, HashSet};

use mc_core::ParamValue;

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
            // this first slice.
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

fn draw(ui: &egui::Ui, image_rect: egui::Rect, path: &mc_core::Path, selected: &HashSet<PointKey>) {
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
}

/// Draws the current Path and handles select/drag/delete for it. Called
/// once per frame the 2D View is showing a selected Path node's single
/// (non-tiled) preview — `response` is the same drag-sense response the
/// 2D View already created for panning, reused rather than duplicated so
/// the two don't race for the same pointer input; the caller is
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
                old_path: path.clone(),
            });
        }
    }

    if response.dragged() {
        if let (Some(active), Some(cur)) = (&state.drag, response.interact_pointer_pos()) {
            let mut new_path = active.old_path.clone();
            apply_drag(&mut new_path, active.target, screen_to_uv(cur, image_rect));
            let _ = graph.set_param(node_id, "path", ParamValue::Path(new_path));
        }
    }

    if response.drag_stopped() {
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

    if response.hovered() {
        if !state.selected.is_empty()
            && ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace))
        {
            delete_selected(graph, history, node_id, &path, state);
        }
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            state.selected.clear();
        }
    }

    // Re-read for drawing — a drag or delete just above may have changed
    // it this same frame.
    if let Some(current) = graph
        .node(node_id)
        .and_then(|n| n.params.get("path"))
        .and_then(ParamValue::as_path)
    {
        draw(ui, image_rect, current, &state.selected);
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

#[cfg(test)]
mod tests {
    use super::*;
    use mc_core::{Point, PointKind, SubPath};

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
            [0.5 + 0.0, 0.5 + 0.08], // new h_out points straight up
        );
        let moved = &path.subpaths[0].points[1];
        assert_ne!(moved.h_in, original_h_in);
        assert!(
            moved.h_in[1] < 0.0,
            "h_in should now point the opposite way"
        );
    }
}
