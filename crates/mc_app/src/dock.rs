//! A fixed-topology docking layout for the four main panels ("crates"): 2D
//! View, 3D View, Node Graph, and Parameters. Built directly on raw egui
//! primitives (`Painter`, `Sense::drag`, rects we own) rather than
//! `egui::Window`/`Panel`/`ui.columns` — every one of those auto-sizes to its
//! content, and that behavior has repeatedly corrupted sibling layouts in
//! this app: the previews panel's resize handle, the Material side panel
//! overflowing its column, the 3D viewport rendering squashed, and (the
//! immediate reason this module replaced `egui::Window`-based floating
//! crates) flicker from recreating a window every resized frame plus
//! independent crates that wouldn't stay aligned with their neighbors.
//!
//! A dock tree of shared split lines — the same idea as Blender's screen
//! areas or UE5's Slate docking — avoids all of that: dragging a border
//! moves both neighbors at once because they read the *same* stored ratio,
//! so there's nothing to keep in sync after the fact and nothing to destroy
//! and recreate on a resize.

use egui::{CursorIcon, Pos2, Rect, Sense, StrokeKind, UiBuilder, Vec2};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CrateId {
    TwoDView,
    ThreeDView,
    Parameters,
    NodeGraph,
    NodeLibrary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SplitAxis {
    /// Divider is a vertical line; children sit left/right.
    Horizontal,
    /// Divider is a horizontal line; children sit top/bottom.
    Vertical,
}

enum DockNode {
    Leaf(CrateId),
    Split {
        /// Assigned once at tree construction so the divider's drag hit-test
        /// has a stable identity across frames — unlike the node's current
        /// rect, which changes every time `ratio` does.
        id: egui::Id,
        axis: SplitAxis,
        /// Fraction of the split's extent given to `first`.
        ratio: f32,
        first: Box<DockNode>,
        second: Box<DockNode>,
    },
}

pub struct DockLayout {
    root: DockNode,
}

impl Default for DockLayout {
    /// A Node Library column on the far left, mirroring Parameters' full-
    /// height column on the far right; between them, the original layout
    /// unchanged — 2D View and 3D View side by side on top, Node Graph
    /// spanning that same width underneath. Built as one new outer split
    /// wrapping the previous tree whole, so none of the inner splits'
    /// ratios (or the behavior they already had) needed to change.
    fn default() -> Self {
        use DockNode::{Leaf, Split};

        let top_row = Split {
            id: egui::Id::new("dock_split_2d_3d"),
            axis: SplitAxis::Horizontal,
            ratio: 0.5,
            first: Box::new(Leaf(CrateId::TwoDView)),
            second: Box::new(Leaf(CrateId::ThreeDView)),
        };
        let left_column = Split {
            id: egui::Id::new("dock_split_top_nodegraph"),
            axis: SplitAxis::Vertical,
            ratio: 0.55,
            first: Box::new(top_row),
            second: Box::new(Leaf(CrateId::NodeGraph)),
        };
        let main_and_params = Split {
            id: egui::Id::new("dock_split_main_params"),
            axis: SplitAxis::Horizontal,
            ratio: 0.72,
            first: Box::new(left_column),
            second: Box::new(Leaf(CrateId::Parameters)),
        };
        let root = Split {
            id: egui::Id::new("dock_split_library_main"),
            axis: SplitAxis::Horizontal,
            ratio: 0.1,
            first: Box::new(Leaf(CrateId::NodeLibrary)),
            second: Box::new(main_and_params),
        };
        Self { root }
    }
}

const DIVIDER_THICKNESS: f32 = 6.0;
const MIN_RATIO: f32 = 0.08;
const MAX_RATIO: f32 = 0.92;
const HEADER_HEIGHT: f32 = 28.0;

impl DockLayout {
    /// Renders every leaf and every divider in the tree inside `rect`.
    /// `leaf_title` names a crate for its header bar; `leaf_content` draws a
    /// crate's actual contents into a `Ui` already clipped to its cell.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        rect: Rect,
        leaf_title: impl Fn(CrateId) -> &'static str,
        leaf_content: &mut dyn FnMut(&mut egui::Ui, CrateId),
    ) {
        Self::show_node(&mut self.root, ui, rect, &leaf_title, leaf_content);

        // Drawn once here, around the whole dock area, *after* every leaf
        // (and every internal divider) so it paints on top instead of
        // getting erased by a leaf's own background fill flush against
        // this same outer edge — each *internal* boundary gets its own
        // single line the same way, drawn once by whichever `Split` owns
        // it (see `show_node`), rather than each neighboring leaf drawing
        // its own full-perimeter border. Two independently-drawn,
        // anti-aliased 1px lines sitting right on top of each other don't
        // perfectly cancel out to one crisp 1px line — they visibly read as
        // a thicker, slightly blurred one.
        ui.painter()
            .rect_stroke(rect, 0.0, ui.visuals().window_stroke(), StrokeKind::Inside);
    }

    fn show_node(
        node: &mut DockNode,
        ui: &mut egui::Ui,
        rect: Rect,
        leaf_title: &impl Fn(CrateId) -> &'static str,
        leaf_content: &mut dyn FnMut(&mut egui::Ui, CrateId),
    ) {
        match node {
            DockNode::Leaf(crate_id) => {
                draw_leaf(ui, rect, leaf_title(*crate_id), *crate_id, leaf_content);
            }
            DockNode::Split {
                id,
                axis,
                ratio,
                first,
                second,
            } => {
                let (mut r1, mut r2) = split_rect(rect, *axis, *ratio);

                let handle_rect = match axis {
                    SplitAxis::Horizontal => Rect::from_center_size(
                        Pos2::new(r1.right(), rect.center().y),
                        Vec2::new(DIVIDER_THICKNESS, rect.height()),
                    ),
                    SplitAxis::Vertical => Rect::from_center_size(
                        Pos2::new(rect.center().x, r1.bottom()),
                        Vec2::new(rect.width(), DIVIDER_THICKNESS),
                    ),
                };

                let response = ui.interact(handle_rect, *id, Sense::drag());
                if response.dragged() {
                    let delta = response.drag_delta();
                    let (extent, d) = match axis {
                        SplitAxis::Horizontal => (rect.width(), delta.x),
                        SplitAxis::Vertical => (rect.height(), delta.y),
                    };
                    if extent > 1.0 {
                        *ratio = (*ratio + d / extent).clamp(MIN_RATIO, MAX_RATIO);
                    }
                    // Recompute with the new ratio so the children lay out at
                    // the dragged-to position *this* frame instead of
                    // trailing the mouse by one frame.
                    let (nr1, nr2) = split_rect(rect, *axis, *ratio);
                    r1 = nr1;
                    r2 = nr2;
                }
                if response.hovered() || response.dragged() {
                    ui.ctx().set_cursor_icon(match axis {
                        SplitAxis::Horizontal => CursorIcon::ResizeHorizontal,
                        SplitAxis::Vertical => CursorIcon::ResizeVertical,
                    });
                }

                Self::show_node(first, ui, r1, leaf_title, leaf_content);
                Self::show_node(second, ui, r2, leaf_title, leaf_content);

                // The one and only line for this boundary — drawn *after*
                // both children (and everything inside them, including
                // their own leaf background fills) rather than before, so
                // it paints on top instead of immediately getting painted
                // over by a child leaf's opaque fill starting at this exact
                // boundary coordinate. See `show`'s doc comment on why this
                // isn't also drawn by `first`/`second` themselves.
                let stroke = ui.visuals().window_stroke();
                match axis {
                    SplitAxis::Horizontal => ui.painter().line_segment(
                        [
                            Pos2::new(r1.right(), rect.top()),
                            Pos2::new(r1.right(), rect.bottom()),
                        ],
                        stroke,
                    ),
                    SplitAxis::Vertical => ui.painter().line_segment(
                        [
                            Pos2::new(rect.left(), r1.bottom()),
                            Pos2::new(rect.right(), r1.bottom()),
                        ],
                        stroke,
                    ),
                };
            }
        }
    }
}

fn split_rect(rect: Rect, axis: SplitAxis, ratio: f32) -> (Rect, Rect) {
    // Snapped to a whole pixel: a boundary at a fractional pixel position
    // draws anti-aliased (smeared across two physical pixels) even when
    // only one line is ever drawn there now — snapping keeps that one line
    // crisp, and keeps `r1`/`r2` meeting at the exact same coordinate the
    // line is drawn at instead of a near-but-not-quite match.
    match axis {
        SplitAxis::Horizontal => {
            let x = (rect.left() + rect.width() * ratio).round();
            (
                Rect::from_min_max(rect.min, Pos2::new(x, rect.bottom())),
                Rect::from_min_max(Pos2::new(x, rect.top()), rect.max),
            )
        }
        SplitAxis::Vertical => {
            let y = (rect.top() + rect.height() * ratio).round();
            (
                Rect::from_min_max(rect.min, Pos2::new(rect.right(), y)),
                Rect::from_min_max(Pos2::new(rect.left(), y), rect.max),
            )
        }
    }
}

fn draw_leaf(
    ui: &mut egui::Ui,
    rect: Rect,
    title: &str,
    id: CrateId,
    leaf_content: &mut dyn FnMut(&mut egui::Ui, CrateId),
) {
    let visuals = ui.visuals().clone();
    let painter = ui.painter();
    // All four dock cells sit on Neutron Star — only the chrome around them
    // (menu bar, node cards, the 3D viewport's slide-out drawer) keeps the
    // default panel fill (Wormhole Grey, set globally in `theme::apply`).
    painter.rect_filled(rect, 0.0, crate::theme::NEUTRON_STAR);

    let header_rect = Rect::from_min_size(rect.min, Vec2::new(rect.width(), HEADER_HEIGHT));
    painter.text(
        header_rect.center(),
        egui::Align2::CENTER_CENTER,
        title,
        egui::FontId::proportional(15.0),
        crate::theme::SUPERNOVA_LIGHT,
    );
    painter.line_segment(
        [header_rect.left_bottom(), header_rect.right_bottom()],
        visuals.window_stroke(),
    );

    let content_rect =
        Rect::from_min_max(Pos2::new(rect.left(), rect.top() + HEADER_HEIGHT), rect.max);
    let mut child = ui.new_child(UiBuilder::new().max_rect(content_rect));
    child.set_clip_rect(content_rect);
    leaf_content(&mut child, id);
}
