// SPDX-License-Identifier: MIT OR Apache-2.0
//! The "pick a starting graph" screen: the preset scaffold of all 8 Output
//! nodes, or a blank canvas, plus the working resolution. Shown on every
//! launch before the editor itself, and again whenever the artist presses
//! "New" in the menu bar (`main.rs`'s `do_new`) — the only place
//! `mc_core::GraphSettings::resolution` is directly user-editable (CLAUDE.md
//! §9's layout doesn't have one; the export dialog's own resolution picker
//! is separate and export-time-only).
//!
//! Deliberately shares `export_dialog.rs`'s `[256, 512, 1024, 2048, 4096]`
//! resolution list — same reasoning the owner gave for the graph's own
//! working resolution range: 8192 is `GraphSettings`'s own sanity ceiling
//! (`mc_core::graph::MAX_RESOLUTION`, guarding a hand-edited or corrupted
//! `.mcgraph` file), not a resolution worth offering as a day-to-day choice.
//!
//! The Blender/Unreal Engine options wrap `mc_export::naming::Preset`
//! rather than a second, parallel enum — "Blender" vs "Unreal Engine" is
//! exactly the same target-engine choice the export dialog already makes,
//! and the owner's own framing for why they'll eventually differ (channel
//! packing) is an export-time concept to begin with. Both presets build the
//! identical 8-output starter graph today; `Preset::normal_format()`
//! already exists if a real distinction is wanted before packing lands, but
//! isn't wired in here yet since the owner scoped the difference to "once
//! we add more outputs."
//!
//! Custom-drawn header (`title_bar(false)`, a hand-painted label dragged via
//! `Sense::drag()` — same pattern `dock.rs`'s own split-divider handle
//! uses) rather than `egui::Window`'s default title bar, since that default
//! renders at a much larger font than the rest of the app's panel headers
//! and there was no way to restyle just the text size short of replacing it
//! entirely.
//!
//! No frame stroke and no inner margin: `egui::Frame::stroke` reserves its
//! own width as extra margin *outside* the content area it hands back
//! (confirmed in egui's own `Window` source — `resize.max_size` is shrunk
//! by `frame.total_margin()`, which includes the stroke width), so a
//! border drawn that way never quite reaches the same edge the content
//! itself gets laid out to. The border is hand-painted directly on
//! `ui.max_rect()` instead (last, after everything else, so it's never
//! drawn over), so it and the header separator/image placement all come
//! from the exact same rect and can't disagree.
//!
//! The window's own [`CORNER_RADIUS`] and the preview image's own corner
//! rounding are kept equal and drawn against the same outer rect on
//! purpose — the image is flush against the window's right/bottom edges,
//! so a *square*-cornered image there would stick out past the window's
//! rounded silhouette at the two corners it touches.

const RESOLUTIONS: [u32; 5] = [256, 512, 1024, 2048, 4096];

/// Every fixed dimension below is this fraction of its original design
/// size — written as `BASE * SHRINK` rather than pre-multiplied literals,
/// so the relationship (and the fact they're still all consistent with
/// each other) stays visible.
const SHRINK: f32 = 0.8;

const CORNER_RADIUS: f32 = 8.0;
/// Matches `dock.rs`'s own `HEADER_HEIGHT` exactly (2D View/3D
/// View/Parameters/Node Graph's panel headers) — not shrunk by [`SHRINK`]
/// like everything else here, since it's meant to look consistent with
/// those, not scaled relative to this window's own content.
const HEADER_HEIGHT: f32 = 28.0;
/// Space between the header separator and the controls block below it.
const HEADER_GAP: f32 = 8.0 * SHRINK;
const LEFT_MARGIN: f32 = 16.0 * SHRINK;
const CONTROLS_WIDTH: f32 = 260.0 * SHRINK;
/// Fixed, rather than measured from the controls' own natural height — so
/// the image below always starts at the same y regardless of minor widget
/// spacing changes. Not scaled by [`SHRINK`] like the layout constants
/// above — widget sizes (combo boxes, buttons) come from the shared egui
/// theme, not from anything here, so shrinking this value doesn't shrink
/// them; it just reserves less room than they actually need. Measured
/// directly (2 combo boxes + 12px gap + a button needs ~77px) with a
/// margin on top, after the previous 64px value — a guess, not a
/// measurement — clipped the Create button into the image below it.
const CONTROLS_HEIGHT: f32 = 88.0;
const GAP_BEFORE_IMAGE: f32 = 8.0 * SHRINK;
/// A scaled-down 1920x1080 (16:9) — the resolution a real material preview
/// image would eventually be captured at, not literally rendered that big
/// inside a startup dialog. Spans the window's full width, flush against
/// its left/right/bottom outline.
const IMAGE_SIZE: egui::Vec2 = egui::Vec2::new(640.0 * SHRINK, 360.0 * SHRINK);

/// Sized to exactly fit the header, the controls block, and [`IMAGE_SIZE`]
/// stacked vertically and flush against the window's own outline — not
/// picked independently of them.
const WINDOW_SIZE: egui::Vec2 = egui::Vec2::new(
    IMAGE_SIZE.x,
    HEADER_HEIGHT + HEADER_GAP + CONTROLS_HEIGHT + GAP_BEFORE_IMAGE + IMAGE_SIZE.y,
);

/// Bundled placeholder (`assets/splash/sea-breeze-1.jpg`, downsized from the
/// owner's original to keep the repository small) — stand-in for a real
/// MaterialCreator-rendered material preview or feature showcase once one
/// exists. Unlike every other bundled asset (see ASSETS.md), this one's
/// source/license isn't recorded there yet, since the owner called it out
/// as a "for now" placeholder rather than a permanent asset — swap it (and
/// add that entry) before a real release.
const PREVIEW_IMAGE_BYTES: &[u8] = include_bytes!("../../../assets/splash/sea-breeze-1.jpg");

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StartChoice {
    Empty,
    Blender,
    Unreal,
}

impl StartChoice {
    const ALL: [StartChoice; 3] = [
        StartChoice::Empty,
        StartChoice::Blender,
        StartChoice::Unreal,
    ];

    fn label(self) -> &'static str {
        match self {
            StartChoice::Empty => "Empty",
            StartChoice::Blender => "Blender",
            StartChoice::Unreal => "Unreal Engine",
        }
    }
}

pub struct SplashState {
    pub open: bool,
    start: StartChoice,
    resolution: u32,
    /// `None` until the first frame it's shown, at which point it's
    /// centered on screen and then only ever moved by dragging the header
    /// (see `show`'s own doc comment) — an `egui::Window` can't be both
    /// screen-centered *and* freely draggable at once (`.anchor(...)`
    /// re-pins it every frame, overriding any drag), so centering has to
    /// happen exactly once, here, instead.
    pos: Option<egui::Pos2>,
    /// Decoded once (JPEG decode + GPU upload isn't free) and reused every
    /// frame after.
    preview_texture: Option<egui::TextureHandle>,
    /// The controls block's own natural height, measured from its response
    /// rect the frame after it's first laid out — used to vertically
    /// centre it within [`CONTROLS_HEIGHT`] (see `show`'s own comment on
    /// why this is measured rather than a second guessed constant:
    /// [`CONTROLS_HEIGHT`] itself already notes a guessed value clipped
    /// the Create button once). `None` for exactly one frame (the very
    /// first time the splash is shown), during which the controls render
    /// top-aligned instead of centred — imperceptible at 60fps, and the
    /// same "settle after one frame" idiom egui's own popups/tooltips use
    /// for their own not-yet-measured sizes.
    controls_content_height: Option<f32>,
}

impl Default for SplashState {
    fn default() -> Self {
        Self {
            open: true,
            start: StartChoice::Empty,
            resolution: 1024,
            pos: None,
            preview_texture: None,
            controls_content_height: None,
        }
    }
}

/// Renders the splash window if `state.open`. Returns `Some` once the
/// artist picks a starting point; the caller is responsible for acting on
/// it and setting `state.open = false`.
pub fn show(ctx: &egui::Context, state: &mut SplashState) -> Option<(StartChoice, u32)> {
    if !state.open {
        return None;
    }

    let pos = state
        .pos
        .unwrap_or_else(|| ctx.content_rect().center() - WINDOW_SIZE / 2.0);
    let mut new_pos = pos;

    let preview_texture = state.preview_texture.get_or_insert_with(|| {
        let decoded = image::load_from_memory(PREVIEW_IMAGE_BYTES)
            .expect("bundled splash preview image must decode")
            .to_rgba8();
        let size = [decoded.width() as usize, decoded.height() as usize];
        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, decoded.as_raw());
        ctx.load_texture("splash_preview", color_image, egui::TextureOptions::LINEAR)
    });

    // A soft dim veil over whatever's behind the splash (the editor's own
    // panels, on every launch's first frame already laid out underneath
    // it) — `Order::Middle` so it always paints above the panels
    // themselves (`Order::Background`, per `egui::Order`'s own doc
    // comment: "Painted behind all floating windows"), but *before* the
    // splash `Window` below so that — both now being the same tier — the
    // window still paints on top of it, not the other way around. Purely
    // visual, not an input-blocking modal scrim: nothing here calls
    // `Sense`, so a click still reaches whatever's underneath.
    egui::Area::new(egui::Id::new("splash_dim_veil"))
        .order(egui::Order::Middle)
        .fixed_pos(egui::Pos2::ZERO)
        .interactable(false)
        .show(ctx, |ui| {
            ui.painter().rect_filled(
                ctx.content_rect(),
                0.0,
                egui::Color32::from_rgba_unmultiplied(0x13, 0x17, 0x18, 140),
            );
        });

    let mut confirmed = false;
    egui::Window::new("Start a New Graph")
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .fixed_size(WINDOW_SIZE)
        .current_pos(pos)
        .frame(
            egui::Frame::new()
                .fill(crate::theme::NEUTRON_STAR)
                .inner_margin(0)
                .corner_radius(CORNER_RADIUS),
        )
        .show(ctx, |ui| {
            // Every gap in this layout is explicit (the `add_space` calls
            // below) — without this, egui's own default `item_spacing`
            // sneaks in an extra few px before every `allocate_exact_size`
            // call too, which compounded enough to push the image a few
            // pixels past the window's own bottom edge.
            ui.spacing_mut().item_spacing = egui::Vec2::ZERO;

            let full_rect = ui.max_rect();

            // Matches `dock.rs`'s own panel header style (2D View/3D
            // View/Parameters/Node Graph) rather than egui::Window's much
            // larger default title size — and doubles as the drag handle
            // that moves the window, now that its real title bar is off.
            let (header_rect, header_response) = ui.allocate_exact_size(
                egui::vec2(full_rect.width(), HEADER_HEIGHT),
                egui::Sense::drag(),
            );
            if header_response.dragged() {
                new_pos += header_response.drag_delta();
            }
            ui.painter().text(
                header_rect.left_center() + egui::vec2(LEFT_MARGIN, 0.0),
                egui::Align2::LEFT_CENTER,
                "Start a New Graph",
                egui::FontId::proportional(15.0),
                crate::theme::SUPERNOVA_LIGHT,
            );
            // Lets the artist dismiss this without picking a starting
            // point — the same "New" invocation that opened it already
            // left whatever graph was open untouched (`do_new` only sets
            // `self.splash.open = true`), so closing here just cancels
            // that, same as it would on a fresh launch with no graph to
            // lose. Shares `dialog_chrome.rs`'s own close button drawing
            // for the same glyph/hover styling every other dialog's X
            // already uses.
            let close_rect = egui::Rect::from_center_size(
                header_rect.right_center()
                    - egui::vec2(
                        LEFT_MARGIN + crate::dialog_chrome::CLOSE_BUTTON_SIZE / 2.0,
                        0.0,
                    ),
                egui::Vec2::splat(crate::dialog_chrome::CLOSE_BUTTON_SIZE),
            );
            if crate::dialog_chrome::close_button(ui, "splash", close_rect) {
                state.open = false;
            }
            ui.painter().hline(
                header_rect.x_range(),
                header_rect.bottom(),
                egui::Stroke::new(1.0, crate::theme::SUPERNOVA_LIGHT),
            );
            ui.add_space(HEADER_GAP);

            let (controls_outer, _) = ui.allocate_exact_size(
                egui::vec2(full_rect.width(), CONTROLS_HEIGHT),
                egui::Sense::hover(),
            );
            // Vertically centred within `CONTROLS_HEIGHT` using the real
            // height measured last frame (see `controls_content_height`'s
            // own doc comment) rather than a second guessed constant on
            // top of `CONTROLS_HEIGHT`'s own already-adjusted-once one.
            let top_offset = state
                .controls_content_height
                .map(|measured| ((CONTROLS_HEIGHT - measured) / 2.0).max(0.0))
                .unwrap_or(0.0);
            let controls_rect = egui::Rect::from_min_size(
                controls_outer.min + egui::vec2(LEFT_MARGIN, top_offset),
                egui::vec2(CONTROLS_WIDTH, CONTROLS_HEIGHT - top_offset),
            );
            let mut controls_ui = ui.new_child(egui::UiBuilder::new().max_rect(controls_rect));
            // `new_child` inherits the outer `ui`'s spacing, including the
            // zeroed-out `item_spacing` above — fine for the outer layout's
            // own precise gaps, but it also squashed the combo
            // boxes/button together with no breathing room between them.
            // Restored to egui's own default here, just for this block.
            controls_ui.spacing_mut().item_spacing = egui::Vec2::new(8.0, 4.0);
            let content_response = controls_ui
                .vertical(|controls_ui| {
                    egui::ComboBox::from_label("Preset")
                        .selected_text(state.start.label())
                        .show_ui(controls_ui, |ui| {
                            for choice in StartChoice::ALL {
                                ui.selectable_value(&mut state.start, choice, choice.label());
                            }
                        });
                    egui::ComboBox::from_label("Resolution")
                        .selected_text(format!("{0}x{0}", state.resolution))
                        .show_ui(controls_ui, |ui| {
                            for &resolution in &RESOLUTIONS {
                                ui.selectable_value(
                                    &mut state.resolution,
                                    resolution,
                                    format!("{resolution}x{resolution}"),
                                );
                            }
                        });
                    controls_ui.add_space(12.0);
                    if controls_ui.button("Create").clicked() {
                        confirmed = true;
                    }
                })
                .response;
            state.controls_content_height = Some(content_response.rect.height());

            ui.add_space(GAP_BEFORE_IMAGE);

            // Flush against the window's own left/right/bottom outline —
            // no border of its own, just the bundled placeholder image
            // (see [`PREVIEW_IMAGE_BYTES`]'s own doc comment). Only its
            // bottom two corners are rounded, matching the window's own —
            // the top edge sits mid-window, nowhere near a rounded corner
            // of the window to match.
            let preview_rect = ui.allocate_exact_size(IMAGE_SIZE, egui::Sense::hover()).0;
            // `paint_at` renders exactly at `preview_rect` with no layout
            // pass of its own — `ui.put` goes through normal widget
            // layout instead, which was centering/padding the image
            // within the rect slightly rather than filling it exactly,
            // leaving a visible gap at the bottom despite `preview_rect`
            // itself already being flush with the window's own edge.
            let image = egui::Image::new((preview_texture.id(), preview_texture.size_vec2()))
                .corner_radius(egui::CornerRadius {
                    nw: 0,
                    ne: 0,
                    sw: CORNER_RADIUS as u8,
                    se: CORNER_RADIUS as u8,
                });
            image.paint_at(ui, preview_rect);

            // Drawn last so it's never covered by the header/controls/image
            // painted above it.
            ui.painter().rect_stroke(
                full_rect,
                CORNER_RADIUS,
                egui::Stroke::new(1.0, crate::theme::SUPERNOVA_LIGHT),
                egui::StrokeKind::Inside,
            );
        });

    state.pos = Some(new_pos);
    confirmed.then_some((state.start, state.resolution))
}
