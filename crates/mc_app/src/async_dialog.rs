// SPDX-License-Identifier: MIT OR Apache-2.0
//! A safe way to show a native file dialog from inside the egui update
//! loop.
//!
//! Every dialog call used to go through `rfd::FileDialog`'s synchronous
//! API (`.save_file()`, `.pick_file()`, `.pick_folder()`), which on macOS
//! blocks by pumping `NSSavePanel`/`NSOpenPanel`'s own *nested* native run
//! loop. If any trackpad/mouse gesture event arrives while that nested
//! loop is running, it gets routed back into winit's own event-loop
//! internals mid-reentrancy and can panic across an `extern "C"` boundary
//! winit can't unwind across — aborting the whole process instead of
//! failing gracefully. Confirmed against a real crash report: the app
//! aborted inside `NSSavePanel::runModal`, from a gesture event arriving
//! while the Save As panel was open.
//!
//! The fix is `rfd::AsyncFileDialog`, which on macOS presents the panel as
//! a non-blocking sheet (`beginSheetModalForWindow:completionHandler:`)
//! instead of a blocking nested loop — no reentrancy hazard. [`PendingDialog`]
//! is the glue that makes that usable from egui's own per-frame,
//! no-async-runtime update loop: the native panel's own completion handler
//! fires independently on the main run loop regardless of when Rust code
//! gets around to polling the `Future` `rfd::AsyncFileDialog` returns, so
//! this just polls it unconditionally once per frame (requesting a
//! repaint while still pending, so frames keep coming even with no user
//! input) rather than needing a real executor or a `Waker` that does
//! anything beyond nothing — `std::task::Waker::noop()` is enough.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// A file-dialog result being awaited without blocking the main event
/// loop. Store as `Option<PendingDialog<T>>` in app state; call
/// [`Self::poll`] every frame and act on `Some(result)` once it resolves
/// (then set the field back to `None`).
pub struct PendingDialog<T> {
    future: Pin<Box<dyn Future<Output = T>>>,
}

impl<T> PendingDialog<T> {
    pub fn new(future: impl Future<Output = T> + 'static) -> Self {
        Self {
            future: Box::pin(future),
        }
    }

    /// `Some(result)` the first frame the native panel's own completion
    /// handler has actually fired; `None` while still waiting.
    pub fn poll(&mut self, ctx: &egui::Context) -> Option<T> {
        let mut cx = Context::from_waker(std::task::Waker::noop());
        match self.future.as_mut().poll(&mut cx) {
            Poll::Ready(value) => Some(value),
            Poll::Pending => {
                ctx.request_repaint();
                None
            }
        }
    }
}
