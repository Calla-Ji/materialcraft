//! Crash-safe autosave: periodically snapshots the current graph to a fixed
//! recovery file outside the user's own save location, so a crash or force-
//! quit doesn't lose work back to the last manual save (CLAUDE.md's M6
//! "crash-safe autosave" item). The recovery file is cleared on every clean
//! New/Open/Save and on a graceful exit — if it's still present at the next
//! launch, the previous session didn't shut down cleanly, and `main.rs`
//! offers to restore it.

use std::path::PathBuf;
use std::time::{Duration, Instant};

const INTERVAL: Duration = Duration::from_secs(30);

/// A fixed location outside the user's own files — recovery, not a real
/// save, so it doesn't belong in whatever folder they're organizing actual
/// graphs in.
pub fn recovery_path() -> PathBuf {
    std::env::temp_dir().join("materialcraft_autosave.mcgraph")
}

pub struct Autosave {
    last_saved_at: Instant,
}

impl Default for Autosave {
    fn default() -> Self {
        Self {
            last_saved_at: Instant::now(),
        }
    }
}

impl Autosave {
    /// Snapshots `graph` to the recovery path if `INTERVAL` has passed since
    /// the last snapshot. Best-effort: a failed write here (e.g. a full
    /// temp directory) isn't worth interrupting the artist over — the worst
    /// case is just a missed safety snapshot, not lost work relative to
    /// today's session.
    pub fn maybe_save(&mut self, graph: &mc_core::Graph) {
        if self.last_saved_at.elapsed() < INTERVAL {
            return;
        }
        self.last_saved_at = Instant::now();
        if let Err(err) = graph.save_to_file(recovery_path()) {
            log::warn!("autosave failed: {err}");
        }
    }

    /// Call after any clean New/Open/Save — that action is now the
    /// authoritative copy, so a leftover recovery snapshot would only ever
    /// be stale from here on.
    pub fn clear() {
        let _ = std::fs::remove_file(recovery_path());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test, not several, since every case here touches the same fixed
    // `recovery_path()` file — splitting into separate `#[test]` fns would
    // race against each other under cargo test's default parallel runner.
    #[test]
    fn maybe_save_respects_the_interval_and_clear_removes_the_file() {
        Autosave::clear();
        let graph = mc_core::Graph::new();

        let mut autosave = Autosave::default();
        autosave.maybe_save(&graph);
        assert!(
            !recovery_path().exists(),
            "a freshly-created Autosave shouldn't write before its interval elapses"
        );

        autosave.last_saved_at = Instant::now() - INTERVAL - Duration::from_secs(1);
        autosave.maybe_save(&graph);
        assert!(
            recovery_path().exists(),
            "maybe_save should write once the interval has elapsed"
        );

        Autosave::clear();
        assert!(!recovery_path().exists());
        // Calling again with nothing left to remove must not panic.
        Autosave::clear();
    }
}
