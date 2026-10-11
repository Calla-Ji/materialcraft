// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared "Tiling Mode" param for any node whose output can show a visible
//! edge-duplication artifact when it reads its input with wrap-around
//! addressing — e.g. Transform 2D backward-sampling an offset/rotated UV,
//! or Blur/Warp reading neighboring texels past the tile edge. `Both` (full
//! wraparound on both axes) is every node's default, matching the
//! always-on behavior every node had before this param existed — adding it
//! never changes a graph's existing output unless the artist touches it.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TilingMode {
    #[default]
    Both,
    HorizontalOnly,
    VerticalOnly,
    Off,
}

impl TilingMode {
    pub const NAMES: [&'static str; 4] =
        ["Both (Seamless)", "Horizontal Only", "Vertical Only", "Off"];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Both (Seamless)" => Some(Self::Both),
            "Horizontal Only" => Some(Self::HorizontalOnly),
            "Vertical Only" => Some(Self::VerticalOnly),
            "Off" => Some(Self::Off),
            _ => None,
        }
    }

    pub fn wrap_x(self) -> bool {
        matches!(self, Self::Both | Self::HorizontalOnly)
    }

    pub fn wrap_y(self) -> bool {
        matches!(self, Self::Both | Self::VerticalOnly)
    }

    /// Packed as a single `u32` for a node's uniform buffer: bit 0 = wrap X,
    /// bit 1 = wrap Y. Matches `wraps_x`/`wraps_y`'s bit layout in
    /// `common.wgsl`.
    pub fn as_u32(self) -> u32 {
        (self.wrap_x() as u32) | ((self.wrap_y() as u32) << 1)
    }
}

#[cfg(test)]
mod tests {
    use super::TilingMode;

    #[test]
    fn every_name_round_trips() {
        for name in TilingMode::NAMES {
            assert!(
                TilingMode::from_name(name).is_some(),
                "{name:?} should parse"
            );
        }
    }

    #[test]
    fn both_wraps_every_axis_matching_the_pre_existing_always_on_behavior() {
        assert!(TilingMode::Both.wrap_x());
        assert!(TilingMode::Both.wrap_y());
        assert_eq!(TilingMode::default(), TilingMode::Both);
    }

    #[test]
    fn off_wraps_no_axis() {
        assert!(!TilingMode::Off.wrap_x());
        assert!(!TilingMode::Off.wrap_y());
    }

    #[test]
    fn horizontal_and_vertical_only_wrap_exactly_one_axis_each() {
        assert!(TilingMode::HorizontalOnly.wrap_x());
        assert!(!TilingMode::HorizontalOnly.wrap_y());
        assert!(!TilingMode::VerticalOnly.wrap_x());
        assert!(TilingMode::VerticalOnly.wrap_y());
    }
}
