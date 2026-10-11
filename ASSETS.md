# Bundled assets

Per CLAUDE.md §13, every bundled asset must be CC0 or similarly permissive, recorded here with
its source and license.

## HDRIs

- `assets/hdri/studio_small_03_1k.exr` — "Studio Small 03" by Greg Zaal, via
  [Poly Haven](https://polyhaven.com/a/studio_small_03). License: CC0 (public domain). Used by
  `mc_app`'s 3D viewport (`crates/mc_app/src/hdri.rs`) as the sky and reflection environment.
  Downloaded at 1k resolution to keep the repository small.

## Fonts

- `assets/fonts/MaterialIconsRound-Regular.otf` — "Material Icons" (Round style) by Google, from
  the [material-design-icons](https://github.com/google/material-design-icons) repository.
  License: Apache License 2.0. Used by `mc_app`'s 3D viewport toolbar
  (`crates/mc_app/src/viewport3d_panels.rs`) for the Mesh/Material/Camera/Environment icon
  buttons, via specific glyph codepoints (not the whole icon set, but the font is bundled
  in full since Apache-2.0 doesn't require subsetting).

## App icon

- `assets/icon/icon_256.png` (window/taskbar icon, embedded via `mc_app::app_icon`) and
  `assets/icon/icon_1024.png` (master, for any future use — app bundle icons, favicon, GitHub
  social image) — the Material Icons Round "texture" glyph (same font/license as above),
  centered on a rounded-rect Neutron Star (`#131718`) background with a soft glass-style
  highlight gradient, rendered in Supernova Light (`#FEE6EA`), per the owner's brand palette.
  Generated with Pillow from the bundled OTF, not drawn by hand.
- `assets/fonts/Geist-Regular.ttf` — "Geist" by Vercel, in collaboration with basement.studio,
  from the [geist-font](https://github.com/vercel/geist-font) repository. License: SIL Open Font
  License 1.1. Used as `mc_app`'s UI typeface throughout, replacing egui's default font.
