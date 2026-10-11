# CREDITS.md — Algorithm sources

Every node in MaterialCraft is implemented from scratch in Rust/WGSL against the general,
published technique it's named after — never against another tool's code, documentation, or
internal file formats (see [INDEPENDENCE.md](INDEPENDENCE.md)). This file records which
published technique each node builds on, as the paper trail for that.

## Noise & patterns

- **Perlin Noise** — Ken Perlin's gradient noise, as described in *An Image Synthesizer*
  (SIGGRAPH 1985) and refined in *Improving Noise* (SIGGRAPH 2002). The reference lattice/
  gradient-hash structure is standard and public domain; this implementation is tileable by
  construction (the gradient hash wraps at a fixed period), which the original algorithm
  doesn't guarantee on its own.
- **Simplex Noise** — the triangular-lattice noise Ken Perlin introduced in 2001, implemented
  here following the practical breakdown in Stefan Gustavson's *Simplex Noise Demystified*
  (2005). Tileability is handled the same way as Perlin Noise above: by blending multiple
  period-shifted copies so both tile edges evaluate identically.
- **Voronoi** — cellular/Worley noise, from Steven Worley's *A Cellular Texture Basis
  Function* (SIGGRAPH 1996): distances from each pixel to the nearest of a set of scattered
  feature points, with F1/F2/F2−F1/edges as the standard output modes.
- **Fractal Sum / Clouds** — fractional Brownian motion (fBm): summing several octaves of a
  base noise at increasing frequency and decreasing amplitude, a standard technique with no
  single canonical source (see e.g. Ken Musgrave's writing on fractal terrain synthesis).

## Filters

- **Blur / Directional Blur / Slope Blur** — a standard separable Gaussian blur (textbook
  image-processing technique), wrap-addressed for tiling.
- **Edge Detect / Height to Normal**'s gradient estimation — the Sobel operator (Sobel &
  Feldman, 1968), a standard image-gradient kernel.
- **Warp / Directional Warp** — displacing a texture by the gradient of a second ("intensity")
  input, a standard procedural-texturing technique with no single canonical source.
- **Bevel** — box-blur-then-remap: a uniform box blur (not a Gaussian — a box blur of a hard
  step produces an exactly linear ramp over its own width, which reads as a defined chamfer
  rather than a generic soft blur) at a mask/height input's edges, then reshape *how much the
  blur changed each pixel* with a power curve, added back onto the unblurred original. A
  standard, cheap approximation of a distance-based bevel (vs. computing a true distance-to-edge
  field), with no single canonical source — this project's own derivation of the box-blur
  radius/remap exponent parameterization, including the delta-based remap formulation (rather
  than reshaping the blurred value directly) that guarantees a no-bevel setting is an exact
  no-op on any input. The box blur itself is computed with a sliding-window running sum (one
  thread per row/column, add the texel entering the window and subtract the one leaving it) —
  a standard, well-known box-filter technique, chosen here over a fixed-stride subsample after
  the latter produced a visible staircase artifact on hard edges (see PROGRESS.md).

## Stylize & glitch

- **Dither Pattern**'s Bayer 2×2/4×4/8×8 modes — the recursively-defined ordered-dither (Bayer)
  threshold matrix, a public-domain technique (see e.g. Wikipedia's "Ordered dithering"
  article for the general recursive definition); implemented from scratch in
  `shaders/common.wgsl`'s `bayer_threshold`, verified against the standard 2×2 and 4×4
  matrices by this project's own tests.
- **Dither Pattern**'s Blue Noise mode — samples the bundled CC0 blue noise texture
  (Christoph Peters, see ASSETS.md) rather than computing noise procedurally.
- **Dither Pattern**'s Rotation option — a classic halftone "screen angle," implemented via
  an integer-lattice rotation matrix (small (a, b) pairs, e.g. (2,1) for ~26.6°) rather than a
  continuous angle, since that's what keeps it exactly tileable; a standard, independently
  worked-out technique (see e.g. "rational tangent screening" in print-halftone literature),
  not adapted from any specific source.
- **Ordered Dither** and **Palette Dither** — standard ordered-dithering math (quantize, then
  nudge by a threshold mask before rounding), the same public-domain technique Dither Pattern's
  masks are built on, applied here to an actual image instead of just outputting the mask
  itself. Palette Dither's "nearest palette color" step and its Perceptual (luminance-weighted)
  distance option are a standard color-quantization technique, original implementation.
- **Block Shift**, **Channel Split**, **Scanline Jitter**, **Data Corrupt** — original glitch
  nodes built from general, independently-implemented ideas (seeded per-block/per-band hashing,
  wrap-addressed resampling, per-channel UV offsets), with
  [KinoGlitch](https://github.com/keijiro/KinoGlitch) (Keijiro Takahashi, MIT) as inspiration
  for which effects are worth having, not as ported code — this project's versions are WGSL
  compute shaders built against `common.wgsl`'s own hash helper, not a line-by-line port of
  KinoGlitch's Unity HLSL.
- Background reading for the family as a whole (concepts only; no code taken): Surma's
  [*Ditherpunk*](https://surma.dev/things/ditherpunk/) and its
  [sequel](https://www.makeworld.space/2021/02/dithering.html), and
  [KinoGlitch](https://github.com/keijiro/KinoGlitch) (Keijiro Takahashi, MIT) for
  general glitch-node ideas, reimplemented in WGSL rather than ported.

## 3D preview

- **Tangent-space basis computation** (`compute_tangents` in `mc_app::viewport3d`) — Eric
  Lengyel's method for deriving a per-vertex tangent from a triangle's position/UV gradient,
  as published in *Foundations of Game Engine Development, Volume 2: Rendering* and his
  earlier widely-circulated article *Computing Tangent Space Basis Vectors for an Arbitrary
  Mesh*.
- **PBR shading** (`viewport3d.wgsl`) — the metallic-roughness GGX microfacet model used by
  virtually every modern physically-based renderer (Blender's Principled BSDF, Unreal, Godot,
  glTF's own material model included): Brent Burley's *Physically Based Shading at Disney*
  (SIGGRAPH 2012 course notes) and Brian Karis's *Real Shading in Unreal Engine 4* (SIGGRAPH
  2013 course notes), both public and widely cited.

## Shape math

The Shape node's Star (an exact straight-edge star-polygon distance field), Leaf (an
asymmetric superellipse with independently blended top/middle/bottom exponents), and the
per-kind upright-orientation offsets are original derivations written for this project —
worked out and verified against this project's own test suite and rendered output, not
adapted from any other tool.

---

If you add a node, add an entry here naming the published technique (or "original
derivation, see `<path>`" if there isn't one) before merging.
