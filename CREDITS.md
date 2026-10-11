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
