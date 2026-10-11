# MaterialCraft

A free, open-source, native node-based procedural material authoring tool, built in Rust.

Connect nodes — noises, patterns, filters, blends — into a graph that outputs a seamless,
tileable PBR texture set, previews it live in 2D and 3D, and exports maps ready for
**Blender** and **Unreal Engine**.

> Status: early development (v0.1, pre-release). See [PROGRESS.md](PROGRESS.md) for the
> current state and [CLAUDE.md](CLAUDE.md) for the full project scope and roadmap.

## Features

- **Real-time GPU node graph.** Every node is a WGSL compute shader; editing a param or
  rewiring a connection re-evaluates only what's downstream and dirty, live.
- **Live 2D and 3D preview.** The 2D view shows any selected node's output at full
  resolution, with tiling and single-channel views. The 3D view renders the graph's output
  maps on a sphere, cube, plane, rounded cube, or rounded cylinder under a GGX
  metallic-roughness PBR shader, with real per-vertex tangents and height-driven
  displacement on every mesh — not just a flat plane. Channel textures get a real generated
  mip chain and 16x anisotropic filtering, with an Unlit Base Color debug mode, an Exposure
  slider, and a Pixel/Nearest filtering toggle for inspecting dither/pixel-art materials
  crisply. The Alpha output channel drives real transparency in the preview, not just an
  exported mask.
- **Seamless tiling, guaranteed.** Every generator is periodic and every filter samples with
  wrap-around addressing; each node ships with its own automated tiling test.
  Resolution-independent by design — a graph looks the same at 512px and 4096px, detail
  aside.
- **Deterministic.** Same graph, same seed, same pixels — every time, on the same GPU.
- **A custom node-graph canvas** — pan, zoom (including trackpad pinch-to-zoom, working
  anywhere on the canvas), box-select, duplicate, undo/redo, and a category-colored,
  alphabetically-sorted, keyboard-navigable (arrow keys + Enter) searchable add-node menu —
  built directly on top of the UI toolkit's own primitives rather than a third-party graph
  widget. Every node card is color-coded by category, matching a dedicated Node Library panel
  along the canvas's left edge for browsing and adding nodes by icon. New nodes always land
  wherever you last clicked on the canvas.
- **Crash-safe autosave and error toasts.** The current graph is snapshotted to a recovery file
  every 30 seconds; if MaterialCraft doesn't shut down cleanly, the next launch offers to
  restore it. Failed opens/saves and graph evaluation errors surface as a dismissable
  notification instead of only reaching the terminal log.
- **Export** to PNG (8-bit and 16-bit) and 32-bit float EXR, at resolutions from 256px up to
  4096px (8192px where VRAM allows), with ready-made naming presets for Blender and for
  Unreal Engine (including an ORM-packed texture: AO/Roughness/Metallic in one file).
- **Save/load** as a human-readable, versioned `.mcgraph` JSON file.

## Nodes

39 nodes, each with its own tiling and determinism tests.

### Generators — Noise
- **Perlin Noise** — classic gradient noise, smooth and organic.
- **Simplex Noise** — gradient noise on a triangular lattice, fewer directional artifacts
  than Perlin.
- **Voronoi** — cellular noise: distances to a scattered set of points (F1, F2, F2−F1, edges).
- **White Noise** — fully uncorrelated random cells, no interpolation.
- **Fractal Sum** — several octaves of a chosen base noise, summed (fBm).
- **Clouds** — an fBm preset tuned for a cloud-like look.

### Generators — Patterns
- **Pattern Generator** — a brick or grid pattern, with gap, bevel, and per-tile size/rotation/
  luminance jitter.
- **Shape** — a disc, square, polygon, star, or leaf, centered in the tile. The star and leaf
  are original constructions, not smoothed approximations: the star's points are exact
  straight edges that can "unfold" a plain polygon into a star via one setting, and the leaf
  is a single continuous curve with independently shapeable top, middle, and bottom
  curvature, plus control over where its widest point sits.
- **Gradient Linear** — a ramp from black to white, rotatable in quarter turns.
- **Gradient Radial** — a ramp from the center outward to a given radius.
- **Checker** — an alternating black/white grid.
- **Uniform Color** — a flat, solid color fill.
- **Dither Pattern** — a Bayer, blue noise, or halftone dot mask, for use in Blend, Threshold,
  and similar nodes.

### Filters — Adjustments
- **Levels** — remap an input/output range, with a gamma curve in between.
- **Curve** — a 5-point adjustable response curve.
- **Invert** and **Clamp** — the usual tone-mapping building blocks.
- **HSL Adjust** — shift hue, scale saturation, shift lightness.
- **Gradient Map** — map a grayscale input through a color gradient.
- **Grayscale Conversion** — Color to Gray by luminance.

### Filters — Blending & Masking
- **Blend** — 9 blend modes (Copy, Add, Subtract, Multiply, Screen, Overlay, Max, Min,
  Difference), opacity, and an optional mask.
- **Value Range** — threshold/window an input around a position.
- **Threshold** — a hard black/white cutoff at a given value.

### Filters — Stylize
- **Ordered Dither** — reduce an image to a few tones with a Bayer or blue noise dither
  pattern; Per Channel or hue-preserving Luminance mode.
- **Palette Dither** — dither a Color image down to a 2–8 color palette, with RGB or
  perceptual (luminance-weighted) nearest-color matching.

### Filters — Glitch
- **Block Shift** — seeded random blocks shift sideways.
- **Channel Split** — sample each RGB channel at its own offset, for a chromatic-aberration
  look.
- **Scanline Jitter** — seeded random horizontal or vertical bands shift sideways.
- **Data Corrupt** — posterize, channel-swap, or invert random blocks for a "corrupted file"
  look.

### Filters — Spatial
- **Blur** and **Directional Blur** — separable Gaussian blur, omnidirectional or along one
  fixed angle.
- **Warp** and **Push Warp** — distort an input by the gradient of an intensity map,
  omnidirectional or along a fixed angle.
- **Flow Blur** — blur along a control map's own gradient, strength scaled by its steepness.
- **Transform 2D** — offset, rotate, and scale, with wrap-around tiling.
- **Edge Detect** — highlight where an input changes sharply.

### Converters
- **Height to Normal** — a tangent-space normal map from a height map, Sobel-based, with an
  OpenGL/DirectX toggle.
- **Height to AO** — a screen-space-style ambient occlusion pass from height.
- **Curvature from Normal** — convex edges read bright, concave crevices read dark.

### Outputs
One pass-through marker node per PBR channel: Base Color, Normal, Roughness, Metallic,
Height, Ambient Occlusion, Emissive, Alpha. Alpha drives real transparency in the 3D preview
(not just an exported mask) and exports as its own standalone grayscale map. (A Bevel node is
planned next — see [PROGRESS.md](PROGRESS.md).)

## Building from source

Requires a stable Rust toolchain ([rustup.rs](https://rustup.rs)) and a GPU with a
Vulkan/Metal/DX12 driver (`wgpu`'s supported backends — no OpenGL fallback).

- **macOS**: install the Xcode Command Line Tools if you haven't already
  (`xcode-select --install`). No other setup needed; `wgpu` uses Metal.
- **Windows**: rustup's default `stable-x86_64-pc-windows-msvc` toolchain is enough; it
  prompts to install the Visual Studio Build Tools (the "Desktop development with C++"
  workload) the first time you build if they're missing. `wgpu` uses DX12.
- **Linux**: `wgpu` uses Vulkan, so you need a Vulkan-capable driver (Mesa's RADV/ANV on
  AMD/Intel, or the proprietary driver on Nvidia) plus your distro's windowing dev packages —
  on Debian/Ubuntu: `apt install libxkbcommon-dev libx11-dev libwayland-dev mesa-vulkan-drivers`.

```sh
cargo run -p mc_app   # launch the desktop app
cargo run -p mc_cli   # headless renderer (see CLAUDE.md §8 for usage)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — short version: sign off your commits (`git commit
-s`), keep `cargo fmt`/`cargo clippy`/`cargo test` clean, and don't submit anything derived
from Adobe Substance.

## License

Dual-licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

## Inspiration

MaterialCraft is an independent implementation, built from scratch for this project. It does
not use, reference, or reproduce any other procedural texturing tool's code, file formats,
trademarked names, or UI design — generic, industry-standard terms for node types (Levels,
Blend, Blur, Height to Normal, and so on) are the only overlap you should ever find with any
other tool in this space.

Its broader motivation — that ambitious, professional-grade creative software is worth
building in the open, for artists — takes a cue from [ArtCraft](https://github.com/storytold/artcraft),
an open-source creative tool for artists and filmmakers.

See [INDEPENDENCE.md](INDEPENDENCE.md) for the project's working checklist on staying clearly
independent as it grows.

---

MaterialCraft is an independent open-source project and is not affiliated with, endorsed by,
or sponsored by Adobe. Adobe and Adobe Substance 3D are either registered trademarks or
trademarks of Adobe in the United States and/or other countries.
