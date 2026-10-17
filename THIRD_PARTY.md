# THIRD_PARTY.md — Third-party formulas referenced (not code copied)

This project doesn't vendor or copy any third-party source code. The entries below are
published formulas/techniques this project's own independent WGSL implementation was checked
against — recorded here, alongside [CREDITS.md](CREDITS.md), as the paper trail for exactly
what was referenced and under what license its publisher released it.

## hg_sdf

- **Source**: [mercury.sexy/hg_sdf](https://mercury.sexy/hg_sdf/), a signed-distance-field
  function library by Mercury Demogroup.
- **License**: MIT.
- **What was referenced**: the mathematical form of the round, chamfer, and groove boolean
  operators (`opUnionRound`/`opUnionChamfer`/`opUnionGroove` and their subtract/intersect
  counterparts) used by the **Shape Boolean** node (`crates/mc_nodes/src/shape_boolean.rs`).
  No GLSL source from hg_sdf was copied — this project's own WGSL compute shader
  (`shaders/shape_boolean.wgsl`) is an independent implementation of the same published
  formulas, parameterized to this project's own node/param conventions.

## Inigo Quilez's smooth-minimum

- **Source**: [iquilezles.org/articles/smin](https://iquilezles.org/articles/smin/), Inigo
  Quilez's widely-cited writeup of smooth-minimum functions for SDF blending.
- **License**: published openly for reference; no formal license statement on the article
  itself, treated here the same way this industry treats it generally — a citable public
  technique, not licensed source to vendor.
- **What was referenced**: the smooth-minimum formula used by Shape Boolean's "Smooth" blend
  mode. Independently implemented in WGSL.
