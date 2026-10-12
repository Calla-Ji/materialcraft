# Contributing

Thanks for your interest in MaterialCreator. A few ground rules before you open a PR.

## Independence from Adobe Substance

This project is independent and stays that way. Concretely, for any contribution:

- **Don't submit code, text, icons, or assets derived from Adobe Substance 3D Designer, the
  Substance Automation Toolkit, any Substance SDK, or Adobe's documentation.** That includes
  code you wrote yourself while referencing Substance's source, decompiled output, or
  internal `.sbs`/`.sbsar` file contents.
- Implement new nodes against published, independent sources (academic papers, textbooks,
  permissively-licensed reference implementations) and add an entry to
  [CREDITS.md](CREDITS.md) describing the source.
- Use generic, industry-standard names and terms for new nodes/params — not names borrowed
  from any specific commercial tool's product line.
- No Adobe screenshots, icons, or trademarked names in docs, code comments, or PR
  descriptions.

## Sign-off (DCO)

Every commit must include a `Signed-off-by` line, certifying you wrote it or otherwise have
the right to submit it under this project's license. Use `git commit -s`, which appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

## Project layout

A Cargo workspace: `mc_core` (graph model, no GPU/UI dependency), `mc_gpu` (wgpu device/pipeline
plumbing), `mc_nodes` (one `.rs` + one `.wgsl` compute shader per node, under `crates/mc_nodes/`),
`mc_export` (PNG/EXR writing), `mc_app` (the eframe desktop UI), `mc_cli` (headless rendering,
used by tests). Shared WGSL helpers (hashing, gradients, tiling) live in
`crates/mc_nodes/shaders/common.wgsl` — reuse them rather than duplicating noise/hash code in a
new node's shader.

## Before opening a PR

- Every generator must be periodic over the texture domain, and every filter must sample with
  wrap/repeat addressing — texture tiling is non-negotiable here. New nodes need their own
  tiling and determinism test (same graph + seed = bit-identical output); bug fixes need a
  regression test where practical.
- `cargo fmt --all` and `cargo clippy --workspace -- -D warnings` must pass clean.
- `cargo test --workspace` must pass.
- Keep commits conventional (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`) and scoped to one
  logical change each.

## License

By contributing, you agree your contribution is dual-licensed under MIT OR Apache-2.0, same
as the rest of the project (see [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE)).
