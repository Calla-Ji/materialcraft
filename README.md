# MaterialCraft

A free, open-source, native node-based procedural material authoring tool, built in Rust —
in the spirit of Substance Designer, but its own thing.

Connect nodes (noises, patterns, filters, blends) into a graph that outputs a seamless,
tileable PBR texture set, previews it live in 2D and 3D, and exports maps ready for
**Blender** and **Unreal Engine**.

> Status: early development (v0.1, pre-release). See [PROGRESS.md](PROGRESS.md) for the
> current state and [CLAUDE.md](CLAUDE.md) for the full project scope and roadmap.

## Building from source

Requires a stable Rust toolchain ([rustup.rs](https://rustup.rs)).

```sh
cargo run -p mc_app   # launch the desktop app
cargo run -p mc_cli   # headless renderer (see CLAUDE.md §8 for usage)
```

## License

Dual-licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

This is an independent implementation inspired by the general category of node-based
procedural texturing tools. It does not use, reference, or reproduce any Adobe Substance
code, file formats, trademarked names, or UI designs.
