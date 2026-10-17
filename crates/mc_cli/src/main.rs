// SPDX-License-Identifier: MIT OR Apache-2.0
//! mc_cli: headless rendering, with no window.
//!
//! Builds the `bricks` example graph (CLAUDE.md's M4 acceptance target),
//! saves it to `examples/graphs/bricks.mcgraph`, reloads it (exercising the
//! real file round trip, not just an in-memory one), evaluates it via
//! `mc_nodes::registry::evaluate`, dumps a quick-look PNG per output under
//! `out/`, and (M5) exports the full material via `mc_nodes::export` for
//! both the Blender and Unreal presets — proving the naming/packing/
//! color-space pipeline end to end, at a resolution different from the
//! graph's own live-preview resolution.
//!
//! Also builds `glitch_dither_demo` (the Glitch & Dither task brief's §6
//! example graph) the same way — built in code and saved, not hand-authored
//! JSON, so it stays the living source of truth the same way `bricks` does.
//!
//! The real `mc_cli render <graph.mcgraph> --res … --preset …` command needs
//! an argument-parsing crate that isn't in CLAUDE.md §3's approved
//! dependency table yet (e.g. `clap`) — ask about adding it before building
//! that out.

use anyhow::Context;

fn build_bricks_graph() -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = 512;

    let bricks = graph.add_node(
        mc_core::Node::new("tile_generator")
            .with_param("rows", mc_core::ParamValue::Int(6))
            .with_param("columns", mc_core::ParamValue::Int(4))
            .with_param("offset", mc_core::ParamValue::Float(0.5))
            .with_param("gap", mc_core::ParamValue::Float(0.06))
            .with_param("bevel", mc_core::ParamValue::Float(0.1))
            .with_param("size_jitter", mc_core::ParamValue::Float(0.1))
            .with_param("luminance_jitter", mc_core::ParamValue::Float(0.2))
            .with_param("seed", mc_core::ParamValue::Seed(1)),
    );

    let mortar_color = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.42, 0.40, 0.38, 1.0])),
    );
    let brick_color = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.55, 0.28, 0.20, 1.0])),
    );
    let colorize = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Copy".to_string())),
    );

    let variation_noise = graph.add_node(
        mc_core::Node::new("perlin_noise")
            .with_param("scale", mc_core::ParamValue::Int(10))
            .with_param("octaves", mc_core::ParamValue::Int(3))
            .with_param("seed", mc_core::ParamValue::Seed(7)),
    );
    let varied_color = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Multiply".to_string()))
            .with_param("opacity", mc_core::ParamValue::Float(0.2)),
    );

    let normal = graph.add_node(
        mc_core::Node::new("height_to_normal")
            .with_param("intensity", mc_core::ParamValue::Float(2.0)),
    );

    let roughness = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.65, 0.65, 0.65, 1.0])),
    );
    let metallic = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.0, 0.0, 0.0, 1.0])),
    );

    let out_base_color = graph.add_node(mc_core::Node::new("output_base_color"));
    let out_normal = graph.add_node(mc_core::Node::new("output_normal"));
    let out_height = graph.add_node(mc_core::Node::new("output_height"));
    let out_roughness = graph.add_node(mc_core::Node::new("output_roughness"));
    let out_metallic = graph.add_node(mc_core::Node::new("output_metallic"));
    let out_ao = graph.add_node(mc_core::Node::new("output_ambient_occlusion"));

    graph
        .connect(mortar_color, "output", colorize, "base")
        .unwrap();
    graph
        .connect(brick_color, "output", colorize, "blend")
        .unwrap();
    graph.connect(bricks, "output", colorize, "mask").unwrap();

    graph
        .connect(colorize, "output", varied_color, "base")
        .unwrap();
    graph
        .connect(variation_noise, "output", varied_color, "blend")
        .unwrap();
    // `varied_color`'s mask is left unconnected — it falls back to the
    // documented solid-white default, so the Multiply applies everywhere.

    graph.connect(bricks, "output", normal, "input").unwrap();

    graph
        .connect(varied_color, "output", out_base_color, "input")
        .unwrap();
    graph
        .connect(normal, "output", out_normal, "input")
        .unwrap();
    graph
        .connect(bricks, "output", out_height, "input")
        .unwrap();
    graph
        .connect(roughness, "output", out_roughness, "input")
        .unwrap();
    graph
        .connect(metallic, "output", out_metallic, "input")
        .unwrap();
    // The mortar/bricks height pattern doubles as a crude ambient-occlusion
    // proxy: mortar gaps (dark) read as occluded, brick faces (bright) as
    // unoccluded — good enough for a demo graph, not a real AO bake.
    graph.connect(bricks, "output", out_ao, "input").unwrap();

    graph
}

/// The Glitch & Dither task brief's §6 example: `noise -> Gradient Map ->
/// Block Shift -> Channel Split -> Ordered Dither -> Output: Base Color`.
fn build_glitch_dither_demo_graph() -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = 512;

    let noise = graph.add_node(
        mc_core::Node::new("perlin_noise")
            .with_param("scale", mc_core::ParamValue::Int(6))
            .with_param("octaves", mc_core::ParamValue::Int(4))
            .with_param("seed", mc_core::ParamValue::Seed(2)),
    );

    let colorize = graph.add_node(mc_core::Node::new("gradient_map").with_param(
        "gradient",
        mc_core::ParamValue::Gradient(vec![
            mc_core::GradientStop {
                position: 0.0,
                color: [0.05, 0.05, 0.2, 1.0],
            },
            mc_core::GradientStop {
                position: 0.5,
                color: [0.9, 0.45, 0.1, 1.0],
            },
            mc_core::GradientStop {
                position: 1.0,
                color: [1.0, 1.0, 1.0, 1.0],
            },
        ]),
    ));

    let block_shift = graph.add_node(
        mc_core::Node::new("block_shift")
            .with_param("columns", mc_core::ParamValue::Int(12))
            .with_param("rows", mc_core::ParamValue::Int(12))
            .with_param("amount", mc_core::ParamValue::Float(0.25))
            .with_param("max_offset", mc_core::ParamValue::Float(0.08))
            .with_param("direction", mc_core::ParamValue::Enum("Both".to_string()))
            .with_param("seed", mc_core::ParamValue::Seed(3)),
    );

    let channel_split = graph.add_node(
        mc_core::Node::new("channel_split")
            .with_param("offset_r", mc_core::ParamValue::Vec2([0.015, 0.0]))
            .with_param("offset_g", mc_core::ParamValue::Vec2([0.0, 0.0]))
            .with_param("offset_b", mc_core::ParamValue::Vec2([-0.015, 0.0]))
            .with_param("strength", mc_core::ParamValue::Float(0.8)),
    );

    let dither = graph.add_node(
        mc_core::Node::new("ordered_dither")
            .with_param(
                "pattern",
                mc_core::ParamValue::Enum("Bayer 4x4".to_string()),
            )
            .with_param("cells", mc_core::ParamValue::Int(96))
            .with_param("levels", mc_core::ParamValue::Int(5))
            .with_param("spread", mc_core::ParamValue::Float(0.8))
            .with_param("mode", mc_core::ParamValue::Enum("Luminance".to_string())),
    );

    let out_base_color = graph.add_node(mc_core::Node::new("output_base_color"));

    graph.connect(noise, "output", colorize, "input").unwrap();
    graph
        .connect(colorize, "output", block_shift, "image")
        .unwrap();
    graph
        .connect(block_shift, "output", channel_split, "image")
        .unwrap();
    graph
        .connect(channel_split, "output", dither, "image")
        .unwrap();
    graph
        .connect(dither, "output", out_base_color, "input")
        .unwrap();

    graph
}

/// CLAUDE.md §10's M6 example graph: Voronoi "Edges" (cell cracks) blended
/// with a surface fractal, colorized gray/tan by height.
fn build_rock_graph() -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = 512;

    // "Edges" reads bright exactly at a cell boundary and ~0 everywhere
    // else (confirmed by rendering it alone) — the opposite polarity of
    // what a height map wants (cracks recessed/dark, faces raised/bright),
    // so it's used as a *mask* carving black into the surface noise below,
    // not blended directly into the driving signal itself.
    let cracks = graph.add_node(
        mc_core::Node::new("voronoi")
            .with_param("scale", mc_core::ParamValue::Int(8))
            .with_param("jitter", mc_core::ParamValue::Float(0.9))
            .with_param("mode", mc_core::ParamValue::Enum("Edges".to_string()))
            .with_param("seed", mc_core::ParamValue::Seed(11)),
    );
    let surface_noise = graph.add_node(
        mc_core::Node::new("fractal_sum")
            .with_param("scale", mc_core::ParamValue::Int(6))
            .with_param("octaves", mc_core::ParamValue::Int(5))
            .with_param("gain", mc_core::ParamValue::Float(0.55))
            .with_param("seed", mc_core::ParamValue::Seed(12)),
    );
    let black = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.0, 0.0, 0.0, 1.0])),
    );
    let combined = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Multiply".to_string()))
            .with_param("opacity", mc_core::ParamValue::Float(0.85)),
    );
    graph
        .connect(surface_noise, "output", combined, "base")
        .unwrap();
    graph.connect(black, "output", combined, "blend").unwrap();
    graph.connect(cracks, "output", combined, "mask").unwrap();

    let colorize = graph.add_node(mc_core::Node::new("gradient_map").with_param(
        "gradient",
        mc_core::ParamValue::Gradient(vec![
            mc_core::GradientStop {
                position: 0.0,
                color: [0.12, 0.11, 0.10, 1.0],
            },
            mc_core::GradientStop {
                position: 0.5,
                color: [0.35, 0.33, 0.30, 1.0],
            },
            mc_core::GradientStop {
                position: 1.0,
                color: [0.58, 0.55, 0.49, 1.0],
            },
        ]),
    ));
    graph
        .connect(combined, "output", colorize, "input")
        .unwrap();

    let normal = graph.add_node(
        mc_core::Node::new("height_to_normal")
            .with_param("intensity", mc_core::ParamValue::Float(1.5)),
    );
    graph.connect(combined, "output", normal, "input").unwrap();

    let roughness = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.85, 0.85, 0.85, 1.0])),
    );
    let metallic = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.0, 0.0, 0.0, 1.0])),
    );

    let out_base_color = graph.add_node(mc_core::Node::new("output_base_color"));
    let out_normal = graph.add_node(mc_core::Node::new("output_normal"));
    let out_height = graph.add_node(mc_core::Node::new("output_height"));
    let out_roughness = graph.add_node(mc_core::Node::new("output_roughness"));
    let out_metallic = graph.add_node(mc_core::Node::new("output_metallic"));
    let out_ao = graph.add_node(mc_core::Node::new("output_ambient_occlusion"));

    graph
        .connect(colorize, "output", out_base_color, "input")
        .unwrap();
    graph
        .connect(normal, "output", out_normal, "input")
        .unwrap();
    graph
        .connect(combined, "output", out_height, "input")
        .unwrap();
    graph
        .connect(roughness, "output", out_roughness, "input")
        .unwrap();
    graph
        .connect(metallic, "output", out_metallic, "input")
        .unwrap();
    // Crevices between rock faces (where cracks/surface noise are dark) read
    // as occluded — the same crude-but-plausible proxy `bricks` uses.
    graph.connect(combined, "output", out_ao, "input").unwrap();

    graph
}

/// CLAUDE.md §10's M6 example graph: a directionally-blurred fractal noise
/// gives both the brush streaks and a matching narrow-band roughness
/// variation, over a fully metallic base.
fn build_brushed_metal_graph() -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = 512;

    let noise = graph.add_node(
        mc_core::Node::new("fractal_sum")
            .with_param("scale", mc_core::ParamValue::Int(24))
            .with_param("octaves", mc_core::ParamValue::Int(3))
            .with_param("gain", mc_core::ParamValue::Float(0.5))
            .with_param("seed", mc_core::ParamValue::Seed(21)),
    );
    let brushed = graph.add_node(
        mc_core::Node::new("directional_blur")
            .with_param("radius", mc_core::ParamValue::Float(0.04))
            .with_param("angle", mc_core::ParamValue::Float(0.0)),
    );
    graph.connect(noise, "output", brushed, "input").unwrap();

    let steel_tint = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.72, 0.73, 0.75, 1.0])),
    );
    let colorize = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Multiply".to_string()))
            .with_param("opacity", mc_core::ParamValue::Float(0.4)),
    );
    graph
        .connect(steel_tint, "output", colorize, "base")
        .unwrap();
    graph.connect(brushed, "output", colorize, "blend").unwrap();

    let normal = graph.add_node(
        mc_core::Node::new("height_to_normal")
            .with_param("intensity", mc_core::ParamValue::Float(0.6)),
    );
    graph.connect(brushed, "output", normal, "input").unwrap();

    // Remaps the same brush-streak noise into a narrow roughness band
    // instead of a flat value — streaks read as faint variation in
    // reflection sharpness, not just in the normal map.
    let roughness = graph.add_node(
        mc_core::Node::new("levels")
            .with_param("out_low", mc_core::ParamValue::Float(0.25))
            .with_param("out_high", mc_core::ParamValue::Float(0.45)),
    );
    graph
        .connect(brushed, "output", roughness, "input")
        .unwrap();

    let metallic = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([1.0, 1.0, 1.0, 1.0])),
    );
    let ao = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([1.0, 1.0, 1.0, 1.0])),
    );

    let out_base_color = graph.add_node(mc_core::Node::new("output_base_color"));
    let out_normal = graph.add_node(mc_core::Node::new("output_normal"));
    let out_height = graph.add_node(mc_core::Node::new("output_height"));
    let out_roughness = graph.add_node(mc_core::Node::new("output_roughness"));
    let out_metallic = graph.add_node(mc_core::Node::new("output_metallic"));
    let out_ao = graph.add_node(mc_core::Node::new("output_ambient_occlusion"));

    graph
        .connect(colorize, "output", out_base_color, "input")
        .unwrap();
    graph
        .connect(normal, "output", out_normal, "input")
        .unwrap();
    graph
        .connect(brushed, "output", out_height, "input")
        .unwrap();
    graph
        .connect(roughness, "output", out_roughness, "input")
        .unwrap();
    graph
        .connect(metallic, "output", out_metallic, "input")
        .unwrap();
    graph.connect(ao, "output", out_ao, "input").unwrap();

    graph
}

/// CLAUDE.md §10's M6 example graph: a small woven grid (Pattern Generator)
/// overlaid with fine fiber noise, colorized as denim.
fn build_fabric_graph() -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = 512;

    let weave = graph.add_node(
        mc_core::Node::new("tile_generator")
            .with_param("rows", mc_core::ParamValue::Int(48))
            .with_param("columns", mc_core::ParamValue::Int(48))
            .with_param("offset", mc_core::ParamValue::Float(0.5))
            .with_param("gap", mc_core::ParamValue::Float(0.12))
            .with_param("bevel", mc_core::ParamValue::Float(0.3))
            .with_param("seed", mc_core::ParamValue::Seed(31)),
    );
    let fiber_noise = graph.add_node(
        mc_core::Node::new("fractal_sum")
            .with_param("scale", mc_core::ParamValue::Int(32))
            .with_param("octaves", mc_core::ParamValue::Int(4))
            .with_param("gain", mc_core::ParamValue::Float(0.5))
            .with_param("seed", mc_core::ParamValue::Seed(32)),
    );
    let combined = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Overlay".to_string()))
            .with_param("opacity", mc_core::ParamValue::Float(0.3)),
    );
    graph.connect(weave, "output", combined, "base").unwrap();
    graph
        .connect(fiber_noise, "output", combined, "blend")
        .unwrap();

    let colorize = graph.add_node(mc_core::Node::new("gradient_map").with_param(
        "gradient",
        mc_core::ParamValue::Gradient(vec![
            mc_core::GradientStop {
                position: 0.0,
                color: [0.10, 0.16, 0.30, 1.0],
            },
            mc_core::GradientStop {
                position: 1.0,
                color: [0.22, 0.32, 0.52, 1.0],
            },
        ]),
    ));
    graph
        .connect(combined, "output", colorize, "input")
        .unwrap();

    let normal = graph.add_node(
        mc_core::Node::new("height_to_normal")
            .with_param("intensity", mc_core::ParamValue::Float(0.8)),
    );
    graph.connect(combined, "output", normal, "input").unwrap();

    let roughness = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.9, 0.9, 0.9, 1.0])),
    );
    let metallic = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.0, 0.0, 0.0, 1.0])),
    );

    let out_base_color = graph.add_node(mc_core::Node::new("output_base_color"));
    let out_normal = graph.add_node(mc_core::Node::new("output_normal"));
    let out_height = graph.add_node(mc_core::Node::new("output_height"));
    let out_roughness = graph.add_node(mc_core::Node::new("output_roughness"));
    let out_metallic = graph.add_node(mc_core::Node::new("output_metallic"));
    let out_ao = graph.add_node(mc_core::Node::new("output_ambient_occlusion"));

    graph
        .connect(colorize, "output", out_base_color, "input")
        .unwrap();
    graph
        .connect(normal, "output", out_normal, "input")
        .unwrap();
    graph
        .connect(combined, "output", out_height, "input")
        .unwrap();
    graph
        .connect(roughness, "output", out_roughness, "input")
        .unwrap();
    graph
        .connect(metallic, "output", out_metallic, "input")
        .unwrap();
    graph.connect(combined, "output", out_ao, "input").unwrap();

    graph
}

/// CLAUDE.md §10's M6 example graph: a fractal noise squashed heavily along
/// one axis (Transform 2D) into grain lines, colorized brown.
fn build_wood_graph() -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = 512;

    let grain_noise = graph.add_node(
        mc_core::Node::new("fractal_sum")
            .with_param("scale", mc_core::ParamValue::Int(5))
            .with_param("octaves", mc_core::ParamValue::Int(5))
            .with_param("gain", mc_core::ParamValue::Float(0.5))
            .with_param("seed", mc_core::ParamValue::Seed(41)),
    );
    let grain = graph.add_node(
        mc_core::Node::new("transform2d")
            .with_param("scale_x", mc_core::ParamValue::Float(6.0))
            .with_param("scale_y", mc_core::ParamValue::Float(1.0)),
    );
    graph
        .connect(grain_noise, "output", grain, "input")
        .unwrap();

    let colorize = graph.add_node(mc_core::Node::new("gradient_map").with_param(
        "gradient",
        mc_core::ParamValue::Gradient(vec![
            mc_core::GradientStop {
                position: 0.0,
                color: [0.20, 0.11, 0.05, 1.0],
            },
            mc_core::GradientStop {
                position: 0.5,
                color: [0.42, 0.26, 0.13, 1.0],
            },
            mc_core::GradientStop {
                position: 1.0,
                color: [0.65, 0.46, 0.28, 1.0],
            },
        ]),
    ));
    graph.connect(grain, "output", colorize, "input").unwrap();

    let normal = graph.add_node(
        mc_core::Node::new("height_to_normal")
            .with_param("intensity", mc_core::ParamValue::Float(1.0)),
    );
    graph.connect(grain, "output", normal, "input").unwrap();

    let roughness = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.5, 0.5, 0.5, 1.0])),
    );
    let metallic = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.0, 0.0, 0.0, 1.0])),
    );

    let out_base_color = graph.add_node(mc_core::Node::new("output_base_color"));
    let out_normal = graph.add_node(mc_core::Node::new("output_normal"));
    let out_height = graph.add_node(mc_core::Node::new("output_height"));
    let out_roughness = graph.add_node(mc_core::Node::new("output_roughness"));
    let out_metallic = graph.add_node(mc_core::Node::new("output_metallic"));
    let out_ao = graph.add_node(mc_core::Node::new("output_ambient_occlusion"));

    graph
        .connect(colorize, "output", out_base_color, "input")
        .unwrap();
    graph
        .connect(normal, "output", out_normal, "input")
        .unwrap();
    graph.connect(grain, "output", out_height, "input").unwrap();
    graph
        .connect(roughness, "output", out_roughness, "input")
        .unwrap();
    graph
        .connect(metallic, "output", out_metallic, "input")
        .unwrap();
    graph.connect(grain, "output", out_ao, "input").unwrap();

    graph
}

/// Hard-Surface Nodes' own demo chain (`docs/hardsurface_nodes/
/// HARDSURFACE_NODES.md` §4): Panel Layout -> a per-panel 0.75/1.0/1.25
/// height variety (`lvl`, a Gradient Map with razor-thin transition
/// bands — not Blend, since `blend.wgsl`'s own `combine()` result is
/// always clamped to `[0, 1]` regardless of mode, so no chain of Blend
/// nodes could ever reach `1.25` at all; `graph.connect` itself does no
/// port-type validation, confirmed by reading `mc_core::Graph::connect`
/// directly, so Gradient Map's declared-`Color` output can freely feed
/// an `Sdf`/`Gray` input the same way the editor's own Gray<->Color
/// broadcast already treats them) -> Bevel Profile -> Fasteners seated
/// on the beveled, leveled height -> Edge Wear. Verified against
/// `golden/demo_height.png` by `tests/hardsurface_demo.rs`.
fn build_hardsurface_demo_graph() -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = 512;

    let panels = graph.add_node(mc_core::Node::new("panel_layout"));

    // lvl = 0.75 / 1.0 / 1.25 below/within/above the [0.25, 0.85] band,
    // matching the reference's own `where(pid<0.25, 0.75, where(pid>0.85,
    // 1.25, 1.0))`. A single Gradient Map with razor-thin transition bands
    // (width 0.0002, far narrower than any gap between this seed's 18
    // panels' own hashed IDs could ever land inside) reproduces that hard
    // step directly — deliberately *not* Blend for this: `blend.wgsl`'s
    // own `combine()` result is always clamped to `[0, 1]` regardless of
    // mode, so no chain of Blend nodes can ever produce `1.25` at all.
    // Gradient Map's own interpolated color has no such clamp.
    let lvl = graph.add_node(mc_core::Node::new("gradient_map").with_param(
        "gradient",
        mc_core::ParamValue::Gradient(vec![
            mc_core::GradientStop {
                position: 0.0,
                color: [0.75, 0.75, 0.75, 1.0],
            },
            mc_core::GradientStop {
                position: 0.2499,
                color: [0.75, 0.75, 0.75, 1.0],
            },
            mc_core::GradientStop {
                position: 0.2501,
                color: [1.0, 1.0, 1.0, 1.0],
            },
            mc_core::GradientStop {
                position: 0.8499,
                color: [1.0, 1.0, 1.0, 1.0],
            },
            mc_core::GradientStop {
                position: 0.8501,
                color: [1.25, 1.25, 1.25, 1.0],
            },
            mc_core::GradientStop {
                position: 1.0,
                color: [1.25, 1.25, 1.25, 1.0],
            },
        ]),
    ));
    graph.connect(panels, "panel_id", lvl, "input").unwrap();

    // Depth: 0.5 pre-folds the reference's own separate `* 0.5` — height
    // scales linearly with depth (base = 0), so bevel(depth=0.5) ==
    // bevel(depth=1.0) * 0.5 exactly.
    let bevel = graph.add_node(
        mc_core::Node::new("bevel_profile")
            .with_param("preset", mc_core::ParamValue::Enum("Round".to_string()))
            .with_param("width", mc_core::ParamValue::Float(0.006))
            .with_param("wall", mc_core::ParamValue::Float(0.35))
            .with_param("depth", mc_core::ParamValue::Float(0.5)),
    );
    graph.connect(panels, "sdf", bevel, "sdf").unwrap();

    let leveled_height = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Multiply".to_string())),
    );
    graph
        .connect(bevel, "height", leveled_height, "base")
        .unwrap();
    graph
        .connect(lvl, "output", leveled_height, "blend")
        .unwrap();

    let fasteners = graph.add_node(
        mc_core::Node::new("fasteners")
            .with_param("kind", mc_core::ParamValue::Enum("Rivet Dome".to_string()))
            .with_param("spacing", mc_core::ParamValue::Float(0.045))
            .with_param("head_radius", mc_core::ParamValue::Float(0.0055))
            .with_param("head_height", mc_core::ParamValue::Float(0.22))
            .with_param("seed", mc_core::ParamValue::Seed(3))
            .with_param("missing", mc_core::ParamValue::Float(0.04)),
    );
    graph.connect(panels, "path", fasteners, "path").unwrap();
    graph
        .connect(leveled_height, "output", fasteners, "base")
        .unwrap();

    let wear = graph.add_node(
        mc_core::Node::new("edge_wear").with_param("amount", mc_core::ParamValue::Float(0.45)),
    );
    graph.connect(panels, "sdf", wear, "sdf").unwrap();
    graph.connect(fasteners, "height", wear, "height").unwrap();

    let normal = graph.add_node(mc_core::Node::new("height_to_normal"));
    graph.connect(wear, "height", normal, "input").unwrap();
    let ao = graph.add_node(mc_core::Node::new("height_to_ao"));
    graph.connect(wear, "height", ao, "input").unwrap();
    let base_color = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.6, 0.6, 0.62, 1.0])),
    );

    let out_base_color = graph.add_node(mc_core::Node::new("output_base_color"));
    let out_normal = graph.add_node(mc_core::Node::new("output_normal"));
    let out_height = graph.add_node(mc_core::Node::new("output_height"));
    let out_ao = graph.add_node(mc_core::Node::new("output_ambient_occlusion"));
    graph
        .connect(base_color, "output", out_base_color, "input")
        .unwrap();
    graph
        .connect(normal, "output", out_normal, "input")
        .unwrap();
    graph.connect(wear, "height", out_height, "input").unwrap();
    graph.connect(ao, "output", out_ao, "input").unwrap();

    graph
}

/// Saves `graph` to `examples/graphs/{name}.mcgraph`, reloads it (the real
/// file round trip, not just in-memory), evaluates it, and writes its
/// Output: Base Color to `{out_dir}/{name}_base_color.png` — the common
/// "example graph" lifecycle every graph after `bricks` follows (`bricks`
/// keeps its own hand-written version above since it additionally exercises
/// the full multi-channel export pipeline, not just a quick-look).
fn save_evaluate_and_render_base_color(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    name: &str,
    graph: mc_core::Graph,
    out_dir: &str,
) -> anyhow::Result<()> {
    let graph_path = format!("examples/graphs/{name}.mcgraph");
    graph.save_to_file(&graph_path)?;
    let mut graph = mc_core::Graph::load_from_file(&graph_path)
        .with_context(|| format!("failed to load {graph_path}"))?;
    mc_nodes::registry::evaluate(gpu, pool, &mut graph)
        .with_context(|| format!("failed to evaluate {name}"))?;

    let resolution = graph.settings.resolution;
    let node_id = graph
        .nodes()
        .find(|(_, node)| node.kind == "output_base_color")
        .map(|(id, _)| id)
        .with_context(|| format!("{name} graph has no output_base_color node"))?;
    let key = mc_nodes::registry::resolve_texture_key(&graph, node_id)
        .with_context(|| format!("{name}'s Output: Base Color input isn't connected"))?;
    let texture = pool
        .texture(&key)
        .context("evaluated output texture missing from the pool")?;
    let pixels = gpu.read_rgba8(texture, resolution);
    let image = image::RgbaImage::from_raw(resolution, resolution, pixels)
        .context("pixel buffer size didn't match the image dimensions")?;
    let image_path = format!("{out_dir}/{name}_base_color.png");
    image.save(&image_path)?;
    println!("Wrote {image_path}");
    println!("Wrote {graph_path}");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let out_dir = "out";
    std::fs::create_dir_all(out_dir)?;

    let graph_path = "examples/graphs/bricks.mcgraph";
    build_bricks_graph().save_to_file(graph_path)?;
    let mut graph = mc_core::Graph::load_from_file(graph_path)
        .with_context(|| format!("failed to load {graph_path}"))?;

    let mut gpu = mc_gpu::Context::new_headless()?;
    let mut pool = mc_gpu::TexturePool::default();
    mc_nodes::registry::evaluate(&mut gpu, &mut pool, &mut graph)
        .context("failed to evaluate graph")?;

    let resolution = graph.settings.resolution;
    let quick_looks = [
        ("output_base_color", "bricks_base_color.png"),
        ("output_normal", "bricks_normal.png"),
        ("output_height", "bricks_height.png"),
    ];

    for (kind, file_name) in quick_looks {
        let node_id = graph
            .nodes()
            .find(|(_, node)| node.kind == kind)
            .map(|(id, _)| id)
            .with_context(|| format!("bricks graph has no {kind} node"))?;
        let texture_key = mc_nodes::registry::resolve_texture_key(&graph, node_id)
            .with_context(|| format!("{kind}'s input isn't connected"))?;
        let texture = pool
            .texture(&texture_key)
            .context("evaluated output texture missing from the pool")?;

        let pixels = gpu.read_rgba8(texture, resolution);
        let image = image::RgbaImage::from_raw(resolution, resolution, pixels)
            .context("pixel buffer size didn't match the image dimensions")?;
        let image_path = format!("{out_dir}/{file_name}");
        image.save(&image_path)?;
        println!("Wrote {image_path}");
    }
    println!("Wrote {graph_path}");

    // Full material export, both presets, at a resolution different from
    // the graph's own live-preview resolution (512) to prove that path.
    let export_resolution = 1024;
    for preset in [
        mc_export::naming::Preset::Blender,
        mc_export::naming::Preset::Unreal,
    ] {
        // A filesystem-safe directory slug, independent of `preset.name()`'s
        // own UI-facing display string (which can contain spaces, e.g.
        // "Unreal Engine").
        let slug = match preset {
            mc_export::naming::Preset::Blender => "blender",
            mc_export::naming::Preset::Unreal => "unreal",
        };
        let export_dir = format!("{out_dir}/export_{slug}");
        // Exercise the full optional-output surface for Unreal so this
        // smoke test actually covers it, not just the required outputs.
        let options = match preset {
            mc_export::naming::Preset::Unreal => mc_nodes::export::ExportOptions {
                opacity: mc_nodes::export::OpacityMode::SeparateFile,
                ..mc_nodes::export::ExportOptions::default()
            },
            mc_export::naming::Preset::Blender => mc_nodes::export::ExportOptions::default(),
        };
        let written = mc_nodes::export::export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            preset,
            export_resolution,
            "bricks",
            std::path::Path::new(&export_dir),
            &options,
        )
        .with_context(|| format!("failed to export {preset:?} preset"))?;
        for path in &written {
            println!("Wrote {}", path.display());
        }
    }

    save_evaluate_and_render_base_color(
        &mut gpu,
        &mut pool,
        "glitch_dither_demo",
        build_glitch_dither_demo_graph(),
        out_dir,
    )?;

    for (name, graph) in [
        ("rock", build_rock_graph()),
        ("brushed_metal", build_brushed_metal_graph()),
        ("fabric", build_fabric_graph()),
        ("wood", build_wood_graph()),
        ("hardsurface_demo", build_hardsurface_demo_graph()),
    ] {
        save_evaluate_and_render_base_color(&mut gpu, &mut pool, name, graph, out_dir)?;
    }

    Ok(())
}
