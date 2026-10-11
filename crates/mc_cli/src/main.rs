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
        let export_dir = format!("{out_dir}/export_{}", preset.name().to_lowercase());
        let written = mc_nodes::export::export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            preset,
            export_resolution,
            "bricks",
            std::path::Path::new(&export_dir),
        )
        .with_context(|| format!("failed to export {preset:?} preset"))?;
        for path in &written {
            println!("Wrote {}", path.display());
        }
    }

    let demo_graph_path = "examples/graphs/glitch_dither_demo.mcgraph";
    build_glitch_dither_demo_graph().save_to_file(demo_graph_path)?;
    let mut demo_graph = mc_core::Graph::load_from_file(demo_graph_path)
        .with_context(|| format!("failed to load {demo_graph_path}"))?;
    mc_nodes::registry::evaluate(&mut gpu, &mut pool, &mut demo_graph)
        .context("failed to evaluate glitch_dither_demo graph")?;

    let demo_resolution = demo_graph.settings.resolution;
    let demo_base_color_id = demo_graph
        .nodes()
        .find(|(_, node)| node.kind == "output_base_color")
        .map(|(id, _)| id)
        .context("glitch_dither_demo graph has no output_base_color node")?;
    let demo_texture_key = mc_nodes::registry::resolve_texture_key(&demo_graph, demo_base_color_id)
        .context("glitch_dither_demo's Output: Base Color input isn't connected")?;
    let demo_texture = pool
        .texture(&demo_texture_key)
        .context("evaluated glitch_dither_demo texture missing from the pool")?;
    let demo_pixels = gpu.read_rgba8(demo_texture, demo_resolution);
    let demo_image = image::RgbaImage::from_raw(demo_resolution, demo_resolution, demo_pixels)
        .context("pixel buffer size didn't match the image dimensions")?;
    let demo_image_path = format!("{out_dir}/glitch_dither_demo_base_color.png");
    demo_image.save(&demo_image_path)?;
    println!("Wrote {demo_image_path}");
    println!("Wrote {demo_graph_path}");

    Ok(())
}
