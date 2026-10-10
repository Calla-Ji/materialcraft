//! mc_cli: headless rendering, with no window.
//!
//! M4: builds the `bricks` example graph (CLAUDE.md's M4 acceptance target),
//! saves it to `examples/graphs/bricks.mcgraph`, reloads it (exercising the
//! real file round trip, not just an in-memory one), evaluates it via
//! `mc_nodes::registry::evaluate`, and exports every Output node's texture
//! to a PNG under `out/` for a quick look.
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

    let out_base_color = graph.add_node(mc_core::Node::new("output_base_color"));
    let out_normal = graph.add_node(mc_core::Node::new("output_normal"));
    let out_height = graph.add_node(mc_core::Node::new("output_height"));

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
    let outputs = [
        ("output_base_color", "bricks_base_color.png"),
        ("output_normal", "bricks_normal.png"),
        ("output_height", "bricks_height.png"),
    ];

    for (kind, file_name) in outputs {
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
    Ok(())
}
