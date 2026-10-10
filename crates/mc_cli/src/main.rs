//! mc_cli: headless rendering, with no window.
//!
//! M2: builds a hardcoded 3-node graph (Perlin Noise → Levels → Output),
//! saves it to a `.mcgraph` file and reloads it (exercising mc_core's real
//! save/load path, not just an in-memory round trip), evaluates the reloaded
//! graph via `mc_nodes::registry::evaluate`, and exports the Output node's
//! texture to a PNG.
//!
//! The real `mc_cli render <graph.mcgraph> --res … --preset …` command needs
//! an argument-parsing crate that isn't in CLAUDE.md §3's approved
//! dependency table yet (e.g. `clap`) — ask about adding it before building
//! that out.

use anyhow::Context;

fn build_demo_graph() -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = 512;

    let noise = graph.add_node(
        mc_core::Node::new("perlin_noise")
            .with_param("scale", mc_core::ParamValue::Int(4))
            .with_param("octaves", mc_core::ParamValue::Int(4))
            .with_param("persistence", mc_core::ParamValue::Float(0.5))
            .with_param("seed", mc_core::ParamValue::Seed(0)),
    );
    let levels = graph.add_node(
        mc_core::Node::new("levels").with_param("gamma", mc_core::ParamValue::Float(1.4)),
    );
    let output = graph.add_node(mc_core::Node::new("output_base_color"));

    graph.connect(noise, "output", levels, "input").unwrap();
    graph.connect(levels, "output", output, "input").unwrap();

    graph
}

fn main() -> anyhow::Result<()> {
    let out_dir = "out";
    std::fs::create_dir_all(out_dir)?;

    // Save then reload, to exercise the real file round trip rather than
    // just evaluating the graph we built in memory.
    let graph_path = format!("{out_dir}/demo.mcgraph");
    build_demo_graph().save_to_file(&graph_path)?;
    let mut graph = mc_core::Graph::load_from_file(&graph_path)
        .with_context(|| format!("failed to load {graph_path}"))?;

    let output_node = graph
        .nodes()
        .find(|(_, node)| node.kind == "output_base_color")
        .map(|(id, _)| id)
        .context("demo graph has no output_base_color node")?;

    let mut gpu = mc_gpu::Context::new_headless()?;
    let mut pool = mc_gpu::TexturePool::default();
    mc_nodes::registry::evaluate(&mut gpu, &mut pool, &mut graph)
        .context("failed to evaluate graph")?;

    let texture_key = mc_nodes::registry::resolve_texture_key(&graph, output_node)
        .context("output node's input isn't connected")?;
    let texture = pool
        .texture(&texture_key)
        .context("evaluated output texture missing from the pool")?;

    let resolution = graph.settings.resolution;
    let pixels = gpu.read_rgba8(texture, resolution);
    let image = image::RgbaImage::from_raw(resolution, resolution, pixels)
        .context("pixel buffer size didn't match the image dimensions")?;

    let image_path = format!("{out_dir}/base_color.png");
    image.save(&image_path)?;
    println!("Wrote {graph_path} and {image_path}");

    Ok(())
}
