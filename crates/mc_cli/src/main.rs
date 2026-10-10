//! mc_cli: headless rendering, with no window.
//!
//! M1: renders the one hardcoded node we have so far (Perlin Noise) to a PNG,
//! proving the GPU pipeline works headlessly. The real `mc_cli render
//! <graph.mcgraph> --res … --preset …` command needs mc_core's graph model
//! (M2) and an argument-parsing crate that isn't in CLAUDE.md §3's approved
//! dependency table yet (e.g. `clap`) — flagged in PROGRESS.md to ask about
//! when M2 gets here.

use anyhow::Context;

fn main() -> anyhow::Result<()> {
    let resolution = 512;

    let mut gpu = mc_gpu::Context::new_headless()?;
    let mut pool = mc_gpu::TexturePool::default();
    let params = mc_nodes::perlin::PerlinParams::default();
    mc_nodes::perlin::dispatch(&mut gpu, &mut pool, "perlin", resolution, &params);

    let texture = pool.texture("perlin").expect("just dispatched above");
    let pixels = gpu.read_rgba8(texture, resolution);
    let image = image::RgbaImage::from_raw(resolution, resolution, pixels)
        .context("pixel buffer size didn't match the image dimensions")?;

    let out_dir = "out";
    std::fs::create_dir_all(out_dir)?;
    let out_path = format!("{out_dir}/perlin_noise.png");
    image.save(&out_path)?;
    println!("Wrote {out_path}");

    Ok(())
}
