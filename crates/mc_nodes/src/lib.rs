//! mc_nodes: node definitions — metadata, params, and the WGSL compute
//! shader that implements each one. [`registry`] bridges mc_core's generic
//! graph model to actually dispatching each node's GPU work.

pub mod levels;
pub mod perlin;
pub mod registry;
mod shaders;

#[cfg(test)]
mod test_support {
    pub const RESOLUTION: u32 = 64;

    pub fn pixel(pixels: &[u8], resolution: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * resolution + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
    }

    pub fn max_channel_diff(a: [u8; 4], b: [u8; 4]) -> i32 {
        (0..4)
            .map(|c| (a[c] as i32 - b[c] as i32).abs())
            .max()
            .unwrap()
    }

    /// Hard rule #1 (CLAUDE.md §6): every generator/filter must tile
    /// seamlessly. This does *not* mean edge pixels are nearly identical —
    /// adjacent pixels in smooth noise normally differ by a few units. It
    /// means the wrap-around step (pixel(res-1) next to pixel(0), as it
    /// would be when the texture repeats) is no bigger a jump than any
    /// ordinary interior pixel-to-pixel step. A real seam would show up as a
    /// jump far larger than that baseline.
    pub fn assert_tiles_seamlessly(pixels: &[u8], resolution: u32) {
        let r = resolution;
        let mut max_interior_diff = 0;
        for y in 0..r {
            for x in 0..r - 1 {
                let diff = max_channel_diff(pixel(pixels, r, x, y), pixel(pixels, r, x + 1, y));
                max_interior_diff = max_interior_diff.max(diff);
            }
        }
        for x in 0..r {
            for y in 0..r - 1 {
                let diff = max_channel_diff(pixel(pixels, r, x, y), pixel(pixels, r, x, y + 1));
                max_interior_diff = max_interior_diff.max(diff);
            }
        }

        // A little slack on top of the observed interior baseline for 8-bit
        // quantization noise.
        let tolerance = max_interior_diff + 2;

        for y in 0..r {
            let diff = max_channel_diff(pixel(pixels, r, 0, y), pixel(pixels, r, r - 1, y));
            assert!(
                diff <= tolerance,
                "left/right seam at y={y}: diff {diff} > interior baseline {max_interior_diff}"
            );
        }
        for x in 0..r {
            let diff = max_channel_diff(pixel(pixels, r, x, 0), pixel(pixels, r, x, r - 1));
            assert!(
                diff <= tolerance,
                "top/bottom seam at x={x}: diff {diff} > interior baseline {max_interior_diff}"
            );
        }
    }
}

#[cfg(test)]
mod perlin_tests {
    use super::perlin::{dispatch, PerlinParams};
    use super::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &PerlinParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("perlin texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn perlin_noise_tiles_seamlessly() {
        assert_tiles_seamlessly(&render(&PerlinParams::default()), RESOLUTION);
    }

    /// Hard rule #3 (CLAUDE.md §6): same graph + same seed = bit-identical
    /// output on the same GPU.
    #[test]
    fn perlin_noise_is_deterministic() {
        let params = PerlinParams {
            seed: 42,
            ..PerlinParams::default()
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same params + seed produced different pixels"
        );
    }

    /// Changing the seed should actually change the output, otherwise the
    /// determinism test above would pass trivially.
    #[test]
    fn perlin_noise_seed_changes_output() {
        let a = render(&PerlinParams {
            seed: 1,
            ..PerlinParams::default()
        });
        let b = render(&PerlinParams {
            seed: 2,
            ..PerlinParams::default()
        });
        assert_ne!(a, b, "different seeds produced identical pixels");
    }
}

#[cfg(test)]
mod levels_tests {
    use super::levels::{dispatch, LevelsParams};
    use super::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use super::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &LevelsParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams::default(),
        );
        dispatch(&mut gpu, &mut pool, "input", "output", RESOLUTION, params);
        let texture = pool.texture("output").expect("levels texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    /// Levels is a pointwise remap (no neighborhood sampling), so it can't
    /// introduce a seam on its own — but a real implementation bug (e.g.
    /// sampling the wrong pixel) could, so this is still worth checking.
    #[test]
    fn levels_preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(&render(&LevelsParams::default()), RESOLUTION);
    }

    #[test]
    fn levels_is_deterministic() {
        let params = LevelsParams {
            gamma: 1.8,
            ..LevelsParams::default()
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same input + params produced different pixels"
        );
    }

    #[test]
    fn levels_default_params_are_a_no_op() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams::default(),
        );
        let before = gpu.read_rgba8(pool.texture("input").unwrap(), RESOLUTION);

        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output",
            RESOLUTION,
            &LevelsParams::default(),
        );
        let after = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        assert_eq!(
            before, after,
            "identity levels params must not change pixels"
        );
    }
}

#[cfg(test)]
mod registry_tests {
    use super::registry::{evaluate, resolve_texture_key};

    /// CLAUDE.md's M2 acceptance check: a 3-node graph (Noise → Levels →
    /// Output) evaluates headlessly.
    #[test]
    fn evaluates_noise_levels_output_graph() {
        let mut graph = mc_core::Graph::new();
        let noise = graph.add_node(mc_core::Node::new("perlin_noise"));
        let levels = graph.add_node(mc_core::Node::new("levels"));
        let output = graph.add_node(mc_core::Node::new("output_base_color"));
        graph.connect(noise, "output", levels, "input").unwrap();
        graph.connect(levels, "output", output, "input").unwrap();
        graph.settings.resolution = 64;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let key = resolve_texture_key(&graph, output).expect("output resolves to a texture");
        let pixels = gpu.read_rgba8(pool.texture(&key).expect("texture exists"), 64);
        assert_eq!(pixels.len(), 64 * 64 * 4);

        for id in [noise, levels, output] {
            assert!(
                !graph.needs_evaluation(id),
                "node {id:?} should be marked up to date after evaluate()"
            );
        }
    }

    #[test]
    fn re_evaluating_an_unchanged_graph_is_a_cache_hit() {
        let mut graph = mc_core::Graph::new();
        let noise = graph.add_node(mc_core::Node::new("perlin_noise"));
        graph.settings.resolution = 32;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).unwrap();
        assert!(!graph.needs_evaluation(noise));

        // Nothing changed, so a second evaluate() pass should be a no-op —
        // verified indirectly: it must not error and the node stays clean.
        evaluate(&mut gpu, &mut pool, &mut graph).unwrap();
        assert!(!graph.needs_evaluation(noise));
    }
}
