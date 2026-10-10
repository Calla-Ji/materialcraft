//! mc_nodes: node definitions — metadata, params, and the WGSL compute
//! shader that implements each one.

pub mod perlin;
mod shaders;

#[cfg(test)]
mod tests {
    use super::perlin::{dispatch, PerlinParams};

    const TEST_RESOLUTION: u32 = 64;

    fn render(params: &PerlinParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", TEST_RESOLUTION, params);
        let texture = pool.texture("test").expect("perlin texture");
        gpu.read_rgba8(texture, TEST_RESOLUTION)
    }

    fn pixel(pixels: &[u8], resolution: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * resolution + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
    }

    fn max_channel_diff(a: [u8; 4], b: [u8; 4]) -> i32 {
        (0..4)
            .map(|c| (a[c] as i32 - b[c] as i32).abs())
            .max()
            .unwrap()
    }

    /// Hard rule #1 (CLAUDE.md §6): every generator must tile seamlessly.
    ///
    /// This does *not* mean edge pixels are nearly identical — adjacent
    /// pixels in smooth noise normally differ by a few units. It means the
    /// wrap-around step (pixel(res-1) next to pixel(0), as it would be when
    /// the texture repeats) is no bigger a jump than any ordinary interior
    /// pixel-to-pixel step. A real seam would show up as a jump far larger
    /// than that baseline.
    #[test]
    fn perlin_noise_tiles_seamlessly() {
        let pixels = render(&PerlinParams::default());
        let r = TEST_RESOLUTION;

        let mut max_interior_diff = 0;
        for y in 0..r {
            for x in 0..r - 1 {
                let diff = max_channel_diff(pixel(&pixels, r, x, y), pixel(&pixels, r, x + 1, y));
                max_interior_diff = max_interior_diff.max(diff);
            }
        }
        for x in 0..r {
            for y in 0..r - 1 {
                let diff = max_channel_diff(pixel(&pixels, r, x, y), pixel(&pixels, r, x, y + 1));
                max_interior_diff = max_interior_diff.max(diff);
            }
        }

        // A little slack on top of the observed interior baseline for 8-bit
        // quantization noise.
        let tolerance = max_interior_diff + 2;

        for y in 0..r {
            let diff = max_channel_diff(pixel(&pixels, r, 0, y), pixel(&pixels, r, r - 1, y));
            assert!(
                diff <= tolerance,
                "left/right seam at y={y}: diff {diff} > interior baseline {max_interior_diff}"
            );
        }
        for x in 0..r {
            let diff = max_channel_diff(pixel(&pixels, r, x, 0), pixel(&pixels, r, x, r - 1));
            assert!(
                diff <= tolerance,
                "top/bottom seam at x={x}: diff {diff} > interior baseline {max_interior_diff}"
            );
        }
    }

    /// Hard rule #3 (CLAUDE.md §6): same graph + same seed = bit-identical
    /// output on the same GPU.
    #[test]
    fn perlin_noise_is_deterministic() {
        let params = PerlinParams {
            seed: 42,
            ..PerlinParams::default()
        };
        let first = render(&params);
        let second = render(&params);
        assert_eq!(
            first, second,
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
