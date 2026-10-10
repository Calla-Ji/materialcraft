//! Shared test helpers (CLAUDE.md §11: every node gets a tiling test and a
//! determinism test). Not compiled outside `#[cfg(test)]`.

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

/// Hard rule #1 (CLAUDE.md §6): every generator/filter must tile seamlessly.
/// This does *not* mean edge pixels are nearly identical — adjacent pixels
/// in smooth noise normally differ by a few units. It means the wrap-around
/// step (pixel(res-1) next to pixel(0), as it would be when the texture
/// repeats) is no bigger a jump than any ordinary interior pixel-to-pixel
/// step. A real seam would show up as a jump far larger than that baseline.
///
/// Not appropriate for nodes whose output is an intentionally discontinuous
/// ramp (Gradient Linear/Radial) — see their own tests for why.
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
