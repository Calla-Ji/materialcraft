// SPDX-License-Identifier: MIT OR Apache-2.0
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
    if let Err(message) = tiles_seamlessly(pixels, resolution) {
        panic!("{message}");
    }
}

/// Same check as [`assert_tiles_seamlessly`], returning the failure message
/// instead of panicking — for a node whose Tiling Mode can be turned off,
/// where the useful assertion is "this is no longer seamless", not a fixed
/// numeric comparison against some other render (not every node's
/// wrap-vs-clamp difference points the same direction pixel-for-pixel —
/// e.g. Warp pushes each pixel by a different, signed amount depending on
/// the local gradient, unlike a blur kernel's always-smoothing effect — so
/// re-using this exact seam-vs-interior-baseline definition is the robust
/// way to confirm "doesn't tile anymore" for any of them).
pub fn tiles_seamlessly(pixels: &[u8], resolution: u32) -> Result<(), String> {
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
        if diff > tolerance {
            return Err(format!(
                "left/right seam at y={y}: diff {diff} > interior baseline {max_interior_diff}"
            ));
        }
    }
    for x in 0..r {
        let diff = max_channel_diff(pixel(pixels, r, x, 0), pixel(pixels, r, x, r - 1));
        if diff > tolerance {
            return Err(format!(
                "top/bottom seam at x={x}: diff {diff} > interior baseline {max_interior_diff}"
            ));
        }
    }
    Ok(())
}

/// Path to a file under `tests/golden/<subdir>/` — shared by any node with
/// golden fixtures there (currently Gabor Noise and Scattered Streaks, both
/// under `gabor_streaks`, ported from the owner's own reference package).
pub fn golden_path(subdir: &str, file_name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(subdir)
        .join(file_name)
}

/// Minimal `.npy` reader for a 2D little-endian `<f4` array — exactly what
/// the Gabor/Streaks reference package's `make_sheets.py` wrote its own
/// `.npy` goldens as (confirmed via a direct header dump before writing
/// this). Only handles the one shape those goldens actually use; not a
/// general-purpose parser.
pub fn read_npy_f32_2d(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    assert_eq!(
        &bytes[0..6],
        b"\x93NUMPY",
        "{}: bad npy magic",
        path.display()
    );
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    assert!(
        header.contains("'<f4'"),
        "{}: expected little-endian f32 dtype, got header {header}",
        path.display()
    );
    let data_start = 10 + header_len;
    bytes[data_start..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Loads a 16-bit grayscale PNG golden as `0..1` `f32`s — the Gabor/Streaks
/// reference package's own goldens are stored at 16-bit depth for extra
/// precision even though its own tolerances are stated in 8-bit (`/255`)
/// units. Returns `(values, resolution)`.
pub fn read_png16_as_unit_floats(path: &std::path::Path) -> (Vec<f32>, u32) {
    let img = image::open(path)
        .unwrap_or_else(|err| panic!("failed to open {}: {err}", path.display()))
        .to_luma16();
    let resolution = img.width();
    assert_eq!(img.height(), resolution, "{}: not square", path.display());
    let values = img
        .into_raw()
        .into_iter()
        .map(|v| f32::from(v) / 65535.0)
        .collect();
    (values, resolution)
}

#[cfg(test)]
mod tests {
    use super::{golden_path, read_npy_f32_2d};

    /// Sanity check for the Hard-Surface Nodes package's own `.npy`
    /// goldens (`docs/hardsurface_nodes/golden/`, copied one file at a time
    /// into `tests/golden/hardsurface/` as each build step needs it — see
    /// `docs/hardsurface_nodes/HARDSURFACE_NODES.md` §5). `read_npy_f32_2d`
    /// already exists for the Gabor/Streaks package; this just confirms it
    /// reads this *other* package's `.npy` files correctly too (same
    /// `<f4` little-endian format, confirmed by direct header inspection),
    /// before any node actually depends on it. Min/max are the real values
    /// read straight from this exact file with numpy.
    #[test]
    fn reads_the_hardsurface_df_circle_sdf_golden() {
        let path = golden_path("hardsurface", "df_circle_sdf.npy");
        let values = read_npy_f32_2d(&path);
        assert_eq!(values.len(), 512 * 512, "expected a flat 512x512 array");
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!((min - (-0.298_622_25)).abs() < 1e-5, "min={min}");
        assert!((max - 0.405_725_72).abs() < 1e-5, "max={max}");
    }
}
