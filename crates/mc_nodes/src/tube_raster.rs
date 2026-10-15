// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared GPU building block: rasterize a round tube (the union of spheres
//! swept along a curve) into a height map, keeping the physically highest
//! surface at every pixel. KNIT_NODE_SPEC.md §5 asks for this as its own
//! module, not inlined into [`crate::knit`], because the Knit node's own
//! spec names a planned future node ("Bouclé") that reuses the same
//! "strands occupy space, the one on top wins" primitive — Knit's curves
//! just happen to be loops instead of helices.
//!
//! Deliberately **not** reusing Thread's per-pixel analytic approach
//! ([`crate::thread_shape`]'s `tube_top`): that shader walks every strand
//! and solves one quadratic per pixel because a thread's cross-section at
//! a given x is always a single point — "one centerline per column" is
//! well-defined. A knit loop's x overshoots and backtracks (Crane's `a`
//! term), so for a given pixel column there can be two or three points on
//! the *same* loop passing near it; there's no per-pixel closed form.
//! Instead (KNIT_NODE_SPEC.md §5's own "Recommended GPU version"):
//!
//! 1. The caller samples a curve densely (< 1px spacing) and turns
//!    consecutive samples into short capsule segments ([`Segment`]).
//! 2. [`bin_segments`] (CPU, deterministic, no atomics) buckets every
//!    segment into a uniform grid of cells over the `[0,1)²` tile, with
//!    wrap-around handled by taking cell indices modulo the grid size
//!    rather than by replicating or clamping coordinates.
//! 3. The compute shader looks up each pixel's own cell plus its 3×3
//!    neighborhood (always enough: [`choose_grid_n`] sizes cells to
//!    several capsule-radii across, and a capsule is built short on
//!    purpose), tests every segment found there, and keeps the highest
//!    `z + sqrt(r² - d²)` — the exact formula in KNIT_NODE_SPEC.md §5.
//!
//! Tile wrap-around for the *height test itself* (not just cell lookup)
//! uses the minimum-image convention from periodic simulation: a segment
//! near one tile edge and a pixel near the opposite edge are compared via
//! `delta - round(delta)`, so it doesn't matter which "unwrapped copy" of
//! a periodic coordinate a segment's own stored numbers happen to be in
//! (see `tube_raster.wgsl`'s `min_image`) — this is what makes the one
//! real wrap-around case (the segment closing a course's loop back to its
//! own first sample) and ordinary inter-course tiling both fall out of
//! the same code path, with no special-casing.

const NODE_KIND: &str = "tube_raster";

/// One capsule: a round tube of (possibly linearly tapering, though every
/// caller so far uses a constant radius) radius between two endpoints, each
/// carrying its own z (how far it dips toward/away from the viewer) and an
/// opaque `tag` the caller can pack however it likes — [`crate::knit`]
/// packs `(course * wales + wale) * 8 + ply`, matching `knit_reference.py`'s
/// own `render()` exactly, so Step 5's Stitch ID/Type can unpack it later.
#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub p0: [f32; 3],
    pub r0: f32,
    pub p1: [f32; 3],
    pub r1: f32,
    pub tag: u32,
}

/// Grid cells several capsule-radii wide: big enough that any one capsule
/// (built short, by convention < 0.5 R — see [`Segment`]'s own doc comment)
/// can never reach past its immediate 3×3 cell neighborhood, so the shader
/// never needs to search further than that fixed window. Clamped to a
/// sane range so a pathological (near-zero or huge) radius can't produce an
/// unusably fine or coarse grid.
fn choose_grid_n(max_radius: f32) -> u32 {
    let cell_size = (4.0 * max_radius).max(1.0 / 256.0);
    (1.0 / cell_size).round().clamp(4.0, 256.0) as u32
}

/// Buckets `segments` into a `grid_n x grid_n` uniform grid over the
/// `[0,1)²` tile (a segment whose expanded bounding box straddles more than
/// one cell is inserted into each one it touches), wrapping cell indices
/// with Euclidean modulo so a segment near one tile edge is found by a
/// pixel query near the opposite edge. Returns a CSR pair: `offsets[cell]`
/// through `offsets[cell+1]` indexes into `indices`, which lists that
/// cell's segment indices in a fixed (ascending) order — the fixed order is
/// what makes the shader's "first one wins on a height tie" rule
/// deterministic run to run, with no atomics (KNIT_NODE_SPEC.md §5).
fn bin_segments(segments: &[Segment], grid_n: u32) -> (Vec<u32>, Vec<u32>) {
    let n = grid_n as i64;
    let cell_size = 1.0 / grid_n as f32;
    let wrap = |i: i64| -> i64 { ((i % n) + n) % n };

    let mut bins: Vec<Vec<u32>> = vec![Vec::new(); (grid_n * grid_n) as usize];
    for (idx, seg) in segments.iter().enumerate() {
        let r = seg.r0.max(seg.r1);
        let xmin = seg.p0[0].min(seg.p1[0]) - r;
        let xmax = seg.p0[0].max(seg.p1[0]) + r;
        let ymin = seg.p0[1].min(seg.p1[1]) - r;
        let ymax = seg.p0[1].max(seg.p1[1]) + r;
        let cx0 = (xmin / cell_size).floor() as i64;
        let cx1 = (xmax / cell_size).floor() as i64;
        let cy0 = (ymin / cell_size).floor() as i64;
        let cy1 = (ymax / cell_size).floor() as i64;
        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                let cell = (wrap(cy) * n + wrap(cx)) as usize;
                bins[cell].push(idx as u32);
            }
        }
    }

    let mut offsets = Vec::with_capacity(bins.len() + 1);
    let mut indices = Vec::new();
    offsets.push(0u32);
    for bin in &bins {
        indices.extend_from_slice(bin);
        offsets.push(indices.len() as u32);
    }
    (offsets, indices)
}

fn pack_segments(segments: &[Segment]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(segments.len() * 36);
    for s in segments {
        bytes.extend_from_slice(&s.p0[0].to_le_bytes());
        bytes.extend_from_slice(&s.p0[1].to_le_bytes());
        bytes.extend_from_slice(&s.p0[2].to_le_bytes());
        bytes.extend_from_slice(&s.r0.to_le_bytes());
        bytes.extend_from_slice(&s.p1[0].to_le_bytes());
        bytes.extend_from_slice(&s.p1[1].to_le_bytes());
        bytes.extend_from_slice(&s.p1[2].to_le_bytes());
        bytes.extend_from_slice(&s.r1.to_le_bytes());
        bytes.extend_from_slice(&s.tag.to_le_bytes());
    }
    bytes
}

fn pack_u32_slice(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn pack_params(resolution: u32, grid_n: u32, total_plies: u32, zmax: f32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&grid_n.to_le_bytes());
    bytes[8..12].copy_from_slice(&total_plies.to_le_bytes());
    bytes[12..16].copy_from_slice(&zmax.to_le_bytes());
    bytes
}

/// Where [`dispatch`] stores its four outputs. `ply_id` is `(ply + 1) /
/// total_plies` wherever a pixel is covered (so it's `1.0` everywhere for a
/// single-strand caller, exactly matching `knit_reference.py`'s own
/// `ply01 = mask.astype(float)` fallback when `plies <= 1`), `0.0`
/// elsewhere. `cell` is the winning segment's raw `tag` (or
/// [`CELL_MISS`] where nothing was hit) — an `R32Uint` texture, not an
/// `Rgba16Float` one like the others, so a caller can decode its own
/// tag-packing scheme exactly as a raw integer rather than through a
/// normalized float channel. Knit's own stitch-info pass
/// (`knit_stitch_info.wgsl`) is the reason this exists: it needs the
/// *wale and course* a pixel's winning capsule belongs to, which only the
/// raw tag carries.
pub struct RasterOutputKeys<'a> {
    pub height: &'a str,
    pub mask: &'a str,
    pub ply_id: &'a str,
    pub cell: &'a str,
}

/// Sentinel `cell` value for a pixel no segment covered.
pub const CELL_MISS: u32 = 0xFFFFFFFF;

/// Rasterize `segments` into `keys`' three textures at `resolution`.
///
/// `total_plies` and `zmax` are the caller's own concern, not derived here:
/// `zmax` is the fixed height-normalization constant from the caller's own
/// geometry solve (KNIT_NODE_SPEC.md §5 — "Do not normalize height per
/// image"), and `total_plies` only has to be consistent with whatever ply
/// index each segment's own `tag` packs in its low 3 bits (`tag % 8`).
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    keys: RasterOutputKeys,
    resolution: u32,
    segments: &[Segment],
    total_plies: u32,
    zmax: f32,
) {
    assert!(
        !segments.is_empty(),
        "tube_raster::dispatch needs at least one segment"
    );

    let shader_source = crate::shaders::full_source("tube_raster.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let max_radius = segments
        .iter()
        .fold(0.0f32, |acc, s| acc.max(s.r0).max(s.r1));
    let grid_n = choose_grid_n(max_radius);
    let (cell_offsets, cell_indices) = bin_segments(segments, grid_n);

    let param_bytes = pack_params(resolution, grid_n, total_plies.max(1), zmax);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tube_raster_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let segment_bytes = pack_segments(segments);
    let segment_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tube_raster_segments"),
        size: segment_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&segment_buffer, 0, &segment_bytes);

    let offsets_bytes = pack_u32_slice(&cell_offsets);
    let offsets_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tube_raster_cell_offsets"),
        size: offsets_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&offsets_buffer, 0, &offsets_bytes);

    let indices_bytes = pack_u32_slice(&cell_indices);
    let indices_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tube_raster_cell_indices"),
        // A grid cell with no segments at all is still legal (the CSR range
        // is simply empty), but wgpu rejects a zero-sized buffer outright —
        // pad to at least 4 bytes (one unread u32) for that case.
        size: (indices_bytes.len() as u64).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !indices_bytes.is_empty() {
        gpu.queue.write_buffer(&indices_buffer, 0, &indices_bytes);
    }

    let height_view = pool
        .get_or_create(
            &gpu.device,
            keys.height,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let mask_view = pool
        .get_or_create(
            &gpu.device,
            keys.mask,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let ply_id_view = pool
        .get_or_create(
            &gpu.device,
            keys.ply_id,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let cell_view = pool
        .get_or_create(
            &gpu.device,
            keys.cell,
            resolution,
            wgpu::TextureFormat::R32Uint,
        )
        .clone();

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tube_raster_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: segment_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: indices_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&height_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&mask_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&ply_id_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&cell_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tube_raster_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tube_raster_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_grid_n_stays_within_its_clamp_range() {
        assert_eq!(choose_grid_n(0.0), 256);
        assert_eq!(choose_grid_n(10.0), 4);
        let n = choose_grid_n(0.02);
        assert!((4..=256).contains(&n));
    }

    fn segment(x0: f32, y0: f32, x1: f32, y1: f32, r: f32, tag: u32) -> Segment {
        Segment {
            p0: [x0, y0, 0.0],
            r0: r,
            p1: [x1, y1, 0.0],
            r1: r,
            tag,
        }
    }

    #[test]
    fn bin_segments_places_every_segment_findable_from_its_own_bounding_box() {
        let segs = vec![
            segment(0.1, 0.1, 0.12, 0.1, 0.02, 1),
            segment(0.5, 0.5, 0.52, 0.5, 0.02, 2),
            segment(0.99, 0.5, 1.01, 0.5, 0.02, 3), // straddles the x=0 tile seam
        ];
        let grid_n = choose_grid_n(0.02);
        let (offsets, indices) = bin_segments(&segs, grid_n);
        assert_eq!(offsets.len() as u32, grid_n * grid_n + 1);

        // Every segment must appear in at least one cell's list.
        for tag in [1u32, 2, 3] {
            let found = indices.iter().any(|&idx| segs[idx as usize].tag == tag);
            assert!(found, "segment tag {tag} missing from every bin");
        }

        // The seam-straddling segment must be reachable from a cell near
        // x=0 too (not just the cell near x=1), since a pixel near x=0
        // needs to find it via wrap-around.
        let cell_size = 1.0 / grid_n as f32;
        let near_zero_cell = (0.0f32 / cell_size).floor() as u32 % grid_n
            + (0.5f32 / cell_size).floor() as u32 % grid_n * grid_n;
        let start = offsets[near_zero_cell as usize] as usize;
        let end = offsets[near_zero_cell as usize + 1] as usize;
        let seam_segment_here = indices[start..end]
            .iter()
            .any(|&idx| segs[idx as usize].tag == 3);
        assert!(
            seam_segment_here,
            "seam-straddling segment should be binned near x=0 too (wrap-around)"
        );
    }

    #[test]
    fn dispatch_rejects_an_empty_segment_list() {
        let result = std::panic::catch_unwind(|| {
            let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
            let mut pool = mc_gpu::TexturePool::default();
            dispatch(
                &mut gpu,
                &mut pool,
                RasterOutputKeys {
                    height: "h",
                    mask: "m",
                    ply_id: "p",
                    cell: "c",
                },
                16,
                &[],
                1,
                0.1,
            );
        });
        assert!(result.is_err());
    }
}
