// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generates a full box-filtered mip chain for a channel texture already in
//! the pool, so the 3D viewport's mipmap-filtered sampler has real data to
//! blend between instead of only ever reading mip 0 (the root cause of the
//! "fine patterns alias into speckle on the 3D mesh" bug — see
//! PROGRESS.md). `mc_gpu::TexturePool::get_or_create` already allocates the
//! mip levels; this is what actually fills them in.

/// Regenerates every mip level beyond 0 for `key`'s texture, reading each
/// level from the one before it. Called unconditionally once per frame the
/// 3D viewport renders (same cadence `resolve_channel` already resolves at)
/// rather than tracked against node dirtiness — each level's dispatch is
/// tiny and the whole chain is a cheap, bounded amount of work (geometric
/// series, ~1.33x one full-resolution pass), so recomputing it even when
/// nothing changed isn't worth the bookkeeping it'd take to skip.
///
/// All levels run in one command encoder/submit (one GPU sync point for the
/// whole chain), but each level is still its own compute pass — wgpu only
/// guarantees a later pass sees an earlier pass's writes at a pass boundary,
/// not between two dispatches inside the same pass, and level L's read
/// depends on level L-1's write completing first.
pub fn generate_mip_chain(
    gpu: &mut mc_gpu::Context,
    pool: &mc_gpu::TexturePool,
    key: &str,
    is_normal_map: bool,
) {
    let Some(texture) = pool.texture(key) else {
        return;
    };
    let mip_count = texture.mip_level_count();
    if mip_count <= 1 {
        return;
    }
    let base_size = texture.width();

    // A storage texture binding's format is fixed at shader-compile time,
    // so the one template substitutes whichever format this channel's own
    // pool texture actually uses — see mip_downsample.wgsl's own doc
    // comment on `dest_tex`. A format with no real-value meaning here
    // (Rgba8Unorm: flat fallback colors, imported images — none of which
    // ever reach a multi-mip pool texture, so `mip_count <= 1` above
    // already returned) is a caller bug, not something a user can trigger
    // by wiring up a graph; log and skip rather than panic either way.
    let (format_token, node_kind) = match texture.format() {
        wgpu::TextureFormat::Rgba16Float => ("rgba16float", "mip_downsample_rgba16float"),
        wgpu::TextureFormat::Rgba32Float => ("rgba32float", "mip_downsample_rgba32float"),
        other => {
            log::warn!("mipgen: no mip_downsample variant for format {other:?} (key {key:?})");
            return;
        }
    };
    let shader_source =
        include_str!("../shaders/mip_downsample.wgsl").replace("__DEST_FORMAT__", format_token);
    let pipeline = gpu.compute_pipeline(node_kind, &shader_source, "main");

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mip_downsample_encoder"),
        });

    for level in 1..mip_count {
        let dest_size = (base_size >> level).max(1);

        let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("mip_downsample_src"),
            base_mip_level: level - 1,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let dest_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("mip_downsample_dest"),
            base_mip_level: level,
            mip_level_count: Some(1),
            ..Default::default()
        });

        let mut param_bytes = [0u8; 16];
        param_bytes[0..4].copy_from_slice(&dest_size.to_le_bytes());
        param_bytes[4..8].copy_from_slice(&(is_normal_map as u32).to_le_bytes());
        let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mip_downsample_params"),
            size: param_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mip_downsample_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&dest_view),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mip_downsample_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = dest_size.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }

    gpu.queue.submit(Some(encoder.finish()));
}

/// A view spanning `key`'s entire mip chain, for the 3D viewport's sampler
/// to actually blend between — `TexturePool::view`'s own cached view is
/// deliberately pinned to mip 0 only (every other caller needs exactly
/// that), so this is built fresh on demand instead.
pub fn full_mip_view(pool: &mc_gpu::TexturePool, key: &str) -> Option<wgpu::TextureView> {
    let texture = pool.texture(key)?;
    Some(texture.create_view(&wgpu::TextureViewDescriptor::default()))
}

#[cfg(test)]
mod tests {
    use super::generate_mip_chain;

    /// Regression test for two crashes found by manual testing (connecting
    /// a node's output straight into a PBR channel and looking at the 3D
    /// viewport): first with any ordinary node (format `Rgba16Float`,
    /// every node's own output texture since the earlier banding fix —
    /// first noticed with Gabor Noise, but not specific to it), then again
    /// with Gabor Noise's `raw` output specifically (format `Rgba32Float`,
    /// its own deliberately higher-precision output). Both were the same
    /// root cause: `mip_downsample.wgsl`'s `dest_tex` binding was a single
    /// hardcoded format (`rgba8unorm`, then `rgba16float`) that didn't
    /// match whichever format the channel's own pool texture actually
    /// used. `generate_mip_chain` now substitutes the right format per
    /// call instead of assuming one.
    #[test]
    fn generates_mips_for_every_real_channel_texture_format() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();

        pool.get_or_create(&gpu.device, "rgba16", 64, wgpu::TextureFormat::Rgba16Float);
        generate_mip_chain(&mut gpu, &pool, "rgba16", false);
        generate_mip_chain(&mut gpu, &pool, "rgba16", true);

        pool.get_or_create(&gpu.device, "rgba32", 64, wgpu::TextureFormat::Rgba32Float);
        generate_mip_chain(&mut gpu, &pool, "rgba32", false);
        generate_mip_chain(&mut gpu, &pool, "rgba32", true);
    }
}
