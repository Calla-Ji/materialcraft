// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Spatial / Transform 2D.
//!
//! Display name: "Transform 2D". Tooltip: "Offset, rotate, and scale, with
//! wrap-around tiling."

use crate::shaders::full_source;

const NODE_KIND: &str = "transform2d";

pub struct Transform2DParams {
    pub offset_x: f32,
    pub offset_y: f32,
    pub rotation: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub flip_x: bool,
    pub flip_y: bool,
}

impl Default for Transform2DParams {
    fn default() -> Self {
        Self {
            offset_x: 0.0,
            offset_y: 0.0,
            rotation: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            flip_x: false,
            flip_y: false,
        }
    }
}

impl Transform2DParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let float = |name: &str, default: f32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(default)
        };
        let boolean = |name: &str, default: bool| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_bool)
                .unwrap_or(default)
        };
        Self {
            offset_x: float("offset_x", defaults.offset_x),
            offset_y: float("offset_y", defaults.offset_y),
            rotation: float("rotation", defaults.rotation),
            scale_x: float("scale_x", defaults.scale_x),
            scale_y: float("scale_y", defaults.scale_y),
            flip_x: boolean("flip_x", defaults.flip_x),
            flip_y: boolean("flip_y", defaults.flip_y),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &Transform2DParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("transform2d: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("transform2d.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("transform2d_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let output_view = pool.get_or_create(
        &gpu.device,
        output_key,
        resolution,
        wgpu::TextureFormat::Rgba8Unorm,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("transform2d_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&input_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("transform2d_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("transform2d_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

fn pack_params(resolution: u32, params: &Transform2DParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.offset_x.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.offset_y.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.rotation.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.scale_x.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.scale_y.to_le_bytes());
    bytes[24..28].copy_from_slice(&(params.flip_x as u32).to_le_bytes());
    bytes[28..32].copy_from_slice(&(params.flip_y as u32).to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, Transform2DParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &Transform2DParams) -> Vec<u8> {
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
        let texture = pool.texture("output").expect("transform2d texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    /// The identity transform (default params) is the one case guaranteed to
    /// stay exactly tileable — see the shader's doc comment on rotation.
    #[test]
    fn identity_transform_preserves_tiling() {
        assert_tiles_seamlessly(&render(&Transform2DParams::default()), RESOLUTION);
    }

    #[test]
    fn identity_transform_is_a_no_op() {
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
            &Transform2DParams::default(),
        );
        let after = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        assert_eq!(before, after, "identity transform must not change pixels");
    }

    #[test]
    fn is_deterministic() {
        let params = Transform2DParams {
            offset_x: 0.1,
            rotation: 0.1,
            ..Transform2DParams::default()
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same input + params produced different pixels"
        );
    }

    /// A 90° rotation (a multiple of a quarter turn) is one of the rotation
    /// angles that *does* stay exactly tileable.
    #[test]
    fn quarter_turn_rotation_preserves_tiling() {
        let params = Transform2DParams {
            rotation: 0.25,
            ..Transform2DParams::default()
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    #[test]
    fn flip_x_mirrors_left_to_right() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams::default(),
        );
        let input = gpu.read_rgba8(pool.texture("input").unwrap(), RESOLUTION);

        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output",
            RESOLUTION,
            &Transform2DParams {
                flip_x: true,
                ..Transform2DParams::default()
            },
        );
        let flipped = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        for y in [0, RESOLUTION / 3, RESOLUTION - 1] {
            for x in [0, RESOLUTION / 4, RESOLUTION - 1] {
                let mirrored_x = RESOLUTION - 1 - x;
                let got = ((y * RESOLUTION + x) * 4) as usize;
                let want = ((y * RESOLUTION + mirrored_x) * 4) as usize;
                assert_eq!(
                    flipped[got..got + 4],
                    input[want..want + 4],
                    "flip_x should mirror column {x} from column {mirrored_x}"
                );
            }
        }
    }

    #[test]
    fn flip_y_mirrors_top_to_bottom() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams::default(),
        );
        let input = gpu.read_rgba8(pool.texture("input").unwrap(), RESOLUTION);

        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output",
            RESOLUTION,
            &Transform2DParams {
                flip_y: true,
                ..Transform2DParams::default()
            },
        );
        let flipped = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        for y in [0, RESOLUTION / 4, RESOLUTION - 1] {
            for x in [0, RESOLUTION / 3, RESOLUTION - 1] {
                let mirrored_y = RESOLUTION - 1 - y;
                let got = ((y * RESOLUTION + x) * 4) as usize;
                let want = ((mirrored_y * RESOLUTION + x) * 4) as usize;
                assert_eq!(
                    flipped[got..got + 4],
                    input[want..want + 4],
                    "flip_y should mirror row {y} from row {mirrored_y}"
                );
            }
        }
    }

    #[test]
    fn flip_preserves_tiling() {
        let params = Transform2DParams {
            flip_x: true,
            flip_y: true,
            ..Transform2DParams::default()
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }
}
