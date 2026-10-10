//! Generators / Patterns / Shape.
//!
//! Display name: "Shape". Tooltip: "A single disc, square, polygon, or star,
//! centered in the tile."

use crate::shaders::full_source;

const NODE_KIND: &str = "shape";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeKind {
    Disc,
    Square,
    Polygon,
    Star,
}

impl ShapeKind {
    pub const NAMES: [&'static str; 4] = ["Disc", "Square", "Polygon", "Star"];

    fn as_u32(self) -> u32 {
        match self {
            ShapeKind::Disc => 0,
            ShapeKind::Square => 1,
            ShapeKind::Polygon => 2,
            ShapeKind::Star => 3,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Disc" => Some(ShapeKind::Disc),
            "Square" => Some(ShapeKind::Square),
            "Polygon" => Some(ShapeKind::Polygon),
            "Star" => Some(ShapeKind::Star),
            _ => None,
        }
    }
}

pub struct ShapeParams {
    pub kind: ShapeKind,
    pub sides: u32,
    pub size: f32,
    pub softness: f32,
    pub rotation: f32,
}

impl Default for ShapeParams {
    fn default() -> Self {
        Self {
            kind: ShapeKind::Disc,
            sides: 5,
            size: 0.6,
            softness: 0.05,
            rotation: 0.0,
        }
    }
}

impl ShapeParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            kind: params
                .get("kind")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => ShapeKind::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.kind),
            sides: params
                .get("sides")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(3) as u32)
                .unwrap_or(defaults.sides),
            size: params
                .get("size")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.size),
            softness: params
                .get("softness")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.softness),
            rotation: params
                .get("rotation")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.rotation),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &ShapeParams,
) {
    let shader_source = full_source("shape.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shape_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let view = pool.get_or_create(
        &gpu.device,
        key,
        resolution,
        wgpu::TextureFormat::Rgba8Unorm,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shape_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shape_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shape_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

fn pack_params(resolution: u32, params: &ShapeParams) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.kind.as_u32().to_le_bytes());
    bytes[8..12].copy_from_slice(&params.sides.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.size.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.softness.to_le_bytes());
    bytes[24..28].copy_from_slice(&params.rotation.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, ShapeKind, ShapeParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &ShapeParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("shape texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    /// A shape sits fully inside the tile on a flat background, so it
    /// trivially tiles — background meets background at every edge.
    #[test]
    fn tiles_seamlessly_for_every_kind() {
        for kind in [
            ShapeKind::Disc,
            ShapeKind::Square,
            ShapeKind::Polygon,
            ShapeKind::Star,
        ] {
            let params = ShapeParams {
                kind,
                ..ShapeParams::default()
            };
            assert_tiles_seamlessly(&render(&params), RESOLUTION);
        }
    }

    #[test]
    fn center_is_inside_every_kind() {
        let r = RESOLUTION as usize;
        for kind in [
            ShapeKind::Disc,
            ShapeKind::Square,
            ShapeKind::Polygon,
            ShapeKind::Star,
        ] {
            let params = ShapeParams {
                kind,
                ..ShapeParams::default()
            };
            let pixels = render(&params);
            let center = pixels[(r / 2 * r + r / 2) * 4];
            assert!(
                center > 200,
                "{kind:?}: center should be inside the shape, got {center}"
            );
        }
    }

    #[test]
    fn corner_is_outside_every_kind() {
        let pixels_for = |kind| {
            let params = ShapeParams {
                kind,
                ..ShapeParams::default()
            };
            render(&params)
        };
        for kind in [
            ShapeKind::Disc,
            ShapeKind::Square,
            ShapeKind::Polygon,
            ShapeKind::Star,
        ] {
            let pixels = pixels_for(kind);
            let corner = pixels[0];
            assert!(
                corner < 50,
                "{kind:?}: corner should be outside the shape, got {corner}"
            );
        }
    }

    #[test]
    fn is_deterministic() {
        let params = ShapeParams {
            kind: ShapeKind::Star,
            rotation: 0.2,
            ..ShapeParams::default()
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same params produced different pixels"
        );
    }

    #[test]
    fn kind_name_round_trips() {
        for name in ShapeKind::NAMES {
            assert!(
                ShapeKind::from_name(name).is_some(),
                "{name:?} should parse"
            );
        }
    }
}
