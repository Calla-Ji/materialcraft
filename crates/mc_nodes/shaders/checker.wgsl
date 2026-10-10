// checker.wgsl — Generators / Patterns / Checker
//
// Output:  `output` — an alternating black/white grid, `scale` cells across.
// Params:  scale — cells spanning the texture. Must be even for the
//          alternating pattern to tile seamlessly (an odd count flips parity
//          exactly at the wrap edge) — enforced on the Rust side.

struct CheckerParams {
    resolution: u32,
    scale: i32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: CheckerParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);
    let cell = vec2<i32>(floor(uv * f32(params.scale)));
    let parity = (cell.x + cell.y) % 2;
    let value = select(0.0, 1.0, parity == 0);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(vec3<f32>(value), 1.0));
}
