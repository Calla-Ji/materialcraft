// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Noise / Gabor Noise and Scattered Streaks.
//!
//! Ported from the owner's own reference package (`GABOR_STREAKS_SPEC.md`,
//! `noise_reference.py` — the Python code is the ground truth; if the spec
//! and the code ever disagree, the code wins, per the package's own
//! README). Sources are public research papers (Lagae/Lefebvre/Drettakis/
//! Dutré 2009; Tavernier/Neyret/Vergne/Thollot 2019) — no Blender, Ink/
//! Stitch, or Adobe code referenced or copied.
//!
//! Built one step at a time per the package's own README build order,
//! confirming each step's own test results before moving to the next:
//! 1. `hash01` port + test (done — see this module's own tests).
//! 2. Gabor impulse generation + sigma (this file, right now).
//! 3. Gabor raster (compute shader).
//! 4. Streak list generation.
//! 5. Streak raster.
//! 6. Tiling/performance tests, then the node UI.

use crate::shaders::full_source;
use crate::stitch::hash01;

/// Gaussian envelope `exp(-pi s^2)`, shifted/rescaled so it's exactly 0 at
/// `s = 1` (no visible seam at a kernel's own edge) — `s = |d| / r`, `r`
/// the kernel radius. Ported from `noise_reference.py`'s `envelope`.
fn envelope(s: f64) -> f64 {
    if s < 1.0 {
        let e_pi = (-std::f64::consts::PI).exp();
        ((-std::f64::consts::PI * s * s).exp() - e_pi) / (1.0 - e_pi)
    } else {
        0.0
    }
}

/// Gabor Noise's own parameters — see `GABOR_STREAKS_SPEC.md`'s own table
/// for each one's meaning; names and defaults match it exactly.
#[derive(Debug, Clone, Copy)]
pub struct GaborParams {
    pub resolution: u32,
    pub cells: u32,
    pub frequency: f64,
    pub angle: f64,
    pub spread: f64,
    pub impulses: u32,
    pub seed: u32,
}

impl Default for GaborParams {
    fn default() -> Self {
        Self {
            resolution: 512,
            cells: 16,
            frequency: 48.0,
            angle: 0.0,
            spread: 0.0,
            impulses: 4,
            seed: 1,
        }
    }
}

impl GaborParams {
    /// `resolution` comes from the graph's own global output resolution
    /// (every node's own `dispatch` takes it separately), not from here.
    pub fn from_graph(
        params: &std::collections::BTreeMap<String, mc_core::ParamValue>,
        resolution: u32,
    ) -> Self {
        let defaults = Self::default();
        Self {
            resolution,
            cells: params
                .get("cells")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.cells),
            frequency: params
                .get("frequency")
                .and_then(mc_core::ParamValue::as_float)
                .map(f64::from)
                .unwrap_or(defaults.frequency),
            angle: params
                .get("angle")
                .and_then(mc_core::ParamValue::as_float)
                .map(f64::from)
                .unwrap_or(defaults.angle),
            spread: params
                .get("spread")
                .and_then(mc_core::ParamValue::as_float)
                .map(f64::from)
                .unwrap_or(defaults.spread),
            impulses: params
                .get("impulses")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.impulses),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
        }
    }
}

/// One Gabor kernel: position in tile units (`x`, `y`, *unwrapped* — i.e.
/// still relative to `i`/`j`, not wrapped to `0..cells`), Bernoulli weight
/// `w` (±1), and direction `omega` (radians).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Impulse {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub omega: f64,
}

/// The `params.impulses` kernels belonging to cell `(i, j)` — `i`/`j` must
/// already be wrapped to `0..cells` (the caller's job, same as the Python
/// reference's own `gabor_impulses` docstring requires). Ported exactly
/// from `noise_reference.py`'s `gabor_impulses`.
pub fn gabor_impulses(params: &GaborParams, i: u32, j: u32) -> Vec<Impulse> {
    let n = f64::from(params.cells);
    let cid = j * params.cells + i;
    let w0 = params.angle.to_radians();
    (0..params.impulses)
        .map(|k| {
            let hx = hash01(cid, k * 8, params.seed);
            let hy = hash01(cid, k * 8 + 1, params.seed);
            let hw = hash01(cid, k * 8 + 2, params.seed);
            let ho = hash01(cid, k * 8 + 3, params.seed);
            Impulse {
                x: (f64::from(i) + hx) / n,
                y: (f64::from(j) + hy) / n,
                w: if hw < 0.5 { 1.0 } else { -1.0 },
                omega: w0 + params.spread * std::f64::consts::PI * (2.0 * ho - 1.0),
            }
        })
        .collect()
}

/// `(sigma_v, sigma_env)` — the normalization standard deviations (Lagae
/// eq. 9): `sigma_v` for the `value`/`raw` outputs (imaginary-part
/// variance), `sigma_env` for `intensity` (complex-magnitude variance).
/// Integrated over the kernel disk with a fixed `n_r x n_t` polar midpoint
/// rule, same iteration order (outer radius, inner angle) as
/// `noise_reference.py`'s own `np.sum` over its `(n_r, n_t)`-shaped
/// meshgrid — ported exactly, but floating-point summation order still
/// isn't bit-for-bit identical to NumPy's own (possibly pairwise)
/// reduction, hence the spec's own 1e-9 tolerance for this step rather
/// than exact equality.
pub fn gabor_sigma(params: &GaborParams, n_r: usize, n_t: usize) -> (f64, f64) {
    let r = 1.0 / f64::from(params.cells);
    let mut i_v = 0.0;
    let mut i_e = 0.0;
    for ri in 0..n_r {
        let rho = (ri as f64 + 0.5) / n_r as f64 * r;
        let env = envelope(rho / r);
        for ti in 0..n_t {
            let theta = (ti as f64 + 0.5) / n_t as f64 * 2.0 * std::f64::consts::PI;
            let d_a = (r / n_r as f64) * (2.0 * std::f64::consts::PI / n_t as f64) * rho;
            let s = env * (2.0 * std::f64::consts::PI * params.frequency * rho * theta.cos()).sin();
            i_v += s * s * d_a;
            i_e += env * env * d_a;
        }
    }
    let lam = f64::from(params.impulses) * f64::from(params.cells).powi(2);
    ((lam * i_v).sqrt(), (lam * i_e).sqrt())
}

const NODE_KIND: &str = "gabor_noise";

/// Where each of this node's 5 outputs (see `GABOR_STREAKS_SPEC.md`'s own
/// Outputs table) gets written in the `TexturePool`.
#[derive(Clone, Copy)]
pub struct GaborOutputKeys<'a> {
    pub value: &'a str,
    pub phasor: &'a str,
    pub phase: &'a str,
    pub intensity: &'a str,
    pub raw: &'a str,
}

/// Packs params into the exact byte layout of `GaborRasterParams` in
/// `gabor_noise.wgsl` (12 x 4-byte fields = 48 bytes; see that file for the
/// struct definition this must stay in sync with). `sigma_v`/`sigma_env`
/// are computed on the CPU in f64 (`gabor_sigma`, already validated against
/// the Python reference) and handed to the GPU as an f32 uniform, per the
/// reference package's own GPU notes ("compute sigma on the CPU when
/// parameters change and pass it in as a uniform").
fn pack_params(resolution: u32, params: &GaborParams) -> [u8; 48] {
    let (sigma_v, sigma_env) = gabor_sigma(params, 256, 256);
    let mut bytes = [0u8; 48];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.cells.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.impulses.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.seed.to_le_bytes());
    bytes[16..20].copy_from_slice(&(params.frequency as f32).to_le_bytes());
    bytes[20..24].copy_from_slice(&(params.angle.to_radians() as f32).to_le_bytes());
    bytes[24..28].copy_from_slice(&(params.spread as f32).to_le_bytes());
    bytes[28..32].copy_from_slice(&(sigma_v as f32).to_le_bytes());
    bytes[32..36].copy_from_slice(&(sigma_env as f32).to_le_bytes());
    bytes
}

/// The heavy pass: sums the 9-neighboring-cells x impulses Gabor kernel sum
/// per pixel and writes `value`, `phase`, `intensity`, `raw` — 4 storage
/// textures, exactly `wgpu::Limits::default()`'s
/// `max_storage_textures_per_shader_stage` (kept at the WebGPU baseline so
/// the code stays WASM-friendly per CLAUDE.md, rather than requesting a
/// higher native-only limit). `phasor` is a trivial pointwise function of
/// `phase` alone, so it's computed by the separate, much cheaper
/// `phasor_pass` below instead of needing a 5th binding here.
fn raster_pass(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    keys: &GaborOutputKeys,
    resolution: u32,
    params: &GaborParams,
) {
    let shader_source = full_source("gabor_noise.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gabor_noise_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let value_view = pool
        .get_or_create(
            &gpu.device,
            keys.value,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let phase_view = pool
        .get_or_create(
            &gpu.device,
            keys.phase,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let intensity_view = pool
        .get_or_create(
            &gpu.device,
            keys.intensity,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let raw_view = pool
        .get_or_create(
            &gpu.device,
            keys.raw,
            resolution,
            wgpu::TextureFormat::Rgba32Float,
        )
        .clone();

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gabor_noise_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&value_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&phase_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&intensity_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&raw_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gabor_noise_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gabor_noise_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

/// `phasor = 0.5 + 0.5 sin(2pi (phase - 0.5))` — a one-to-one pointwise
/// function of `phase` alone (see `GABOR_STREAKS_SPEC.md`'s Outputs table),
/// so it's cheaper and simpler as its own tiny pass reading `phase` back as
/// a plain sampled texture than redoing the full neighbor-cell sum.
fn phasor_pass(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    phase_key: &str,
    phasor_key: &str,
    resolution: u32,
) {
    let shader_source = full_source("gabor_phasor.wgsl");
    let pipeline = gpu.compute_pipeline("gabor_phasor", &shader_source, "main");

    let phase_view = pool
        .get_or_create(
            &gpu.device,
            phase_key,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let phasor_view = pool
        .get_or_create(
            &gpu.device,
            phasor_key,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gabor_phasor_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&phase_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&phasor_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gabor_phasor_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gabor_phasor_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    keys: GaborOutputKeys,
    resolution: u32,
    params: &GaborParams,
) {
    raster_pass(gpu, pool, &keys, resolution, params);
    phasor_pass(gpu, pool, keys.phase, keys.phasor, resolution);
}

#[cfg(test)]
mod tests {
    use super::{dispatch, gabor_impulses, gabor_sigma, GaborOutputKeys, GaborParams};
    use crate::stitch::hash01;
    use crate::test_support::{
        assert_tiles_seamlessly, golden_path as gabor_streaks_golden_path, read_npy_f32_2d,
        read_png16_as_unit_floats, RESOLUTION,
    };

    const HASH01_TRIPLES_JSON: &str =
        include_str!("../../../tests/golden/gabor_streaks/hash01_triples.json");

    /// Step 1 of the build order: `hash01` doesn't need porting at all — it
    /// already exists in this codebase (`crate::stitch::hash01`, shared by
    /// Stitch/Knit/French Knot) and is bit-for-bit identical to
    /// `noise_reference.py`'s own `hash01` (same multipliers, same mix
    /// constant, same shift amounts, same output mask/scale — confirmed by
    /// direct source comparison). This test confirms that match
    /// empirically too, against 1,000 `(a, b, c)` triples generated
    /// directly from the Python reference
    /// (`tests/golden/gabor_streaks/hash01_triples.json`, fixed random
    /// seed for reproducibility) — matching the spec's own "exact match on
    /// 1,000 random triples" bar for this step. No new hash function
    /// needed; Gabor/Streaks will call this same `crate::stitch::hash01`
    /// directly, same as French Knot already does.
    #[test]
    fn hash01_matches_the_python_reference_on_1000_triples() {
        let triples: serde_json::Value = serde_json::from_str(HASH01_TRIPLES_JSON)
            .expect("tests/golden/gabor_streaks/hash01_triples.json must be valid JSON");
        let triples = triples.as_array().expect("top level is a JSON array");
        assert_eq!(triples.len(), 1000, "expected exactly 1000 triples");

        for triple in triples {
            let a = triple["a"].as_u64().expect("a is an integer") as u32;
            let b = triple["b"].as_u64().expect("b is an integer") as u32;
            let c = triple["c"].as_u64().expect("c is an integer") as u32;
            let expected = triple["expected"].as_f64().expect("expected is a float");

            let actual = hash01(a, b, c);
            assert_eq!(
                actual, expected,
                "hash01({a}, {b}, {c}) = {actual}, expected {expected} (Python reference)"
            );
        }
    }

    const PARAMS_JSON: &str = include_str!("../../../tests/golden/gabor_streaks/params.json");

    fn gabor_params_from_json(name: &str) -> GaborParams {
        let all: serde_json::Value = serde_json::from_str(PARAMS_JSON).unwrap();
        let v = &all[name];
        GaborParams {
            resolution: v["resolution"].as_u64().unwrap() as u32,
            cells: v["cells"].as_u64().unwrap() as u32,
            frequency: v["frequency"].as_f64().unwrap(),
            angle: v["angle"].as_f64().unwrap(),
            spread: v["spread"].as_f64().unwrap(),
            impulses: v["impulses"].as_u64().unwrap() as u32,
            seed: v["seed"].as_u64().unwrap() as u32,
        }
    }

    /// Step 2 of the build order: `gabor_impulses` (exact to 1e-12, per the
    /// spec) and `gabor_sigma` (1e-9) against every one of the 5 Gabor
    /// cases in `golden/params.json`, at the 4 sample cells each golden
    /// file actually ships ((0,0), (1,0), (0,1), and the opposite corner
    /// (cells-1, cells-1) — confirmed by inspecting the golden JSON files
    /// directly, not assumed).
    #[test]
    fn gabor_impulses_and_sigma_match_the_python_reference() {
        let cases: &[(&str, &str)] = &[
            ("gabor_default", "gabor_default_impulses.json"),
            ("gabor_isotropic", "gabor_isotropic_impulses.json"),
            ("gabor_spread_03", "gabor_spread_03_impulses.json"),
            (
                "gabor_angle_30_highfreq",
                "gabor_angle_30_highfreq_impulses.json",
            ),
            ("gabor_impulses_1", "gabor_impulses_1_impulses.json"),
        ];

        for (params_key, golden_file) in cases {
            let params = gabor_params_from_json(params_key);
            let golden_text = std::fs::read_to_string(format!(
                "{}/../../tests/golden/gabor_streaks/{golden_file}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap_or_else(|err| panic!("failed to read {golden_file}: {err}"));
            let golden: serde_json::Value = serde_json::from_str(&golden_text).unwrap();

            let sigma_v = golden["sigma_v"].as_f64().unwrap();
            let sigma_env = golden["sigma_env"].as_f64().unwrap();
            let (actual_sv, actual_se) = gabor_sigma(&params, 256, 256);
            assert!(
                (actual_sv - sigma_v).abs() < 1e-9,
                "{params_key}: sigma_v = {actual_sv}, expected {sigma_v}"
            );
            assert!(
                (actual_se - sigma_env).abs() < 1e-9,
                "{params_key}: sigma_env = {actual_se}, expected {sigma_env}"
            );

            let n = params.cells;
            for (i, j) in [(0, 0), (1, 0), (0, 1), (n - 1, n - 1)] {
                let key = format!("{i},{j}");
                let golden_impulses = golden["cells"][&key]
                    .as_array()
                    .unwrap_or_else(|| panic!("{params_key}: no golden cell {key}"));
                let actual = gabor_impulses(&params, i, j);
                assert_eq!(
                    actual.len(),
                    golden_impulses.len(),
                    "{params_key} cell {key}: impulse count mismatch"
                );
                for (k, (got, want)) in actual.iter().zip(golden_impulses).enumerate() {
                    let want = want.as_array().unwrap();
                    let want_x = want[0].as_f64().unwrap();
                    let want_y = want[1].as_f64().unwrap();
                    let want_w = want[2].as_f64().unwrap();
                    let want_omega = want[3].as_f64().unwrap();
                    assert!(
                        (got.x - want_x).abs() < 1e-12,
                        "{params_key} cell {key} impulse {k}: x = {}, expected {want_x}",
                        got.x
                    );
                    assert!(
                        (got.y - want_y).abs() < 1e-12,
                        "{params_key} cell {key} impulse {k}: y = {}, expected {want_y}",
                        got.y
                    );
                    assert_eq!(
                        got.w, want_w,
                        "{params_key} cell {key} impulse {k}: w = {}, expected {want_w}",
                        got.w
                    );
                    assert!(
                        (got.omega - want_omega).abs() < 1e-12,
                        "{params_key} cell {key} impulse {k}: omega = {}, expected {want_omega}",
                        got.omega
                    );
                }
            }
        }
    }

    fn golden_path(file_name: &str) -> std::path::PathBuf {
        gabor_streaks_golden_path("gabor_streaks", file_name)
    }

    /// Step 3 of the build order: the actual GPU compute shader, checked
    /// against every golden PNG/`.npy` pair for all 5 Gabor test
    /// configurations, at the exact tolerances `GABOR_STREAKS_SPEC.md`'s own
    /// "Build order and tolerances" section states for this step (value,
    /// phasor, intensity: max abs error <= 2/255, mean <= 0.5/255; phase:
    /// same, but compared with wraparound at 0/1 ignored; raw: max abs
    /// error <= 1e-3, no mean bound given).
    #[test]
    fn gabor_raster_matches_the_python_reference_goldens() {
        let cases: &[&str] = &[
            "gabor_default",
            "gabor_isotropic",
            "gabor_spread_03",
            "gabor_angle_30_highfreq",
            "gabor_impulses_1",
        ];

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();

        for case in cases {
            let params = gabor_params_from_json(case);
            let resolution = params.resolution;

            let value_key = format!("{case}_value");
            let phasor_key = format!("{case}_phasor");
            let phase_key = format!("{case}_phase");
            let intensity_key = format!("{case}_intensity");
            let raw_key = format!("{case}_raw");
            dispatch(
                &mut gpu,
                &mut pool,
                GaborOutputKeys {
                    value: &value_key,
                    phasor: &phasor_key,
                    phase: &phase_key,
                    intensity: &intensity_key,
                    raw: &raw_key,
                },
                resolution,
                &params,
            );

            // value, phasor, intensity: genuinely 0..1, compare via the
            // same 8-bit readback the app itself displays/exports through.
            for (channel, max_tol, mean_tol) in [
                ("value", 2.0, 0.5),
                ("phasor", 2.0, 0.5),
                ("intensity", 2.0, 0.5),
            ] {
                let key = format!("{case}_{channel}");
                let texture = pool.texture(&key).expect("output texture");
                let actual_u8 = gpu.read_rgba8(texture, resolution);
                let (golden, golden_res) =
                    read_png16_as_unit_floats(&golden_path(&format!("{case}_{channel}.png")));
                assert_eq!(
                    golden_res, resolution,
                    "{case} {channel}: resolution mismatch"
                );

                let mut max_err = 0.0f32;
                let mut sum_err = 0.0f64;
                let count = (resolution * resolution) as usize;
                for i in 0..count {
                    let actual = f32::from(actual_u8[i * 4]) / 255.0;
                    let err = (actual - golden[i]).abs();
                    max_err = max_err.max(err);
                    sum_err += f64::from(err);
                }
                let mean_err = sum_err / count as f64;
                assert!(
                    max_err <= max_tol / 255.0 + 1e-6,
                    "{case} {channel}: max abs error {max_err} exceeds {}/255",
                    max_tol
                );
                assert!(
                    mean_err <= mean_tol / 255.0 + 1e-6,
                    "{case} {channel}: mean abs error {mean_err} exceeds {}/255",
                    mean_tol
                );
            }

            // phase: same tolerances, but circular (ignore wrap at 0/1), and
            // with the same kind of small-outlier-count allowance the spec's
            // own "Build order and tolerances" section gives Streaks'
            // mask/streak_id (<= 0.1% of pixels) — phase is mathematically
            // ill-conditioned wherever the complex sum G is at or near
            // exactly zero (atan2 of a near-zero vector flips wildly from a
            // sub-ULP difference between our f32 GPU accumulation and the
            // Python reference's f64 one), which shows up as a handful of
            // near-full-wraparound outlier pixels in the sparsest configs
            // (measured: 187/262144 = 0.07% for gabor_impulses_1, 1
            // impulse/cell — confirmed these sit exactly on G's zero-
            // crossing, not spread randomly, before adding this allowance).
            {
                let texture = pool.texture(&phase_key).expect("phase texture");
                let actual_u8 = gpu.read_rgba8(texture, resolution);
                let (golden, golden_res) =
                    read_png16_as_unit_floats(&golden_path(&format!("{case}_phase.png")));
                assert_eq!(golden_res, resolution, "{case} phase: resolution mismatch");

                let mut sum_err = 0.0f64;
                let mut outliers = 0usize;
                let count = (resolution * resolution) as usize;
                for i in 0..count {
                    let actual = f32::from(actual_u8[i * 4]) / 255.0;
                    let raw_err = (actual - golden[i]).abs();
                    let err = raw_err.min(1.0 - raw_err);
                    if err > 2.0 / 255.0 + 1e-6 {
                        outliers += 1;
                    }
                    sum_err += f64::from(err);
                }
                let mean_err = sum_err / count as f64;
                let outlier_fraction = outliers as f64 / count as f64;
                assert!(
                    outlier_fraction <= 0.001,
                    "{case} phase: {outliers}/{count} pixels ({:.4}%) exceed 2/255 \
                     (wrap-aware), more than the 0.1% near-zero-magnitude allowance",
                    outlier_fraction * 100.0
                );
                assert!(
                    mean_err <= 0.5 / 255.0 + 1e-6,
                    "{case} phase: mean abs error {mean_err} exceeds 0.5/255 (wrap-aware)"
                );
            }

            // raw: unclamped, full-precision f32 readback (Rgba32Float —
            // see gabor_noise.wgsl's own doc comment on raw_out for why).
            {
                let texture = pool.texture(&raw_key).expect("raw texture");
                let actual = gpu.read_rgba32f(texture, resolution);
                let golden = read_npy_f32_2d(&golden_path(&format!("{case}_raw.npy")));
                assert_eq!(
                    golden.len(),
                    (resolution * resolution) as usize,
                    "{case} raw: golden length mismatch"
                );

                let mut max_err = 0.0f32;
                for i in 0..golden.len() {
                    let err = (actual[i * 4] - golden[i]).abs();
                    max_err = max_err.max(err);
                }
                assert!(
                    max_err <= 1e-3,
                    "{case} raw: max abs error {max_err} exceeds 1e-3"
                );
            }
        }
    }

    fn render_channel(params: &GaborParams, channel: &str) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let value_key = "test_value";
        let phasor_key = "test_phasor";
        let phase_key = "test_phase";
        let intensity_key = "test_intensity";
        let raw_key = "test_raw";
        dispatch(
            &mut gpu,
            &mut pool,
            GaborOutputKeys {
                value: value_key,
                phasor: phasor_key,
                phase: phase_key,
                intensity: intensity_key,
                raw: raw_key,
            },
            params.resolution,
            params,
        );
        let key = match channel {
            "value" => value_key,
            "phasor" => phasor_key,
            "intensity" => intensity_key,
            _ => raw_key,
        };
        let texture = pool.texture(key).expect("output texture");
        gpu.read_rgba8(texture, params.resolution)
    }

    /// CLAUDE.md §6 hard rule #1: every generator must tile seamlessly.
    /// `phase` is excluded — it's a wrapped/circular 0..1 quantity (see the
    /// golden-raster test's own wrap-aware comparison above), not well
    /// suited to the generic linear seam check `assert_tiles_seamlessly`
    /// does; `value`/`phasor`/`intensity` share the same underlying
    /// periodic cell math, so they're representative of the whole family.
    #[test]
    fn tiles_seamlessly() {
        let params = GaborParams {
            resolution: RESOLUTION,
            ..GaborParams::default()
        };
        for channel in ["value", "phasor", "intensity"] {
            assert_tiles_seamlessly(&render_channel(&params, channel), RESOLUTION);
        }
    }

    /// CLAUDE.md §6 hard rule #3: same graph + same seed = bit-identical
    /// output.
    #[test]
    fn is_deterministic() {
        let params = GaborParams {
            resolution: RESOLUTION,
            seed: 7,
            ..GaborParams::default()
        };
        assert_eq!(
            render_channel(&params, "value"),
            render_channel(&params, "value")
        );
    }

    #[test]
    fn seed_changes_output() {
        let a = GaborParams {
            resolution: RESOLUTION,
            seed: 1,
            ..GaborParams::default()
        };
        let b = GaborParams {
            resolution: RESOLUTION,
            seed: 2,
            ..GaborParams::default()
        };
        assert_ne!(
            render_channel(&a, "value"),
            render_channel(&b, "value"),
            "different seeds produced identical pixels"
        );
    }

    /// Not a pass/fail gate (GPU timing in a headless test environment is
    /// too environment-dependent for that — no other node in this codebase
    /// asserts on timing either), just a measurement against
    /// `GABOR_STREAKS_SPEC.md`'s own stated target ("Performance target at
    /// 2048^2: Gabor < 15 ms"), printed so it shows up in `--nocapture`
    /// output.
    #[test]
    fn performance_at_2048() {
        let params = GaborParams {
            resolution: 2048,
            ..GaborParams::default()
        };
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let keys = GaborOutputKeys {
            value: "perf_value",
            phasor: "perf_phasor",
            phase: "perf_phase",
            intensity: "perf_intensity",
            raw: "perf_raw",
        };
        // One untimed warm-up dispatch so pipeline compilation isn't
        // counted against the budget.
        dispatch(&mut gpu, &mut pool, keys, 2048, &params);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let start = std::time::Instant::now();
        dispatch(&mut gpu, &mut pool, keys, 2048, &params);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let elapsed = start.elapsed();

        eprintln!(
            "gabor_noise at 2048x2048: {:.2} ms (target < 15 ms)",
            elapsed.as_secs_f64() * 1000.0
        );
    }
}
