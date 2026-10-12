// SPDX-License-Identifier: MIT OR Apache-2.0
//! Node metadata: display name, category, inputs/outputs, params with
//! defaults, and a tooltip for every node kind (CLAUDE.md §6.6). This is the
//! single source of truth the UI (M3) uses to populate the add-node search
//! menu, render a generic parameter panel, and block incompatible
//! connections — none of it is hardcoded per node in `mc_app`.
//!
//! The GPU dispatch code in `perlin.rs`/`levels.rs`/`blend.rs` has its own
//! small `Default` impls for its params struct; those are a safety fallback
//! for a graph that doesn't specify a param; the defaults declared here are
//! what a freshly-added node starts with. The two are meant to agree, but
//! are deliberately independent — dispatch code doesn't depend on this
//! metadata layer.

use mc_core::{GradientStop, ParamValue, PortType};

/// One parameter a node declares, independent of its current value in any
/// particular graph.
pub struct ParamSpec {
    pub name: &'static str,
    pub display_name: &'static str,
    pub tooltip: &'static str,
    pub kind: ParamSpecKind,
}

/// A param's type and allowed range/options. New variants (Bool) get added
/// here once a node that needs them exists.
pub enum ParamSpecKind {
    Float {
        min: f32,
        max: f32,
        default: f32,
    },
    Int {
        min: i32,
        max: i32,
        default: i32,
    },
    Seed {
        default: u32,
    },
    Enum {
        options: &'static [&'static str],
        default: &'static str,
    },
    Color {
        default: [f32; 4],
    },
    Bool {
        default: bool,
    },
    /// Gradient Map's color ramp — at most 8 stops (`gradient_map.wgsl`'s
    /// fixed-size uniform array), interpolated in position order.
    Gradient {
        default: &'static [GradientStop],
    },
    /// A 2-component value in the same units on both axes (e.g. a UV-space
    /// offset) — `min`/`max` apply to both components identically, since
    /// every current use (Channel Split's per-channel offsets) is
    /// symmetric. First used by the Glitch family.
    Vec2 {
        min: f32,
        max: f32,
        default: [f32; 2],
    },
}

impl ParamSpec {
    fn default_value(&self) -> ParamValue {
        match self.kind {
            ParamSpecKind::Float { default, .. } => ParamValue::Float(default),
            ParamSpecKind::Int { default, .. } => ParamValue::Int(default),
            ParamSpecKind::Seed { default } => ParamValue::Seed(default),
            ParamSpecKind::Enum { default, .. } => ParamValue::Enum(default.to_string()),
            ParamSpecKind::Color { default } => ParamValue::Color(default),
            ParamSpecKind::Bool { default } => ParamValue::Bool(default),
            ParamSpecKind::Gradient { default } => ParamValue::Gradient(default.to_vec()),
            ParamSpecKind::Vec2 { default, .. } => ParamValue::Vec2(default),
        }
    }
}

/// The shared "Tiling Mode" param (see `crate::tiling::TilingMode`) for any
/// node whose neighbor-sampling can show a visible edge-duplication
/// artifact — one helper rather than repeating the same `ParamSpec` at
/// every call site, so the name/wording stays identical everywhere it
/// appears.
const fn tiling_mode_param() -> ParamSpec {
    ParamSpec {
        name: "tiling_mode",
        display_name: "Tiling",
        tooltip: "This node can read past a tile's edge (e.g. a blur or an offset), which \
                  normally wraps around to the opposite edge to stay seamless. Turn an axis \
                  off here if you'd rather that content just clip at the edge instead of \
                  reappearing on the other side.",
        kind: ParamSpecKind::Enum {
            options: &crate::tiling::TilingMode::NAMES,
            default: "Both (Seamless)",
        },
    }
}

/// One input or output port a node declares.
pub struct PortSpec {
    pub name: &'static str,
    pub display_name: &'static str,
    pub port_type: PortType,
}

/// A node kind's full declaration.
pub struct NodeDef {
    pub kind: &'static str,
    pub display_name: &'static str,
    pub category: &'static str,
    pub tooltip: &'static str,
    pub inputs: &'static [PortSpec],
    pub outputs: &'static [PortSpec],
    pub params: &'static [ParamSpec],
}

impl NodeDef {
    /// A fresh `mc_core::Node` of this kind, with every param set to its
    /// declared default — what dragging this node onto the canvas creates.
    pub fn instantiate(&self) -> mc_core::Node {
        let mut node = mc_core::Node::new(self.kind);
        for param in self.params {
            node.params
                .insert(param.name.to_string(), param.default_value());
        }
        node
    }

    pub fn output(&self, name: &str) -> Option<&PortSpec> {
        self.outputs.iter().find(|p| p.name == name)
    }

    pub fn input(&self, name: &str) -> Option<&PortSpec> {
        self.inputs.iter().find(|p| p.name == name)
    }
}

/// All known node kinds. Looked up by [`node_def`]; iterated by the add-node
/// search menu via [`all_node_defs`].
pub fn all_node_defs() -> &'static [NodeDef] {
    NODE_DEFS
}

pub fn node_def(kind: &str) -> Option<&'static NodeDef> {
    NODE_DEFS.iter().find(|def| def.kind == kind)
}

/// Whether the UI should allow connecting a `from`-typed output to a
/// `to`-typed input (CLAUDE.md §5): Gray broadcasts implicitly into Color,
/// but Color needs an explicit Grayscale Conversion node to become Gray —
/// which doesn't exist yet, so that direction is simply blocked for now.
pub fn ports_compatible(from: PortType, to: PortType) -> bool {
    !matches!((from, to), (PortType::Color, PortType::Gray))
}

/// Node kinds whose single "input"/"output" pair behaves identically
/// per-channel, regardless of whether that channel is a lone Gray value or
/// one of Color's three — so their declared-Color output port is really
/// just a default, not a fact about what they produce. Left hardcoded to
/// Color, a Gray source (e.g. a height map) couldn't pass through Levels
/// and still reach a Gray-only input (e.g. Height to Normal) without an
/// unnecessary Grayscale Conversion node in between.
const TYPE_PRESERVING_KINDS: &[&str] = &[
    "levels",
    "invert",
    "clamp",
    "threshold",
    "curve",
    "block_shift",
    "scanline_jitter",
    "ordered_dither",
    "data_corrupt",
    // Added in the same audit that caught Blend (see `effective_output_type`'s
    // "blend" special case): every one of these reads its *whole* input
    // texel (full vec4 — base/blur/offset/resample math, no luminance
    // reduction) and writes it straight back out, so a Gray-broadcast input
    // (R=G=B) comes out exactly as Gray-broadcast, same as the originally
    // fixed Levels/Invert/Clamp/Threshold/Curve group — verified by reading
    // each one's own .wgsl file, not assumed from its category.
    "histogram_scan",
    "blur",
    "warp",
    "transform2d",
    "directional_blur",
    "directional_warp",
    "slope_blur",
    "bevel",
];

/// The name of the port `effective_output_type` should follow upstream for a
/// given type-preserving node kind — most call it "input", but the Glitch
/// family's image input is named "image" to read naturally alongside their
/// optional "mask" input.
fn type_preserving_input_port(kind: &str) -> &'static str {
    match kind {
        "block_shift" | "scanline_jitter" | "ordered_dither" | "data_corrupt" => "image",
        _ => "input",
    }
}

/// The port type a node's output *actually* carries right now, given what
/// (if anything) is wired into its input(s) — for the nodes in
/// `TYPE_PRESERVING_KINDS` this follows the upstream chain back through any
/// number of other type-preserving nodes (Blend included) to the first real
/// source; for every other node it's just the statically declared type.
/// Used by the UI for pin coloring and connection-compatibility checks,
/// never by evaluation (the GPU side treats Gray and Color identically as a
/// broadcast RGBA texture either way).
pub fn effective_output_type(
    graph: &mc_core::Graph,
    node_id: mc_core::NodeId,
    node_kind: &str,
    declared: PortType,
) -> PortType {
    // A graph can't contain cycles (rejected at connect time), so this bound
    // only guards against a future bug reintroducing one — 64 hops is far
    // more than any real graph would chain through type-preserving nodes.
    effective_output_type_bounded(graph, node_id, node_kind, declared, 64)
}

fn effective_output_type_bounded(
    graph: &mc_core::Graph,
    node_id: mc_core::NodeId,
    node_kind: &str,
    declared: PortType,
    depth: u32,
) -> PortType {
    let Some(depth) = depth.checked_sub(1) else {
        return declared;
    };
    if node_kind == "blend" {
        // Color wins over Gray: blending in *any* real color content can
        // produce non-gray output, so Color is the only type-safe result
        // once either input is Color — the reverse of how a lone
        // type-preserving input works, where there's only one source to
        // defer to. Gray only if *both* inputs resolve to Gray.
        let base = resolve_upstream_type(graph, node_id, "base", depth);
        let blend = resolve_upstream_type(graph, node_id, "blend", depth);
        return match (base, blend) {
            (None, None) => declared,
            (Some(only), None) | (None, Some(only)) => only,
            (Some(a), Some(b)) => {
                if a == PortType::Color || b == PortType::Color {
                    PortType::Color
                } else {
                    PortType::Gray
                }
            }
        };
    }
    if !TYPE_PRESERVING_KINDS.contains(&node_kind) {
        return declared;
    }
    resolve_upstream_type(graph, node_id, type_preserving_input_port(node_kind), depth)
        .unwrap_or(declared)
}

/// What's wired into `node_id`'s `port_name` input *actually* outputs, given
/// its own effective type might in turn depend on what's wired into it —
/// recurses one node upstream via `effective_output_type_bounded` rather
/// than reading the upstream node's merely-declared type, so a chain through
/// any mix of type-preserving nodes (including Blend) resolves correctly
/// however deep it goes. `None` if `port_name` isn't connected to anything.
fn resolve_upstream_type(
    graph: &mc_core::Graph,
    node_id: mc_core::NodeId,
    port_name: &str,
    depth: u32,
) -> Option<PortType> {
    let upstream_id = graph.upstream(node_id, port_name)?;
    let upstream_node = graph.node(upstream_id)?;
    let upstream_def = node_def(&upstream_node.kind)?;
    let upstream_output = upstream_def.outputs.first()?;
    Some(effective_output_type_bounded(
        graph,
        upstream_id,
        &upstream_node.kind,
        upstream_output.port_type,
        depth,
    ))
}

/// Whether `param_name` currently means anything for a node of `node_kind`
/// given its other params right now — e.g. Shape's "Sides"/"Point Depth"/
/// "Curvature" only apply to some of its "Kind" values (a Disc has no sides,
/// Point Depth only unfolds a Star, Curvature only shapes a Leaf's tips).
/// Lives here rather than in `mc_app`'s generic parameter panel so a node's
/// own quirks stay in its own metadata, not scattered into per-node-kind UI
/// code. Defaults to always-relevant for every node/param this doesn't know
/// about.
pub fn param_is_relevant(
    node_kind: &str,
    param_name: &str,
    current_params: &std::collections::BTreeMap<String, ParamValue>,
) -> bool {
    if node_kind == "dither_pattern" {
        let pattern = current_params.get("pattern").and_then(|v| match v {
            ParamValue::Enum(name) => Some(name.as_str()),
            _ => None,
        });
        return match param_name {
            "seed" => pattern == Some("Blue Noise"),
            "rotation" => pattern != Some("Blue Noise"),
            "dot_size" => pattern == Some("Dot"),
            _ => true,
        };
    }
    if node_kind == "ordered_dither" || node_kind == "palette_dither" {
        let pattern = current_params.get("pattern").and_then(|v| match v {
            ParamValue::Enum(name) => Some(name.as_str()),
            _ => None,
        });
        return match param_name {
            "seed" => pattern == Some("Blue Noise"),
            _ => true,
        };
    }
    if node_kind != "shape" {
        return true;
    }
    let shape_kind = current_params
        .get("kind")
        .and_then(|v| match v {
            ParamValue::Enum(name) => crate::shape::ShapeKind::from_name(name),
            _ => None,
        })
        .unwrap_or(crate::shape::ShapeKind::Disc);
    match param_name {
        "sides" => matches!(
            shape_kind,
            crate::shape::ShapeKind::Polygon | crate::shape::ShapeKind::Star
        ),
        "point_depth" => matches!(shape_kind, crate::shape::ShapeKind::Star),
        "curvature_top" | "curvature_bottom" | "curvature_middle" | "leaf_width"
        | "middle_position" => {
            matches!(shape_kind, crate::shape::ShapeKind::Leaf)
        }
        _ => true,
    }
}

const fn output_node_def(kind: &'static str, display_name: &'static str) -> NodeDef {
    NodeDef {
        kind,
        display_name,
        category: "Output",
        tooltip: "Marks this input as the material's output for this channel.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[],
        params: &[],
    }
}

static NODE_DEFS: &[NodeDef] = &[
    NodeDef {
        kind: "perlin_noise",
        display_name: "Perlin Noise",
        category: "Generators/Noise",
        tooltip: "Classic gradient noise — smooth, organic variation.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "scale",
                display_name: "Scale",
                tooltip: "How many noise cells tile across the texture.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 4,
                },
            },
            ParamSpec {
                name: "octaves",
                display_name: "Octaves",
                tooltip: "How many layers of noise are summed.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 8,
                    default: 4,
                },
            },
            ParamSpec {
                name: "persistence",
                display_name: "Persistence",
                tooltip: "How much each octave's contribution shrinks.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the pattern without changing its character.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    NodeDef {
        kind: "levels",
        display_name: "Levels",
        category: "Filters/Adjustments",
        tooltip: "Remap an input/output range, with a gamma curve in between.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "in_low",
                display_name: "In Low",
                tooltip: "Input black point.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "in_high",
                display_name: "In High",
                tooltip: "Input white point.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "gamma",
                display_name: "Gamma",
                tooltip: "Midtone curve; 1.0 is linear.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 4.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "out_low",
                display_name: "Out Low",
                tooltip: "Output black point.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "out_high",
                display_name: "Out High",
                tooltip: "Output white point.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
        ],
    },
    NodeDef {
        kind: "blend",
        display_name: "Blend",
        category: "Filters/Blending & Masking",
        tooltip: "Combine two inputs with a blend mode, opacity, and optional mask.",
        inputs: &[
            PortSpec {
                name: "base",
                display_name: "Base",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "blend",
                display_name: "Blend",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "mode",
                display_name: "Mode",
                tooltip: "How base and blend combine.",
                kind: ParamSpecKind::Enum {
                    options: &crate::blend::BlendMode::NAMES,
                    default: "Copy",
                },
            },
            ParamSpec {
                name: "opacity",
                display_name: "Opacity",
                tooltip: "Overall strength of the blend.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
        ],
    },
    NodeDef {
        kind: "voronoi",
        display_name: "Voronoi",
        category: "Generators/Noise",
        tooltip: "Cellular noise — distances to a scattered set of points.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "scale",
                display_name: "Scale",
                tooltip: "How many cells tile across the texture.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 4,
                },
            },
            ParamSpec {
                name: "jitter",
                display_name: "Jitter",
                tooltip: "How far each cell's point can move from center.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the pattern without changing its character.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
            ParamSpec {
                name: "mode",
                display_name: "Mode",
                tooltip: "Which cellular distance to output.",
                kind: ParamSpecKind::Enum {
                    options: &crate::voronoi::VoronoiMode::NAMES,
                    default: "F1",
                },
            },
        ],
    },
    NodeDef {
        kind: "fractal_sum",
        display_name: "Fractal Sum",
        category: "Generators/Noise",
        tooltip: "Several octaves of a base noise, summed (fBm).",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "base",
                display_name: "Base Noise",
                tooltip: "Which noise type to layer.",
                kind: ParamSpecKind::Enum {
                    options: &crate::fractal_sum::FractalBase::NAMES,
                    default: "Perlin",
                },
            },
            ParamSpec {
                name: "scale",
                display_name: "Scale",
                tooltip: "Base noise cells tiling across the texture.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 4,
                },
            },
            ParamSpec {
                name: "octaves",
                display_name: "Octaves",
                tooltip: "How many layers are summed.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 8,
                    default: 4,
                },
            },
            ParamSpec {
                name: "lacunarity",
                display_name: "Lacunarity",
                tooltip: "Frequency multiplier per octave.",
                kind: ParamSpecKind::Float {
                    min: 1.0,
                    max: 4.0,
                    default: 2.0,
                },
            },
            ParamSpec {
                name: "gain",
                display_name: "Gain",
                tooltip: "Amplitude multiplier per octave.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the pattern without changing its character.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    NodeDef {
        kind: "tile_generator",
        display_name: "Pattern Generator",
        category: "Generators/Patterns",
        tooltip: "A brick or grid pattern — Offset 0.5 for a classic running-bond brick layout.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "rows",
                display_name: "Rows",
                tooltip: "Must be even for seamless vertical tiling of the brick stagger.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 6,
                },
            },
            ParamSpec {
                name: "columns",
                display_name: "Columns",
                tooltip: "Grid columns.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 4,
                },
            },
            ParamSpec {
                name: "offset",
                display_name: "Offset",
                tooltip: "Stagger alternate rows by this fraction of a column (0.5 = brick).",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "gap",
                display_name: "Gap",
                tooltip: "Mortar gap width.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.4,
                    default: 0.06,
                },
            },
            ParamSpec {
                // Display name/tooltip only, deliberately not "Bevel" — this
                // is a flat per-tile luminance fade (see tile_generator.wgsl:
                // `mix(0.5, 1.0, bevel_amount)`), not a height-shaping
                // operation, and keeping the old name invited confusion with
                // the real Bevel node (owner feedback: built two comparable
                // graphs and found they "looked nothing alike," which they
                // never were going to — same name, unrelated effects). The
                // `name` key stays "bevel" so existing .mcgraph files with
                // this param saved under that key keep loading correctly.
                name: "bevel",
                display_name: "Edge Darkening",
                tooltip: "How much each tile's edge darkens toward the gap — a flat shading \
                          fade, not a height change.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.4,
                    default: 0.08,
                },
            },
            ParamSpec {
                name: "size_jitter",
                display_name: "Size Jitter",
                tooltip: "Per-tile random shrink.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.08,
                },
            },
            ParamSpec {
                name: "rotation_jitter",
                display_name: "Rotation Jitter",
                tooltip: "Per-tile random tilt.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "luminance_jitter",
                display_name: "Luminance Jitter",
                tooltip: "Per-tile random brightness variation.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.15,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the per-tile randomness.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    NodeDef {
        kind: "shape",
        display_name: "Shape",
        category: "Generators/Patterns",
        tooltip: "A single disc, square, polygon, star, or leaf, centered in the tile.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "kind",
                display_name: "Kind",
                tooltip: "Which shape to draw.",
                kind: ParamSpecKind::Enum {
                    options: &crate::shape::ShapeKind::NAMES,
                    default: "Disc",
                },
            },
            ParamSpec {
                name: "sides",
                display_name: "Sides",
                tooltip: "Vertex/point count for Polygon and Star.",
                kind: ParamSpecKind::Int {
                    min: 3,
                    max: 12,
                    default: 5,
                },
            },
            ParamSpec {
                name: "size",
                display_name: "Size",
                tooltip: "How large the shape is.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 1.0,
                    default: 0.6,
                },
            },
            ParamSpec {
                name: "softness",
                display_name: "Softness",
                tooltip: "How soft the edge is.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.05,
                },
            },
            ParamSpec {
                name: "rotation",
                display_name: "Rotation",
                tooltip: "Fraction of a full turn.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "point_depth",
                display_name: "Point Depth",
                tooltip: "Star only: 1.0 is a plain polygon (e.g. a triangle at 3 sides); lowering it unfolds the same outline into sharper star points.",
                kind: ParamSpecKind::Float {
                    min: 0.05,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "curvature_top",
                display_name: "Curvature Top",
                tooltip: "Leaf only: how pointed (high) or rounded (low) its top tip is.",
                kind: ParamSpecKind::Float {
                    min: 0.2,
                    max: 4.0,
                    default: 0.6,
                },
            },
            ParamSpec {
                name: "curvature_bottom",
                display_name: "Curvature Bottom",
                tooltip: "Leaf only: how pointed (high) or rounded (low) its bottom tip is.",
                kind: ParamSpecKind::Float {
                    min: 0.2,
                    max: 4.0,
                    default: 0.6,
                },
            },
            ParamSpec {
                name: "curvature_middle",
                display_name: "Curvature Middle",
                tooltip: "Leaf only: how pointed (high) or rounded (low) its sides are through the middle (the equator), independently from either tip.",
                kind: ParamSpecKind::Float {
                    min: 0.2,
                    max: 4.0,
                    default: 0.6,
                },
            },
            ParamSpec {
                name: "leaf_width",
                display_name: "Leaf Width",
                tooltip: "Leaf only: how wide it is at its fattest point (the equator), as a fraction of its own height.",
                kind: ParamSpecKind::Float {
                    min: 0.25,
                    max: 0.9,
                    default: 0.55,
                },
            },
            ParamSpec {
                name: "middle_position",
                display_name: "Middle Position",
                tooltip: "Leaf only: where the equator (widest point) sits between the top tip (low) and bottom tip (high). 0.5 is centered.",
                kind: ParamSpecKind::Float {
                    min: 0.15,
                    max: 0.85,
                    default: 0.5,
                },
            },
        ],
    },
    NodeDef {
        kind: "gradient_linear",
        display_name: "Gradient Linear",
        category: "Generators/Patterns",
        tooltip: "A ramp from black to white, rotatable around the tile's center.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[ParamSpec {
            name: "rotation",
            display_name: "Rotation",
            tooltip: "Ramp direction. Limited to quarter turns — the only angles that keep the tile's edges lining up with its neighbor's.",
            kind: ParamSpecKind::Enum {
                options: &crate::gradient_linear::ROTATION_OPTIONS,
                default: "0°",
            },
        }],
    },
    NodeDef {
        kind: "gradient_radial",
        display_name: "Gradient Radial",
        category: "Generators/Patterns",
        tooltip: "A ramp from black at the center to white at the given radius.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[ParamSpec {
            name: "radius",
            display_name: "Radius",
            tooltip: "Distance from center where the ramp reaches white.",
            kind: ParamSpecKind::Float {
                min: 0.05,
                max: 1.0,
                default: 0.5,
            },
        }],
    },
    NodeDef {
        kind: "uniform_color",
        display_name: "Uniform Color",
        category: "Generators/Patterns",
        tooltip: "A flat, solid color fill.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[ParamSpec {
            name: "color",
            display_name: "Color",
            tooltip: "The fill color.",
            kind: ParamSpecKind::Color {
                default: [0.5, 0.5, 0.5, 1.0],
            },
        }],
    },
    NodeDef {
        kind: "invert",
        display_name: "Invert",
        category: "Filters/Adjustments",
        tooltip: "Flips light and dark: output = 1 - input.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[],
    },
    NodeDef {
        kind: "histogram_scan",
        display_name: "Value Range",
        category: "Filters/Blending & Masking",
        tooltip: "Threshold/window the input around a position.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "position",
                display_name: "Position",
                tooltip: "Where the transition is centered.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "contrast",
                display_name: "Contrast",
                tooltip: "How sharp the transition is (1 = hard edge).",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
        ],
    },
    NodeDef {
        kind: "blur",
        display_name: "Blur",
        category: "Filters/Spatial",
        tooltip: "Gaussian blur — softens detail.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "radius",
                display_name: "Radius",
                tooltip: "Blur radius, in UV space.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.1,
                    default: 0.01,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "bevel",
        display_name: "Bevel",
        category: "Filters/Spatial",
        tooltip: "Rounds a mask or height input's edges into a beveled height map — chain \
                  into Height to Normal for the usual beveled-edge look.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "distance",
                display_name: "Distance",
                tooltip: "How far the bevel reaches from an edge, in UV space.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.2,
                    default: 0.05,
                },
            },
            ParamSpec {
                name: "convexity",
                display_name: "Convexity",
                tooltip: "0 is a plain linear chamfer; positive rounds the edge convex (a \
                          dome), negative makes it concave (a scoop).",
                kind: ParamSpecKind::Float {
                    min: -1.0,
                    max: 1.0,
                    default: 0.3,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "warp",
        display_name: "Warp",
        category: "Filters/Spatial",
        tooltip: "Distort an input by the gradient of an intensity map.",
        inputs: &[
            PortSpec {
                name: "input",
                display_name: "Input",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "intensity",
                display_name: "Intensity",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "amount",
                display_name: "Amount",
                tooltip: "Warp strength.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.2,
                    default: 0.05,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "transform2d",
        display_name: "Transform 2D",
        category: "Filters/Spatial",
        tooltip: "Offset, rotate, and scale, with wrap-around tiling.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "offset_x",
                display_name: "Offset X",
                tooltip: "Horizontal offset.",
                kind: ParamSpecKind::Float {
                    min: -1.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "offset_y",
                display_name: "Offset Y",
                tooltip: "Vertical offset.",
                kind: ParamSpecKind::Float {
                    min: -1.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "rotation",
                display_name: "Rotation",
                tooltip: "Fraction of a full turn. Only multiples of 0.25 stay exactly tileable.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "scale_x",
                display_name: "Scale X",
                tooltip: "Horizontal scale.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 4.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "scale_y",
                display_name: "Scale Y",
                tooltip: "Vertical scale.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 4.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "flip_x",
                display_name: "Flip Horizontal",
                tooltip: "Mirrors the input left-to-right.",
                kind: ParamSpecKind::Bool { default: false },
            },
            ParamSpec {
                name: "flip_y",
                display_name: "Flip Vertical",
                tooltip: "Mirrors the input top-to-bottom.",
                kind: ParamSpecKind::Bool { default: false },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "height_to_normal",
        display_name: "Height to Normal",
        category: "Converters",
        tooltip: "Converts a height map to a tangent-space normal map.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Gray,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "intensity",
                display_name: "Intensity",
                tooltip: "How strongly height differences tilt the normal.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 10.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "format",
                display_name: "Format",
                tooltip: "OpenGL (Y+) for Blender, DirectX (Y-) for Unreal.",
                kind: ParamSpecKind::Enum {
                    options: &crate::height_to_normal::NormalFormat::NAMES,
                    default: "OpenGL",
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "clamp",
        display_name: "Clamp",
        category: "Filters/Adjustments",
        tooltip: "Restricts values to a [low, high] range.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "low",
                display_name: "Low",
                tooltip: "Minimum allowed output value.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "high",
                display_name: "High",
                tooltip: "Maximum allowed output value.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
        ],
    },
    NodeDef {
        kind: "threshold",
        display_name: "Threshold",
        category: "Filters/Blending & Masking",
        tooltip: "Hard black/white cutoff at a given value.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[ParamSpec {
            name: "threshold",
            display_name: "Threshold",
            tooltip: "Values at or above this become white, everything else black.",
            kind: ParamSpecKind::Float {
                min: 0.0,
                max: 1.0,
                default: 0.5,
            },
        }],
    },
    NodeDef {
        kind: "grayscale_conversion",
        display_name: "Grayscale Conversion",
        category: "Filters/Adjustments",
        tooltip: "Converts Color to Gray by luminance.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[],
    },
    NodeDef {
        kind: "checker",
        display_name: "Checker",
        category: "Generators/Patterns",
        tooltip: "An alternating black/white grid.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[ParamSpec {
            name: "scale",
            display_name: "Scale",
            tooltip: "Cells spanning the texture. Rounded up to even for seamless tiling.",
            kind: ParamSpecKind::Int {
                min: 2,
                max: 32,
                default: 8,
            },
        }],
    },
    NodeDef {
        kind: "dither_pattern",
        display_name: "Dither Pattern",
        category: "Generators/Patterns",
        tooltip: "The raw ordered-dither threshold mask (Bayer, blue noise, or a halftone dot \
                  grid), for use in Blend, Threshold, and similar nodes.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "pattern",
                display_name: "Pattern",
                tooltip: "Which ordered-dither matrix to use.",
                kind: ParamSpecKind::Enum {
                    options: &["Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Blue Noise", "Dot"],
                    default: "Bayer 4x4",
                },
            },
            ParamSpec {
                name: "cells_x",
                display_name: "Cells X",
                tooltip: "How many dither cells span the texture horizontally. Rounded up to \
                          the nearest multiple of the pattern's own tile size (2/4/8/64) for \
                          seamless tiling.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 512,
                    default: 64,
                },
            },
            ParamSpec {
                name: "cells_y",
                display_name: "Cells Y",
                tooltip: "How many dither cells span the texture vertically. Rounded the same \
                          way as Cells X.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 512,
                    default: 64,
                },
            },
            ParamSpec {
                name: "rotation",
                display_name: "Rotation",
                tooltip: "Tilts the pattern to a classic halftone screen angle (print screens \
                          are rotated per channel to hide the grid and avoid moire). Limited \
                          to angles that still tile exactly, the same way real halftone \
                          screens are restricted to a handful of rational angles — the cell \
                          size shifts slightly with the angle as a result. No effect on Blue \
                          Noise.",
                kind: ParamSpecKind::Enum {
                    options: &["0°", "18.4°", "26.6°", "45°"],
                    default: "0°",
                },
            },
            ParamSpec {
                name: "dot_size",
                display_name: "Dot Size",
                tooltip: "How much of each cell the circle fills. Only used by the Dot \
                          pattern.",
                kind: ParamSpecKind::Float {
                    min: 0.05,
                    max: 1.0,
                    default: 0.6,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Offsets the Blue Noise mask. Has no effect on other patterns.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    NodeDef {
        kind: "hsl_adjust",
        display_name: "HSL Adjust",
        category: "Filters/Adjustments",
        tooltip: "Shift hue, scale saturation, and shift lightness.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "hue_shift",
                display_name: "Hue Shift",
                tooltip: "Fraction of a full turn to rotate hue by.",
                kind: ParamSpecKind::Float {
                    min: -0.5,
                    max: 0.5,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "saturation",
                display_name: "Saturation",
                tooltip: "Multiplier on saturation (0 = grayscale).",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 2.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "lightness",
                display_name: "Lightness",
                tooltip: "Additive shift on lightness.",
                kind: ParamSpecKind::Float {
                    min: -1.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
        ],
    },
    NodeDef {
        kind: "curve",
        display_name: "Curve",
        category: "Filters/Adjustments",
        tooltip: "A 5-point curve — drag each point's output value.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "v0",
                display_name: "At 0.00",
                tooltip: "Output value when the input is 0.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "v1",
                display_name: "At 0.25",
                tooltip: "Output value when the input is 0.25.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.25,
                },
            },
            ParamSpec {
                name: "v2",
                display_name: "At 0.50",
                tooltip: "Output value when the input is 0.5.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "v3",
                display_name: "At 0.75",
                tooltip: "Output value when the input is 0.75.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.75,
                },
            },
            ParamSpec {
                name: "v4",
                display_name: "At 1.00",
                tooltip: "Output value when the input is 1.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
        ],
    },
    NodeDef {
        kind: "gradient_map",
        display_name: "Gradient Map",
        category: "Filters/Adjustments",
        tooltip: "Maps a Gray input through a color gradient.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Gray,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[ParamSpec {
            name: "gradient",
            display_name: "Gradient",
            tooltip: "Color stops the input value is mapped through, in position order.",
            kind: ParamSpecKind::Gradient {
                default: &[
                    GradientStop {
                        position: 0.0,
                        color: [0.0, 0.0, 0.0, 1.0],
                    },
                    GradientStop {
                        position: 1.0,
                        color: [1.0, 1.0, 1.0, 1.0],
                    },
                ],
            },
        }],
    },
    NodeDef {
        kind: "simplex_noise",
        display_name: "Simplex Noise",
        category: "Generators/Noise",
        tooltip:
            "Gradient noise on a triangular lattice — fewer directional artifacts than Perlin.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "scale",
                display_name: "Scale",
                tooltip: "How many noise cells tile across the texture.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 4,
                },
            },
            ParamSpec {
                name: "octaves",
                display_name: "Octaves",
                tooltip: "How many layers of noise are summed.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 8,
                    default: 4,
                },
            },
            ParamSpec {
                name: "persistence",
                display_name: "Persistence",
                tooltip: "How much each octave's contribution shrinks.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the pattern without changing its character.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    NodeDef {
        kind: "white_noise",
        display_name: "White Noise",
        category: "Generators/Noise",
        tooltip: "Fully uncorrelated random cells — no smooth interpolation.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "scale",
                display_name: "Scale",
                tooltip: "Cells spanning the texture.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 16,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the pattern.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    NodeDef {
        kind: "clouds",
        display_name: "Clouds",
        category: "Generators/Noise",
        tooltip: "An fBm preset tuned for a cloud-like look.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "scale",
                display_name: "Scale",
                tooltip: "How many noise cells tile across the texture.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 4,
                },
            },
            ParamSpec {
                name: "softness",
                display_name: "Softness",
                tooltip: "How soft/fluffy the clouds look.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the pattern without changing its character.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    NodeDef {
        kind: "directional_blur",
        display_name: "Directional Blur",
        category: "Filters/Spatial",
        tooltip: "A motion-blur-like smear along one fixed direction.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "radius",
                display_name: "Radius",
                tooltip: "Blur radius, in UV space.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.1,
                    default: 0.02,
                },
            },
            ParamSpec {
                name: "angle",
                display_name: "Angle",
                tooltip: "Blur direction, as a fraction of a full turn.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "directional_warp",
        display_name: "Push Warp",
        category: "Filters/Spatial",
        tooltip: "Shift pixels along one fixed direction by an intensity map's value.",
        inputs: &[
            PortSpec {
                name: "input",
                display_name: "Input",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "intensity",
                display_name: "Intensity",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "angle",
                display_name: "Angle",
                tooltip: "Warp direction, as a fraction of a full turn.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "amount",
                display_name: "Amount",
                tooltip: "Warp strength.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.2,
                    default: 0.1,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "edge_detect",
        display_name: "Edge Detect",
        category: "Filters/Spatial",
        tooltip: "Highlights where the input changes sharply.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        // Not type-preserving like its neighbors above — its own .wgsl file
        // reduces the input to luminance and broadcasts the Sobel gradient
        // magnitude of *that* back out to every channel, so the output is
        // always Gray-equivalent (R=G=B) regardless of whether the input
        // was genuine Color or a Gray broadcast. A real converter, same
        // category as Grayscale Conversion/Curvature from Normal, not a
        // passthrough that happens to need its real type resolved.
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "intensity",
                display_name: "Intensity",
                tooltip: "Multiplier on the detected edge strength.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 16.0,
                    default: 4.0,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "slope_blur",
        display_name: "Flow Blur",
        category: "Filters/Spatial",
        tooltip: "Blur along a control map's gradient, strength scaled by its steepness.",
        inputs: &[
            PortSpec {
                name: "input",
                display_name: "Input",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "control",
                display_name: "Control",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "radius",
                display_name: "Radius",
                tooltip: "Maximum blur radius, reached at the control map's steepest point.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.1,
                    default: 0.03,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "height_to_ao",
        display_name: "Height to AO",
        category: "Converters",
        tooltip: "A simple screen-space-style ambient occlusion map from height.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Gray,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "radius",
                display_name: "Radius",
                tooltip: "How far out the occlusion sample reaches, in UV space.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.1,
                    default: 0.02,
                },
            },
            ParamSpec {
                name: "intensity",
                display_name: "Intensity",
                tooltip: "Multiplier on the occlusion strength.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 8.0,
                    default: 2.0,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "curvature_from_normal",
        display_name: "Curvature from Normal",
        category: "Converters",
        tooltip: "Convex edges read bright, concave crevices read dark.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "intensity",
                display_name: "Intensity",
                tooltip: "Multiplier on the curvature strength.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 16.0,
                    default: 4.0,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "channel_split",
        display_name: "Channel Split",
        category: "Filters/Glitch",
        tooltip: "Samples each RGB channel at its own offset, for chromatic-aberration/glitch \
                  looks.",
        inputs: &[
            PortSpec {
                name: "image",
                display_name: "Image",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "offset_r",
                display_name: "Offset R",
                tooltip: "Red channel's sample offset, in UV units.",
                kind: ParamSpecKind::Vec2 {
                    min: -0.5,
                    max: 0.5,
                    default: [0.01, 0.0],
                },
            },
            ParamSpec {
                name: "offset_g",
                display_name: "Offset G",
                tooltip: "Green channel's sample offset, in UV units.",
                kind: ParamSpecKind::Vec2 {
                    min: -0.5,
                    max: 0.5,
                    default: [0.0, 0.0],
                },
            },
            ParamSpec {
                name: "offset_b",
                display_name: "Offset B",
                tooltip: "Blue channel's sample offset, in UV units.",
                kind: ParamSpecKind::Vec2 {
                    min: -0.5,
                    max: 0.5,
                    default: [-0.01, 0.0],
                },
            },
            ParamSpec {
                name: "strength",
                display_name: "Strength",
                tooltip: "Blends between the original image (0) and the fully split channels \
                          (1).",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "block_shift",
        display_name: "Block Shift",
        category: "Filters/Glitch",
        tooltip: "Shifts random blocks of the image sideways — a seeded hash per block \
                  decides whether and how far.",
        inputs: &[
            PortSpec {
                name: "image",
                display_name: "Image",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "columns",
                display_name: "Columns",
                tooltip: "How many blocks span the texture horizontally.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 8,
                },
            },
            ParamSpec {
                name: "rows",
                display_name: "Rows",
                tooltip: "How many blocks span the texture vertically.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 8,
                },
            },
            ParamSpec {
                name: "amount",
                display_name: "Amount",
                tooltip: "Fraction of blocks that shift at all.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.3,
                },
            },
            ParamSpec {
                name: "max_offset",
                display_name: "Max Offset",
                tooltip: "Largest possible shift for an affected block, in UV units.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.5,
                    default: 0.1,
                },
            },
            ParamSpec {
                name: "direction",
                display_name: "Direction",
                tooltip: "Which axis a shifting block moves along.",
                kind: ParamSpecKind::Enum {
                    options: &["Horizontal", "Vertical", "Both"],
                    default: "Both",
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes which blocks shift and by how much.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "scanline_jitter",
        display_name: "Scanline Jitter",
        category: "Filters/Glitch",
        tooltip: "Shifts random horizontal or vertical bands sideways — the classic 'bad \
                  video signal' glitch.",
        inputs: &[
            PortSpec {
                name: "image",
                display_name: "Image",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "lines",
                display_name: "Lines",
                tooltip: "How many bands span the texture.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 256,
                    default: 32,
                },
            },
            ParamSpec {
                name: "amount",
                display_name: "Amount",
                tooltip: "Fraction of bands that jitter at all.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.3,
                },
            },
            ParamSpec {
                name: "max_offset",
                display_name: "Max Offset",
                tooltip: "Largest possible jitter for an affected band, in UV units.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.5,
                    default: 0.05,
                },
            },
            ParamSpec {
                name: "orientation",
                display_name: "Orientation",
                tooltip: "Whether bands run horizontally or vertically.",
                kind: ParamSpecKind::Enum {
                    options: &["Horizontal", "Vertical"],
                    default: "Horizontal",
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes which bands jitter and by how much.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "ordered_dither",
        display_name: "Ordered Dither",
        category: "Filters/Stylize",
        tooltip: "Reduces an image to a few tones with an ordered (Bayer or blue noise) \
                  dither pattern.",
        inputs: &[PortSpec {
            name: "image",
            display_name: "Image",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "pattern",
                display_name: "Pattern",
                tooltip: "Which ordered-dither matrix to use.",
                kind: ParamSpecKind::Enum {
                    options: &["Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Blue Noise"],
                    default: "Bayer 4x4",
                },
            },
            ParamSpec {
                name: "cells",
                display_name: "Cells",
                tooltip: "How many dither cells span the texture. Rounded up to the nearest \
                          multiple of the pattern's own tile size (2/4/8/64) for seamless \
                          tiling.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 512,
                    default: 64,
                },
            },
            ParamSpec {
                name: "levels",
                display_name: "Levels",
                tooltip: "Tone steps per channel. 2 = pure 1-bit (black/white only).",
                kind: ParamSpecKind::Int {
                    min: 2,
                    max: 16,
                    default: 4,
                },
            },
            ParamSpec {
                name: "spread",
                display_name: "Spread",
                tooltip: "Dither strength: 0 is a plain posterize with no dithering, 1 is the \
                          full dither pattern.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "bias",
                display_name: "Bias",
                tooltip: "Shifts the threshold lighter or darker.",
                kind: ParamSpecKind::Float {
                    min: -0.5,
                    max: 0.5,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "mode",
                display_name: "Mode",
                tooltip: "Per Channel dithers R/G/B independently. Luminance quantizes \
                          brightness only and rescales all 3 channels together, preserving \
                          hue. No visible difference on a Gray-sourced image.",
                kind: ParamSpecKind::Enum {
                    options: &["Per Channel", "Luminance"],
                    default: "Luminance",
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Offsets the Blue Noise mask. Has no effect on Bayer patterns.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    NodeDef {
        kind: "palette_dither",
        display_name: "Palette Dither",
        category: "Filters/Stylize",
        tooltip: "Dithers a Color image down to a fixed palette of 2-8 colors.",
        inputs: &[PortSpec {
            name: "image",
            display_name: "Image",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "palette",
                display_name: "Palette",
                tooltip: "The colors to dither down to, in order (position only decides \
                          order here, not blending).",
                kind: ParamSpecKind::Gradient {
                    default: &[
                        GradientStop {
                            position: 0.0,
                            color: [0.0, 0.0, 0.0, 1.0],
                        },
                        GradientStop {
                            position: 0.5,
                            color: [0.5, 0.5, 0.5, 1.0],
                        },
                        GradientStop {
                            position: 1.0,
                            color: [1.0, 1.0, 1.0, 1.0],
                        },
                    ],
                },
            },
            ParamSpec {
                name: "pattern",
                display_name: "Pattern",
                tooltip: "Which ordered-dither matrix to use.",
                kind: ParamSpecKind::Enum {
                    options: &["Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Blue Noise"],
                    default: "Bayer 4x4",
                },
            },
            ParamSpec {
                name: "cells",
                display_name: "Cells",
                tooltip: "How many dither cells span the texture. Rounded up to the nearest \
                          multiple of the pattern's own tile size (2/4/8/64) for seamless \
                          tiling.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 512,
                    default: 64,
                },
            },
            ParamSpec {
                name: "spread",
                display_name: "Spread",
                tooltip: "Dither strength: 0 picks the nearest palette color directly with no \
                          dithering, 1 uses the full dither pattern.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "distance",
                display_name: "Distance",
                tooltip: "How \"nearest\" is measured when picking a palette color. RGB is \
                          plain Euclidean distance; Perceptual weights channels by luminance.",
                kind: ParamSpecKind::Enum {
                    options: &["RGB", "Perceptual"],
                    default: "Perceptual",
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Offsets the Blue Noise mask. Has no effect on Bayer patterns.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    NodeDef {
        kind: "data_corrupt",
        display_name: "Data Corrupt",
        category: "Filters/Glitch",
        tooltip: "Posterizes, channel-swaps, or inverts random blocks — a glitchy 'corrupted \
                  file' look.",
        inputs: &[
            PortSpec {
                name: "image",
                display_name: "Image",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "columns",
                display_name: "Columns",
                tooltip: "How many blocks span the texture horizontally.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 8,
                },
            },
            ParamSpec {
                name: "rows",
                display_name: "Rows",
                tooltip: "How many blocks span the texture vertically.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 8,
                },
            },
            ParamSpec {
                name: "amount",
                display_name: "Amount",
                tooltip: "Fraction of blocks that get corrupted.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.2,
                },
            },
            ParamSpec {
                name: "effect",
                display_name: "Effect",
                tooltip: "Posterize crushes a block to 3 tones; Channel Swap permutes its RGB \
                          channels; Invert flips it; Mix picks one of the three per block.",
                kind: ParamSpecKind::Enum {
                    options: &["Posterize", "Channel Swap", "Invert", "Mix"],
                    default: "Mix",
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes which blocks are affected and how.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    output_node_def("output_base_color", "Output: Base Color"),
    output_node_def("output_normal", "Output: Normal"),
    output_node_def("output_roughness", "Output: Roughness"),
    output_node_def("output_metallic", "Output: Metallic"),
    output_node_def("output_height", "Output: Height"),
    output_node_def("output_ambient_occlusion", "Output: Ambient Occlusion"),
    output_node_def("output_emissive", "Output: Emissive"),
    output_node_def("output_alpha", "Output: Alpha"),
];

#[cfg(test)]
mod tests {
    use super::effective_output_type;
    use mc_core::{Graph, Node, PortType};

    /// Regression test for the Levels/Invert/Clamp/Threshold/Curve fix: a
    /// type-preserving node fed a Gray source reports Gray, not its
    /// statically-declared Color.
    #[test]
    fn type_preserving_node_follows_a_gray_source() {
        let mut graph = Graph::new();
        let noise = graph.add_node(Node::new("perlin_noise")); // declared Gray
        let levels = graph.add_node(Node::new("levels"));
        graph.connect(noise, "output", levels, "input").unwrap();

        assert_eq!(
            effective_output_type(&graph, levels, "levels", PortType::Color),
            PortType::Gray
        );
    }

    /// An unconnected type-preserving node has nothing to follow, so it falls
    /// back to its declared type.
    #[test]
    fn type_preserving_node_with_no_input_falls_back_to_declared_type() {
        let mut graph = Graph::new();
        let levels = graph.add_node(Node::new("levels"));

        assert_eq!(
            effective_output_type(&graph, levels, "levels", PortType::Color),
            PortType::Color
        );
    }

    /// Blend fed two Gray-sourced inputs ("While testing I realized that the
    /// Blend node always considers inputs to be color and therefore only
    /// outputs color") should now report Gray.
    #[test]
    fn blend_with_two_gray_inputs_reports_gray_output() {
        let mut graph = Graph::new();
        let noise_a = graph.add_node(Node::new("perlin_noise"));
        let noise_b = graph.add_node(Node::new("perlin_noise"));
        let blend = graph.add_node(Node::new("blend"));
        graph.connect(noise_a, "output", blend, "base").unwrap();
        graph.connect(noise_b, "output", blend, "blend").unwrap();

        assert_eq!(
            effective_output_type(&graph, blend, "blend", PortType::Color),
            PortType::Gray
        );
    }

    /// Color wins: a Blend with one Gray and one Color input must report
    /// Color, since the Color input can carry real chroma.
    #[test]
    fn blend_with_one_color_input_reports_color_output() {
        let mut graph = Graph::new();
        let noise = graph.add_node(Node::new("perlin_noise")); // Gray
        let uniform_color = graph.add_node(Node::new("uniform_color")); // Color
        let blend = graph.add_node(Node::new("blend"));
        graph.connect(noise, "output", blend, "base").unwrap();
        graph
            .connect(uniform_color, "output", blend, "blend")
            .unwrap();

        assert_eq!(
            effective_output_type(&graph, blend, "blend", PortType::Color),
            PortType::Color
        );
    }

    /// A Blend with only one input wired defers entirely to that input's
    /// effective type, rather than falling back to the declared Color.
    #[test]
    fn blend_with_only_one_input_wired_follows_that_input() {
        let mut graph = Graph::new();
        let noise = graph.add_node(Node::new("perlin_noise")); // Gray
        let blend = graph.add_node(Node::new("blend"));
        graph.connect(noise, "output", blend, "base").unwrap();

        assert_eq!(
            effective_output_type(&graph, blend, "blend", PortType::Color),
            PortType::Gray
        );
    }

    /// A Blend with neither input wired falls back to its declared Color.
    #[test]
    fn blend_with_no_inputs_falls_back_to_declared_type() {
        let mut graph = Graph::new();
        let blend = graph.add_node(Node::new("blend"));

        assert_eq!(
            effective_output_type(&graph, blend, "blend", PortType::Color),
            PortType::Color
        );
    }

    /// The fix this test guards: resolving an upstream Blend's type must
    /// recurse into its *effective* type, not its merely-declared Color — a
    /// chain of Gray source -> Blend -> Levels must still report Gray all the
    /// way through, since Levels is type-preserving and Blend's two Gray
    /// inputs make it effectively Gray too.
    #[test]
    fn chained_gray_blend_through_a_type_preserving_node_stays_gray() {
        let mut graph = Graph::new();
        let noise_a = graph.add_node(Node::new("perlin_noise"));
        let noise_b = graph.add_node(Node::new("perlin_noise"));
        let blend = graph.add_node(Node::new("blend"));
        let levels = graph.add_node(Node::new("levels"));
        graph.connect(noise_a, "output", blend, "base").unwrap();
        graph.connect(noise_b, "output", blend, "blend").unwrap();
        graph.connect(blend, "output", levels, "input").unwrap();

        assert_eq!(
            effective_output_type(&graph, levels, "levels", PortType::Color),
            PortType::Gray
        );
    }

    /// Regression test for the wider audit this prompted ("a bunch of other
    /// nodes... pretend to be color value"): Transform 2D (and its
    /// newly-added neighbors) fed a Gray source must report Gray, the exact
    /// same bug class as the original Levels/Blend fixes, just not caught
    /// the first time around.
    #[test]
    fn newly_audited_type_preserving_nodes_follow_a_gray_source() {
        for kind in [
            "histogram_scan",
            "blur",
            "transform2d",
            "directional_blur",
            "bevel",
        ] {
            let mut graph = Graph::new();
            let noise = graph.add_node(Node::new("perlin_noise"));
            let node = graph.add_node(Node::new(kind));
            graph.connect(noise, "output", node, "input").unwrap();

            assert_eq!(
                effective_output_type(&graph, node, kind, PortType::Color),
                PortType::Gray,
                "{kind} should report Gray when fed a Gray source"
            );
        }
    }

    /// Warp, Push Warp, and Flow Blur each take a *second*, Gray-only
    /// "intensity"/"control" input that drives the distortion but isn't
    /// itself part of the output's content — the effective type must follow
    /// the main "input" port only, not get confused by that second port.
    #[test]
    fn dual_input_type_preserving_nodes_follow_their_main_input_only() {
        for (kind, second_port) in [
            ("warp", "intensity"),
            ("directional_warp", "intensity"),
            ("slope_blur", "control"),
        ] {
            let mut graph = Graph::new();
            let main_source = graph.add_node(Node::new("perlin_noise")); // Gray
            let control_source = graph.add_node(Node::new("perlin_noise")); // Gray too
            let node = graph.add_node(Node::new(kind));
            graph.connect(main_source, "output", node, "input").unwrap();
            graph
                .connect(control_source, "output", node, second_port)
                .unwrap();

            assert_eq!(
                effective_output_type(&graph, node, kind, PortType::Color),
                PortType::Gray,
                "{kind} should follow its main input's type"
            );
        }
    }

    /// Edge Detect is a real converter, not a passthrough: its own .wgsl
    /// reduces the input to luminance and broadcasts that back out, so the
    /// output is always Gray-equivalent regardless of whether the input was
    /// genuine Color or a Gray broadcast — declared statically Gray, not
    /// resolved via the type-preserving mechanism at all.
    #[test]
    fn edge_detect_output_is_always_gray_even_with_a_color_input() {
        let mut graph = Graph::new();
        let color_source = graph.add_node(Node::new("uniform_color"));
        let edge = graph.add_node(Node::new("edge_detect"));
        graph
            .connect(color_source, "output", edge, "input")
            .unwrap();

        assert_eq!(
            effective_output_type(&graph, edge, "edge_detect", PortType::Gray),
            PortType::Gray
        );
    }
}
