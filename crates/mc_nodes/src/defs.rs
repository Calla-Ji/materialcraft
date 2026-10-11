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

/// A param's type and allowed range/options. New variants (Bool, Vec2) get
/// added here once a node that needs them exists.
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
    /// Gradient Map's color ramp — at most 8 stops (`gradient_map.wgsl`'s
    /// fixed-size uniform array), interpolated in position order.
    Gradient {
        default: &'static [GradientStop],
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
            ParamSpecKind::Gradient { default } => ParamValue::Gradient(default.to_vec()),
        }
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
        display_name: "Tile Generator",
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
                name: "bevel",
                display_name: "Bevel",
                tooltip: "How much of each tile's edge softens toward the gap.",
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
        tooltip: "A single disc, square, polygon, or star, centered in the tile.",
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
        display_name: "Histogram Scan",
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
        params: &[ParamSpec {
            name: "radius",
            display_name: "Radius",
            tooltip: "Blur radius, in UV space.",
            kind: ParamSpecKind::Float {
                min: 0.0,
                max: 0.1,
                default: 0.01,
            },
        }],
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
        params: &[ParamSpec {
            name: "amount",
            display_name: "Amount",
            tooltip: "Warp strength.",
            kind: ParamSpecKind::Float {
                min: 0.0,
                max: 0.2,
                default: 0.05,
            },
        }],
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
        ],
    },
    NodeDef {
        kind: "directional_warp",
        display_name: "Directional Warp",
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
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[ParamSpec {
            name: "intensity",
            display_name: "Intensity",
            tooltip: "Multiplier on the detected edge strength.",
            kind: ParamSpecKind::Float {
                min: 0.1,
                max: 16.0,
                default: 4.0,
            },
        }],
    },
    NodeDef {
        kind: "slope_blur",
        display_name: "Slope Blur",
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
        params: &[ParamSpec {
            name: "radius",
            display_name: "Radius",
            tooltip: "Maximum blur radius, reached at the control map's steepest point.",
            kind: ParamSpecKind::Float {
                min: 0.0,
                max: 0.1,
                default: 0.03,
            },
        }],
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
        params: &[ParamSpec {
            name: "intensity",
            display_name: "Intensity",
            tooltip: "Multiplier on the curvature strength.",
            kind: ParamSpecKind::Float {
                min: 0.1,
                max: 16.0,
                default: 4.0,
            },
        }],
    },
    output_node_def("output_base_color", "Output: Base Color"),
    output_node_def("output_normal", "Output: Normal"),
    output_node_def("output_roughness", "Output: Roughness"),
    output_node_def("output_metallic", "Output: Metallic"),
    output_node_def("output_height", "Output: Height"),
    output_node_def("output_ambient_occlusion", "Output: Ambient Occlusion"),
    output_node_def("output_emissive", "Output: Emissive"),
];
