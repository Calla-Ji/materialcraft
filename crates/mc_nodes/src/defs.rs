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

use mc_core::{ParamValue, PortType};

/// One parameter a node declares, independent of its current value in any
/// particular graph.
pub struct ParamSpec {
    pub name: &'static str,
    pub display_name: &'static str,
    pub tooltip: &'static str,
    pub kind: ParamSpecKind,
}

/// A param's type and allowed range/options. New variants (Bool, Color,
/// Vec2, Gradient) get added here once a node that needs them exists —
/// Perlin Noise, Levels, and Blend only need these four.
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
}

impl ParamSpec {
    fn default_value(&self) -> ParamValue {
        match self.kind {
            ParamSpecKind::Float { default, .. } => ParamValue::Float(default),
            ParamSpecKind::Int { default, .. } => ParamValue::Int(default),
            ParamSpecKind::Seed { default } => ParamValue::Seed(default),
            ParamSpecKind::Enum { default, .. } => ParamValue::Enum(default.to_string()),
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
    output_node_def("output_base_color", "Output: Base Color"),
    output_node_def("output_normal", "Output: Normal"),
    output_node_def("output_roughness", "Output: Roughness"),
    output_node_def("output_metallic", "Output: Metallic"),
    output_node_def("output_height", "Output: Height"),
    output_node_def("output_ambient_occlusion", "Output: Ambient Occlusion"),
    output_node_def("output_emissive", "Output: Emissive"),
];
