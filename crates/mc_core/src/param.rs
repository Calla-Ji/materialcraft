//! Parameter values a node's `params` map can hold (CLAUDE.md §5).
//!
//! The *meaning* of a param — its min/max/default, what it does — is node
//! metadata that lives in `mc_nodes`, keyed by the node's `kind`. This is
//! just the generic value shape every node's `params` map can store.

use serde::{Deserialize, Serialize};

/// One stop in a [`ParamValue::Gradient`] (used by the Gradient Map node).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GradientStop {
    pub position: f32,
    pub color: [f32; 4],
}

impl std::hash::Hash for GradientStop {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.position.to_bits().hash(state);
        for channel in self.color {
            channel.to_bits().hash(state);
        }
    }
}

/// A single parameter's value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum ParamValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    Enum(String),
    Color([f32; 4]),
    Vec2([f32; 2]),
    Seed(u32),
    Gradient(Vec<GradientStop>),
}

impl std::hash::Hash for ParamValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // f32 isn't `Hash` (NaN has no well-defined bitwise identity rule in
        // general, but for our purposes — cache-key stability — hashing the
        // raw bits is exactly what we want).
        core::mem::discriminant(self).hash(state);
        match self {
            ParamValue::Float(v) => v.to_bits().hash(state),
            ParamValue::Int(v) => v.hash(state),
            ParamValue::Bool(v) => v.hash(state),
            ParamValue::Enum(v) => v.hash(state),
            ParamValue::Color(v) => v.iter().for_each(|c| c.to_bits().hash(state)),
            ParamValue::Vec2(v) => v.iter().for_each(|c| c.to_bits().hash(state)),
            ParamValue::Seed(v) => v.hash(state),
            ParamValue::Gradient(stops) => stops.hash(state),
        }
    }
}

impl ParamValue {
    pub fn as_float(&self) -> Option<f32> {
        match self {
            ParamValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i32> {
        match self {
            ParamValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParamValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_seed(&self) -> Option<u32> {
        match self {
            ParamValue::Seed(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_vec2(&self) -> Option<[f32; 2]> {
        match self {
            ParamValue::Vec2(v) => Some(*v),
            _ => None,
        }
    }
}
