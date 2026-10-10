//! Export naming presets (CLAUDE.md §8): fixed for v0.1 rather than the
//! fully editable templates the spec eventually wants, but using the same
//! `{name}` substitution so making them editable later is a small change.

/// Which target engine's conventions to export for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    Blender,
    Unreal,
}

impl Preset {
    pub const NAMES: [&'static str; 2] = ["Blender", "Unreal"];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Blender" => Some(Self::Blender),
            "Unreal" => Some(Self::Unreal),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Preset::Blender => "Blender",
            Preset::Unreal => "Unreal",
        }
    }

    /// The normal map convention this preset expects.
    pub fn normal_format(self) -> mc_core::NormalFormat {
        match self {
            Preset::Blender => mc_core::NormalFormat::OpenGl,
            Preset::Unreal => mc_core::NormalFormat::DirectX,
        }
    }

    pub fn base_color_file_name(self, graph_name: &str) -> String {
        match self {
            Preset::Blender => format!("{graph_name}_BaseColor.png"),
            Preset::Unreal => format!("T_{graph_name}_BC.png"),
        }
    }

    pub fn normal_file_name(self, graph_name: &str) -> String {
        match self {
            Preset::Blender => format!("{graph_name}_Normal_GL.png"),
            Preset::Unreal => format!("T_{graph_name}_N.png"),
        }
    }

    pub fn height_file_name(self, graph_name: &str) -> String {
        match self {
            Preset::Blender => format!("{graph_name}_Height.png"),
            Preset::Unreal => format!("T_{graph_name}_H.png"),
        }
    }

    /// `None` for Unreal — it packs these into one ORM texture instead, via
    /// [`Self::orm_file_name`].
    pub fn roughness_file_name(self, graph_name: &str) -> Option<String> {
        match self {
            Preset::Blender => Some(format!("{graph_name}_Roughness.png")),
            Preset::Unreal => None,
        }
    }

    pub fn metallic_file_name(self, graph_name: &str) -> Option<String> {
        match self {
            Preset::Blender => Some(format!("{graph_name}_Metallic.png")),
            Preset::Unreal => None,
        }
    }

    pub fn ambient_occlusion_file_name(self, graph_name: &str) -> Option<String> {
        match self {
            Preset::Blender => Some(format!("{graph_name}_AO.png")),
            Preset::Unreal => None,
        }
    }

    /// `None` for Blender — it exports Roughness/Metallic/AO as separate
    /// textures instead.
    pub fn orm_file_name(self, graph_name: &str) -> Option<String> {
        match self {
            Preset::Blender => None,
            Preset::Unreal => Some(format!("T_{graph_name}_ORM.png")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blender_names_match_spec() {
        assert_eq!(
            Preset::Blender.base_color_file_name("bricks"),
            "bricks_BaseColor.png"
        );
        assert_eq!(
            Preset::Blender.normal_file_name("bricks"),
            "bricks_Normal_GL.png"
        );
        assert_eq!(
            Preset::Blender.roughness_file_name("bricks"),
            Some("bricks_Roughness.png".to_string())
        );
        assert_eq!(
            Preset::Blender.metallic_file_name("bricks"),
            Some("bricks_Metallic.png".to_string())
        );
        assert_eq!(
            Preset::Blender.height_file_name("bricks"),
            "bricks_Height.png"
        );
        assert_eq!(
            Preset::Blender.ambient_occlusion_file_name("bricks"),
            Some("bricks_AO.png".to_string())
        );
        assert_eq!(Preset::Blender.orm_file_name("bricks"), None);
        assert_eq!(
            Preset::Blender.normal_format(),
            mc_core::NormalFormat::OpenGl
        );
    }

    #[test]
    fn unreal_names_match_spec() {
        assert_eq!(
            Preset::Unreal.base_color_file_name("bricks"),
            "T_bricks_BC.png"
        );
        assert_eq!(Preset::Unreal.normal_file_name("bricks"), "T_bricks_N.png");
        assert_eq!(Preset::Unreal.height_file_name("bricks"), "T_bricks_H.png");
        assert_eq!(
            Preset::Unreal.orm_file_name("bricks"),
            Some("T_bricks_ORM.png".to_string())
        );
        assert_eq!(Preset::Unreal.roughness_file_name("bricks"), None);
        assert_eq!(Preset::Unreal.metallic_file_name("bricks"), None);
        assert_eq!(Preset::Unreal.ambient_occlusion_file_name("bricks"), None);
        assert_eq!(
            Preset::Unreal.normal_format(),
            mc_core::NormalFormat::DirectX
        );
    }

    #[test]
    fn preset_name_round_trips() {
        for name in Preset::NAMES {
            assert!(Preset::from_name(name).is_some(), "{name:?} should parse");
        }
    }
}
