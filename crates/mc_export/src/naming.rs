// SPDX-License-Identifier: MIT OR Apache-2.0
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
    pub const NAMES: [&'static str; 2] = ["Blender", "Unreal Engine"];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Blender" => Some(Self::Blender),
            "Unreal Engine" => Some(Self::Unreal),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Preset::Blender => "Blender",
            Preset::Unreal => "Unreal Engine",
        }
    }

    /// The normal map convention this preset expects.
    pub fn normal_format(self) -> mc_core::NormalFormat {
        match self {
            Preset::Blender => mc_core::NormalFormat::OpenGl,
            Preset::Unreal => mc_core::NormalFormat::DirectX,
        }
    }

    /// `ext` ("png"/"tga") is only consulted for Unreal — Blender always
    /// writes `.png`, matching its own fixed export format.
    pub fn base_color_file_name(self, graph_name: &str, ext: &str) -> String {
        match self {
            Preset::Blender => format!("{graph_name}_BaseColor.png"),
            Preset::Unreal => format!("T_{graph_name}_BC.{ext}"),
        }
    }

    pub fn normal_file_name(self, graph_name: &str, ext: &str) -> String {
        match self {
            Preset::Blender => format!("{graph_name}_Normal_GL.png"),
            Preset::Unreal => format!("T_{graph_name}_N.{ext}"),
        }
    }

    /// Always `.png` — Height is 16-bit for both presets, and TGA has no
    /// clean 16-bit single-channel equivalent, so this never consults a
    /// format choice.
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
    pub fn orm_file_name(self, graph_name: &str, ext: &str) -> Option<String> {
        match self {
            Preset::Blender => None,
            Preset::Unreal => Some(format!("T_{graph_name}_ORM.{ext}")),
        }
    }

    pub fn alpha_file_name(self, graph_name: &str) -> String {
        match self {
            Preset::Blender => format!("{graph_name}_Alpha.png"),
            Preset::Unreal => format!("T_{graph_name}_A.png"),
        }
    }

    /// Emissive's own filename — both presets can produce this file (unlike
    /// most of the methods above, which are genuinely preset-specific);
    /// whether it's actually *written* is a per-export toggle decided by
    /// the caller, not something this method knows about.
    pub fn emissive_file_name(self, graph_name: &str, ext: &str) -> String {
        match self {
            Preset::Blender => format!("{graph_name}_Emissive.png"),
            Preset::Unreal => format!("T_{graph_name}_E.{ext}"),
        }
    }

    /// `None` for Blender, which has no "separate opacity mask" concept —
    /// only its own always-on [`Self::alpha_file_name`]. `Some` for Unreal,
    /// used only when opacity is exported as its own file rather than
    /// folded into Base Color's alpha channel.
    pub fn opacity_mask_file_name(self, graph_name: &str, ext: &str) -> Option<String> {
        match self {
            Preset::Blender => None,
            Preset::Unreal => Some(format!("T_{graph_name}_M.{ext}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blender_names_match_spec() {
        assert_eq!(
            Preset::Blender.base_color_file_name("bricks", "png"),
            "bricks_BaseColor.png"
        );
        assert_eq!(
            Preset::Blender.normal_file_name("bricks", "png"),
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
        assert_eq!(Preset::Blender.orm_file_name("bricks", "png"), None);
        assert_eq!(
            Preset::Blender.alpha_file_name("bricks"),
            "bricks_Alpha.png"
        );
        assert_eq!(
            Preset::Blender.emissive_file_name("bricks", "png"),
            "bricks_Emissive.png"
        );
        assert_eq!(
            Preset::Blender.opacity_mask_file_name("bricks", "png"),
            None
        );
        assert_eq!(
            Preset::Blender.normal_format(),
            mc_core::NormalFormat::OpenGl
        );
        // `ext` is ignored for Blender, which always writes PNG.
        assert_eq!(
            Preset::Blender.base_color_file_name("bricks", "tga"),
            "bricks_BaseColor.png"
        );
    }

    #[test]
    fn unreal_names_match_spec() {
        assert_eq!(
            Preset::Unreal.base_color_file_name("bricks", "png"),
            "T_bricks_BC.png"
        );
        assert_eq!(
            Preset::Unreal.normal_file_name("bricks", "png"),
            "T_bricks_N.png"
        );
        assert_eq!(Preset::Unreal.height_file_name("bricks"), "T_bricks_H.png");
        assert_eq!(
            Preset::Unreal.orm_file_name("bricks", "png"),
            Some("T_bricks_ORM.png".to_string())
        );
        assert_eq!(Preset::Unreal.roughness_file_name("bricks"), None);
        assert_eq!(Preset::Unreal.metallic_file_name("bricks"), None);
        assert_eq!(Preset::Unreal.ambient_occlusion_file_name("bricks"), None);
        assert_eq!(Preset::Unreal.alpha_file_name("bricks"), "T_bricks_A.png");
        assert_eq!(
            Preset::Unreal.emissive_file_name("bricks", "png"),
            "T_bricks_E.png"
        );
        assert_eq!(
            Preset::Unreal.opacity_mask_file_name("bricks", "png"),
            Some("T_bricks_M.png".to_string())
        );
        assert_eq!(
            Preset::Unreal.normal_format(),
            mc_core::NormalFormat::DirectX
        );
        assert_eq!(Preset::Unreal.name(), "Unreal Engine");
    }

    /// TGA is Unreal-only — every `ext`-taking method actually honors it.
    #[test]
    fn unreal_names_honor_tga_extension() {
        assert_eq!(
            Preset::Unreal.base_color_file_name("bricks", "tga"),
            "T_bricks_BC.tga"
        );
        assert_eq!(
            Preset::Unreal.normal_file_name("bricks", "tga"),
            "T_bricks_N.tga"
        );
        assert_eq!(
            Preset::Unreal.orm_file_name("bricks", "tga"),
            Some("T_bricks_ORM.tga".to_string())
        );
        assert_eq!(
            Preset::Unreal.emissive_file_name("bricks", "tga"),
            "T_bricks_E.tga"
        );
        assert_eq!(
            Preset::Unreal.opacity_mask_file_name("bricks", "tga"),
            Some("T_bricks_M.tga".to_string())
        );
        // Height stays PNG regardless — no `ext` param to even pass one.
        assert_eq!(Preset::Unreal.height_file_name("bricks"), "T_bricks_H.png");
    }

    #[test]
    fn preset_name_round_trips() {
        for name in Preset::NAMES {
            assert!(Preset::from_name(name).is_some(), "{name:?} should parse");
        }
    }
}
