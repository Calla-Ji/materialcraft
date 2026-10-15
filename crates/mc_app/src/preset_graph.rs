// SPDX-License-Identifier: MIT OR Apache-2.0
//! Builds the splash screen's starter graphs — one per engine preset, since
//! Blender and Unreal Engine 5 want genuinely different output shapes
//! (separate Roughness/Metallic/AO files vs. one packed ORM texture).
//! Every node starts unconnected (aside from Unreal Engine 5's own Channel
//! Packer → Output: ORM wire, a fixed pairing with no reason not to
//! pre-connect it) — a ready scaffold for an artist who wants to start
//! wiring up material channels immediately instead of adding each Output
//! node by hand from an empty canvas.

const COLUMN_X: f32 = 480.0;
/// Clears an Output node's own card height (CLAUDE.md/`canvas.rs`'s
/// `node_size`: ~220 at zoom 1, since every node reserves thumbnail space
/// regardless of its own port count) with headroom to spare, so the preset
/// graph never needs untangling before an artist can actually use it.
const ROW_SPACING: f32 = 260.0;
/// Clears a node card's own width (`canvas.rs`'s own `NODE_WIDTH`, 170) with
/// room for its wire — used only for the Channel Packer → Output: ORM pair,
/// the one place this file places two nodes side by side instead of
/// stacking everything in one column.
const PACKER_COLUMN_GAP: f32 = 220.0;
/// `ROW_SPACING`'s own 40px headroom only clears a *normal* (single
/// input/output row) node's ~220-tall card. Channel Packer has 3 input
/// rows instead of 1, so its own card is taller — about 260, following
/// `canvas.rs`'s own `node_size` formula — leaving zero gap before the row
/// below it if plain `ROW_SPACING` is used after it. This is that taller
/// card's own height plus the same 40px margin every other row gets.
const PACKER_ROW_SPACING: f32 = 300.0;

fn instantiate_at(kind: &str, position: [f32; 2]) -> mc_core::Node {
    let def = mc_nodes::defs::node_def(kind)
        .unwrap_or_else(|| panic!("{kind} must be a registered NodeDef"));
    let mut node = def.instantiate();
    node.position = position;
    node
}

/// Base Color first, then Normal/Height/Roughness/Metallic (the order the
/// splash screen's own brief asked for), then Ambient Occlusion/Emissive
/// (the two channels it didn't explicitly order) just above Alpha, which
/// stays last as the "final" mask-like channel.
const BLENDER_OUTPUT_KINDS: [&str; 8] = [
    "output_base_color",
    "output_normal",
    "output_height",
    "output_roughness",
    "output_metallic",
    "output_ambient_occlusion",
    "output_emissive",
    "output_alpha",
];

/// A fresh graph containing just the 8 Output nodes from
/// [`BLENDER_OUTPUT_KINDS`], stacked top to bottom in that order, evaluated
/// at `resolution`. None are connected — the artist wires up whatever
/// generates each channel.
pub fn build_blender(resolution: u32) -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = resolution;
    for (i, kind) in BLENDER_OUTPUT_KINDS.iter().enumerate() {
        graph.add_node(instantiate_at(kind, [COLUMN_X, i as f32 * ROW_SPACING]));
    }
    graph
}

/// A fresh graph for the Unreal Engine 5 export preset: Base Color, Normal,
/// a Channel Packer pre-wired into Output: ORM (its own R/G/B inputs left
/// open — connect whatever generates AO/Roughness/Metallic to them), then
/// Height, Emissive, Alpha. Matches `mc_nodes::export`'s own UE5 output
/// list exactly, so every node this graph starts with corresponds to a
/// real file that preset can write.
pub fn build_unreal_engine_5(resolution: u32) -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = resolution;

    graph.add_node(instantiate_at("output_base_color", [COLUMN_X, 0.0]));
    graph.add_node(instantiate_at("output_normal", [COLUMN_X, ROW_SPACING]));

    let packer_row_y = 2.0 * ROW_SPACING;
    let packer = graph.add_node(instantiate_at("channel_pack", [COLUMN_X, packer_row_y]));
    let orm_output = graph.add_node(instantiate_at(
        "output_orm",
        [COLUMN_X + PACKER_COLUMN_GAP, packer_row_y],
    ));
    graph
        .connect(packer, "output", orm_output, "input")
        .expect("Channel Packer's output and Output: ORM's input are both Color ports");

    let height_row_y = packer_row_y + PACKER_ROW_SPACING;
    graph.add_node(instantiate_at("output_height", [COLUMN_X, height_row_y]));
    let emissive_row_y = height_row_y + ROW_SPACING;
    graph.add_node(instantiate_at(
        "output_emissive",
        [COLUMN_X, emissive_row_y],
    ));
    let alpha_row_y = emissive_row_y + ROW_SPACING;
    graph.add_node(instantiate_at("output_alpha", [COLUMN_X, alpha_row_y]));

    graph
}

#[cfg(test)]
mod tests {
    use super::{build_blender, build_unreal_engine_5, BLENDER_OUTPUT_KINDS};

    #[test]
    fn blender_contains_exactly_the_eight_output_kinds_in_order() {
        let graph = build_blender(1024);
        let kinds: Vec<&str> = graph.nodes().map(|(_, node)| node.kind.as_str()).collect();
        assert_eq!(kinds, BLENDER_OUTPUT_KINDS);
    }

    #[test]
    fn blender_sets_the_requested_resolution() {
        let graph = build_blender(2048);
        assert_eq!(graph.settings.resolution, 2048);
    }

    #[test]
    fn blender_no_two_nodes_share_a_position() {
        let graph = build_blender(1024);
        let positions: Vec<[f32; 2]> = graph.nodes().map(|(_, node)| node.position).collect();
        for (i, a) in positions.iter().enumerate() {
            for b in &positions[i + 1..] {
                assert_ne!(a, b, "two Output nodes landed on top of each other");
            }
        }
    }

    #[test]
    fn unreal_engine_5_contains_the_right_node_kinds() {
        let graph = build_unreal_engine_5(1024);
        let mut kinds: Vec<&str> = graph.nodes().map(|(_, node)| node.kind.as_str()).collect();
        kinds.sort_unstable();
        let mut expected = vec![
            "output_base_color",
            "output_normal",
            "channel_pack",
            "output_orm",
            "output_height",
            "output_emissive",
            "output_alpha",
        ];
        expected.sort_unstable();
        assert_eq!(kinds, expected);
    }

    #[test]
    fn unreal_engine_5_pre_wires_the_channel_packer_into_output_orm() {
        let graph = build_unreal_engine_5(1024);
        let packer = graph
            .nodes()
            .find(|(_, n)| n.kind == "channel_pack")
            .map(|(id, _)| id)
            .expect("Channel Packer node exists");
        let orm_output = graph
            .nodes()
            .find(|(_, n)| n.kind == "output_orm")
            .map(|(id, _)| id)
            .expect("Output: ORM node exists");
        let (upstream_id, upstream_port) = graph
            .upstream_from(orm_output, "input")
            .expect("Output: ORM's input is connected");
        assert_eq!(upstream_id, packer);
        assert_eq!(upstream_port, "output");
    }

    #[test]
    fn unreal_engine_5_sets_the_requested_resolution() {
        let graph = build_unreal_engine_5(2048);
        assert_eq!(graph.settings.resolution, 2048);
    }

    #[test]
    fn unreal_engine_5_no_two_nodes_share_a_position() {
        let graph = build_unreal_engine_5(1024);
        let positions: Vec<[f32; 2]> = graph.nodes().map(|(_, node)| node.position).collect();
        for (i, a) in positions.iter().enumerate() {
            for b in &positions[i + 1..] {
                assert_ne!(a, b, "two nodes landed on top of each other");
            }
        }
    }
}
