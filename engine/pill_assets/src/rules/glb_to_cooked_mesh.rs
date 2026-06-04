use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::convert::{self, CookedVertex};
use crate::Rule;

/// GLB → cooked_mesh + sidecar textures.
///
/// Primary output: `{stem}.cooked_mesh` (RMSH format, same as ObjToCookedMesh).
/// Side outputs written in the same directory when textures are present:
///   `{stem}_albedo.cooked_tex`             — base color (RTEX RGBA8)
///   `{stem}_normal.cooked_tex`             — normal map (RTEX RGBA8)
///   `{stem}_metallic_roughness.cooked_tex` — G=roughness, B=metallic (RTEX RGBA8)
///   `{stem}_emissive.cooked_tex`           — emissive color (RTEX RGBA8)
///
/// Merges all primitives into one mesh — note the runtime loader keeps them per-primitive for
/// multi-material scenes. Vertex/tangent/RMSH conversion is shared via [`crate::convert`].
pub struct GlbToCookedMesh;

impl Rule for GlbToCookedMesh {
    fn name(&self) -> &'static str {
        "glb_to_cooked_mesh"
    }

    fn input_glob(&self) -> &'static str {
        "**/*.glb"
    }

    fn output_for(&self, input: &Path) -> PathBuf {
        input.with_extension("cooked_mesh")
    }

    fn build(&self, input: &Path, output: &Path) -> Result<()> {
        let bytes = std::fs::read(input).with_context(|| format!("read {input:?}"))?;
        let (doc, buffers, images) =
            gltf::import_slice(&bytes).with_context(|| format!("parse GLB {input:?}"))?;

        // --- Mesh ---
        // Walk the scene node hierarchy. For each node that references a mesh, apply the
        // node's world transform to vertex positions and normals before cooked output.
        // This bakes the root-node transform (scale, rotation, etc.) into the geometry so
        // game code needs no compensating transform on the mesh entity.

        let mut all_vertices: Vec<CookedVertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();

        // Collect (mesh_index, world_transform_4x4) pairs by traversing the scene graph.
        let mut mesh_instances: Vec<(usize, convert::Mat4)> = Vec::new();
        let default_scene = doc.default_scene().or_else(|| doc.scenes().next());
        if let Some(scene) = default_scene {
            let mut stack: Vec<(gltf::Node, convert::Mat4)> =
                scene.nodes().map(|n| (n, convert::IDENTITY)).collect();
            while let Some((node, parent_transform)) = stack.pop() {
                let local = convert::mat4_from_cols(node.transform().matrix());
                let world = convert::mat4_mul(parent_transform, local);
                if let Some(mesh) = node.mesh() {
                    mesh_instances.push((mesh.index(), world));
                }
                for child in node.children() {
                    stack.push((child, world));
                }
            }
        } else {
            // No scene: fall back to processing every mesh with identity transform.
            for (i, _) in doc.meshes().enumerate() {
                mesh_instances.push((i, convert::IDENTITY));
            }
        }

        if mesh_instances.is_empty() {
            anyhow::bail!("{input:?}: no meshes in GLB");
        }

        let meshes: Vec<gltf::Mesh> = doc.meshes().collect();
        for (mesh_idx, world) in mesh_instances {
            let mesh = &meshes[mesh_idx];
            for prim in mesh.primitives() {
                let reader = prim.reader(|buffer| Some(&*buffers[buffer.index()]));
                let vertex_offset = all_vertices.len() as u32;

                let positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .with_context(|| format!("{input:?}: missing positions"))?
                    .collect();
                let normals: Vec<[f32; 3]> = reader
                    .read_normals()
                    .with_context(|| format!("{input:?}: missing normals"))?
                    .collect();
                let uvs: Vec<[f32; 2]> = reader
                    .read_tex_coords(0)
                    .with_context(|| format!("{input:?}: missing UV set 0"))?
                    .into_f32()
                    .collect();
                let tangents: Vec<[f32; 4]> = reader
                    .read_tangents()
                    .map(|tangents_iter| tangents_iter.collect())
                    .unwrap_or_default();
                let prim_indices: Vec<u32> = reader
                    .read_indices()
                    .with_context(|| format!("{input:?}: missing indices"))?
                    .into_u32()
                    .collect();

                let vertices = convert::build_vertices(
                    &positions,
                    &normals,
                    &uvs,
                    &tangents,
                    &prim_indices,
                    world,
                );

                all_indices.extend(prim_indices.iter().map(|&i| i + vertex_offset));
                all_vertices.extend(vertices);
            }
        }

        std::fs::write(output, convert::rmsh_bytes(&all_vertices, &all_indices))
            .with_context(|| format!("write {output:?}"))?;

        // --- Textures (side outputs) ---

        let stem = input
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let dir = output
            .parent()
            .with_context(|| "output has no parent directory")?;

        for mat in doc.materials() {
            let mat_idx = mat.index();
            let suffix = |base: &str| -> String {
                match mat_idx {
                    None | Some(0) => format!("{stem}_{base}.cooked_tex"),
                    Some(i) => format!("{stem}_mat{i}_{base}.cooked_tex"),
                }
            };
            let pbr = mat.pbr_metallic_roughness();

            if let Some(info) = pbr.base_color_texture() {
                write_cooked_tex(
                    &images,
                    info.texture().source().index(),
                    dir.join(suffix("albedo")),
                    "base color texture",
                    input,
                )?;
            }
            if let Some(info) = mat.normal_texture() {
                write_cooked_tex(
                    &images,
                    info.texture().source().index(),
                    dir.join(suffix("normal")),
                    "normal texture",
                    input,
                )?;
            }
            if let Some(info) = pbr.metallic_roughness_texture() {
                write_cooked_tex(
                    &images,
                    info.texture().source().index(),
                    dir.join(suffix("metallic_roughness")),
                    "metallic_roughness texture",
                    input,
                )?;
            }
            if let Some(info) = mat.emissive_texture() {
                write_cooked_tex(
                    &images,
                    info.texture().source().index(),
                    dir.join(suffix("emissive")),
                    "emissive texture",
                    input,
                )?;
            }
        }

        Ok(())
    }
}

fn write_cooked_tex(
    images: &[gltf::image::Data],
    img_idx: usize,
    path: std::path::PathBuf,
    label: &str,
    input: &Path,
) -> Result<()> {
    use gltf::image::Format;
    use std::borrow::Cow;
    let Some(img) = images.get(img_idx) else {
        return Ok(());
    };
    let rgba: Cow<[u8]> = match img.format {
        Format::R8G8B8A8 => Cow::Borrowed(&img.pixels),
        Format::R8G8B8 => Cow::Owned(
            img.pixels
                .chunks_exact(3)
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255u8])
                .collect(),
        ),
        format => bail!("{input:?}: {label}: unsupported GLB image format: {format:?}"),
    };
    // glTF UV origin (0,0) is top-left, matching wgpu/Vulkan — no row flip needed.
    super::studio_equirect::write_rtex(&path, img.width, img.height, &rgba)
        .with_context(|| format!("{input:?}: {label}"))
}
