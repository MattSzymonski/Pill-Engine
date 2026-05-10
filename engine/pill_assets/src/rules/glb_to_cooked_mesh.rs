use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::Rule;

/// Must stay bit-for-bit identical to `pill_engine::resources::MeshVertex`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    texture_coordinates: [f32; 2],
    normal: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
}

/// GLB → cooked_mesh + sidecar textures.
///
/// Primary output: `{stem}.cooked_mesh` (RMSH format, same as ObjToCookedMesh).
/// Side outputs written in the same directory when textures are present:
///   `{stem}_albedo.cooked_tex`             — base color (RTEX RGBA8)
///   `{stem}_normal.cooked_tex`             — normal map (RTEX RGBA8)
///   `{stem}_metallic_roughness.cooked_tex` — G=roughness, B=metallic (RTEX RGBA8)
///   `{stem}_emissive.cooked_tex`           — emissive color (RTEX RGBA8)
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

        let mut all_vertices: Vec<Vertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();

        // Collect (mesh_index, world_transform_4x4) pairs by traversing the scene graph.
        let mut mesh_instances: Vec<(usize, [[f32; 4]; 4])> = Vec::new();
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0f32],
        ];
        let default_scene = doc.default_scene().or_else(|| doc.scenes().next());
        if let Some(scene) = default_scene {
            let mut stack: Vec<(gltf::Node, [[f32; 4]; 4])> =
                scene.nodes().map(|n| (n, identity)).collect();
            while let Some((node, parent_transform)) = stack.pop() {
                let local = node_transform(node.transform());
                let world = mat4_mul(parent_transform, local);
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
                mesh_instances.push((i, identity));
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
                let tangents_glb: Vec<[f32; 4]> = reader
                    .read_tangents()
                    .map(|tangents_iter| tangents_iter.collect())
                    .unwrap_or_default();
                let prim_indices: Vec<u32> = reader
                    .read_indices()
                    .with_context(|| format!("{input:?}: missing indices"))?
                    .into_u32()
                    .collect();

                let mut vertices: Vec<Vertex> = (0..positions.len())
                    .map(|i| {
                        let n = transform_normal(world, normals[i]);
                        let (tx, ty, tz, sign) = tangents_glb
                            .get(i)
                            .map(|tangent| (tangent[0], tangent[1], tangent[2], tangent[3]))
                            .unwrap_or((1.0, 0.0, 0.0, 1.0));
                        let t_world = transform_normal(world, [tx, ty, tz]);
                        let bx = (n[1] * t_world[2] - n[2] * t_world[1]) * sign;
                        let by = (n[2] * t_world[0] - n[0] * t_world[2]) * sign;
                        let bz = (n[0] * t_world[1] - n[1] * t_world[0]) * sign;
                        Vertex {
                            position: transform_point(world, positions[i]),
                            texture_coordinates: uvs[i],
                            normal: n,
                            tangent: t_world,
                            bitangent: [bx, by, bz],
                        }
                    })
                    .collect();

                if tangents_glb.is_empty() {
                    compute_tangents(&mut vertices, &prim_indices);
                }

                all_indices.extend(prim_indices.iter().map(|&i| i + vertex_offset));
                all_vertices.extend(vertices);
            }
        }

        let vertex_bytes: &[u8] = bytemuck::cast_slice(&all_vertices);
        let index_bytes: &[u8] = bytemuck::cast_slice(&all_indices);
        let mut out = Vec::with_capacity(16 + vertex_bytes.len() + index_bytes.len());
        out.extend_from_slice(b"RMSH");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(all_vertices.len() as u32).to_le_bytes());
        out.extend_from_slice(&(all_indices.len() as u32).to_le_bytes());
        out.extend_from_slice(vertex_bytes);
        out.extend_from_slice(index_bytes);
        std::fs::write(output, &out).with_context(|| format!("write {output:?}"))?;

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

fn compute_tangents(vertices: &mut [Vertex], indices: &[u32]) {
    let mut triangle_counts = vec![0usize; vertices.len()];

    for tri in indices.chunks(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = vertices[i0].position;
        let p1 = vertices[i1].position;
        let p2 = vertices[i2].position;
        let uv0 = vertices[i0].texture_coordinates;
        let uv1 = vertices[i1].texture_coordinates;
        let uv2 = vertices[i2].texture_coordinates;

        let dp1 = sub3(p1, p0);
        let dp2 = sub3(p2, p0);
        let duv1 = sub2(uv1, uv0);
        let duv2 = sub2(uv2, uv0);

        let det = duv1[0] * duv2[1] - duv1[1] * duv2[0];
        if det.abs() < 1e-8 {
            continue;
        }
        let inv = 1.0 / det;
        let tangent = scale3(sub3(scale3(dp1, duv2[1]), scale3(dp2, duv1[1])), inv);
        let bitangent = scale3(sub3(scale3(dp2, duv1[0]), scale3(dp1, duv2[0])), inv);

        for &i in &[i0, i1, i2] {
            vertices[i].tangent = add3(vertices[i].tangent, tangent);
            vertices[i].bitangent = add3(vertices[i].bitangent, bitangent);
            triangle_counts[i] += 1;
        }
    }

    for (i, &count) in triangle_counts.iter().enumerate() {
        if count > 0 {
            let inv = 1.0 / count as f32;
            vertices[i].tangent = normalize3(scale3(vertices[i].tangent, inv));
            vertices[i].bitangent = normalize3(scale3(vertices[i].bitangent, inv));
        }
    }
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn sub2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn normalize3(a: [f32; 3]) -> [f32; 3] {
    let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if len < 1e-10 {
        a
    } else {
        [a[0] / len, a[1] / len, a[2] / len]
    }
}

fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut c = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            c[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

fn node_transform(t: gltf::scene::Transform) -> [[f32; 4]; 4] {
    let m = t.matrix();
    // gltf gives column-major; convert to row-major [[f32;4];4]
    [
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ]
}

fn transform_point(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
        m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
        m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3],
    ]
}

fn transform_normal(m: [[f32; 4]; 4], n: [f32; 3]) -> [f32; 3] {
    // Normals transform by the inverse-transpose of the upper-left 3x3.
    // For uniform or orthogonal scaling this equals the 3x3 itself (re-normalized).
    let r = [
        m[0][0] * n[0] + m[0][1] * n[1] + m[0][2] * n[2],
        m[1][0] * n[0] + m[1][1] * n[1] + m[1][2] * n[2],
        m[2][0] * n[0] + m[2][1] * n[1] + m[2][2] * n[2],
    ];
    normalize3(r)
}
