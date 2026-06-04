//! Runtime glTF loader (feature `gltf_loading`).
//!
//! The single wasm-safe path for loading a glTF/glb obtained as bytes at runtime, however you got
//! them: fetched (multi-file `.gltf` + external `.bin`/textures) or dropped/embedded (self-contained
//! `.glb`). Opt-in: games that cook-and-embed at build time don't enable it and pay no binary cost.
//!
//! Parsing uses `goth-gltf` (nanoserde, no serde_json) to keep the wasm binary small; it's a
//! low-level reader, so attribute/index accessors are read out of the buffers here. The
//! vertex/tangent/matrix math and `RMSH`/`RTEX` writers are shared with the build-time cooker via
//! `pill_assets::convert`. Textures decode with the lean `zune-jpeg`/`zune-png` codecs. The caller
//! fetches external resources (see [`gltf_resource_uris`]) and uploads the result with
//! `Mesh::from_cooked_mesh_bytes` / `Texture::from_bytes`.

use goth_gltf::{ComponentType, NodeTransform};
use pill_assets::convert;
use pill_core::{PillError, Result};

use crate::resources::TextureType;

type Doc = goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>;

/// A texture ready to upload via `Texture::from_bytes(name, kind, &rtex)`.
pub struct GltfTextureData {
    pub name: String,
    pub kind: TextureType,
    pub rtex: Vec<u8>,
}

/// A material's factors and texture references (indices into [`GltfSceneData::textures`]).
pub struct GltfMaterialData {
    pub name: String,
    pub albedo: Option<usize>,
    pub normal: Option<usize>,
    pub metallic_roughness: Option<usize>,
    pub emissive: Option<usize>,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
}

/// A mesh primitive ready to upload via `Mesh::from_cooked_mesh_bytes(name, &rmsh)`.
pub struct GltfMeshData {
    pub name: String,
    pub rmsh: Vec<u8>,
    pub material: usize,
}

/// The whole scene, flattened: one mesh entry per primitive (world transform baked into vertices).
pub struct GltfSceneData {
    pub textures: Vec<GltfTextureData>,
    pub materials: Vec<GltfMaterialData>,
    pub meshes: Vec<GltfMeshData>,
}

/// External resources a glTF document references, in document (index) order. An entry is `None`
/// when nothing needs fetching: an embedded BIN buffer or a buffer-view image (both resolved from
/// the document). A self-contained `.glb` yields all-`None` — nothing to fetch.
pub struct GltfResourceUris {
    pub buffers: Vec<Option<String>>,
    pub images: Vec<Option<String>>,
}

/// Lists external buffer/image URIs to fetch, in document order. URIs are returned verbatim
/// (relative to the `.gltf`); the caller resolves them against its base URL.
pub fn gltf_resource_uris(gltf_bytes: &[u8]) -> Result<GltfResourceUris> {
    let (doc, _) =
        Doc::from_bytes(gltf_bytes).map_err(|e| PillError::from(format!("parse glTF: {e:?}")))?;
    Ok(GltfResourceUris {
        buffers: doc.buffers.iter().map(|b| b.uri.clone()).collect(),
        images: doc.images.iter().map(|i| i.uri.clone()).collect(),
    })
}

/// Parses a glTF/glb into uploadable scene data.
///
/// `external_buffers` / `external_images` are indexed 1:1 with the document's buffers/images (the
/// order from [`gltf_resource_uris`]); entries that were `None` there (embedded BIN, buffer-view
/// images) may be empty — they are resolved from the document. A self-contained `.glb` is parsed
/// with both slices empty.
pub fn parse_gltf_scene(
    gltf_bytes: &[u8],
    external_buffers: &[Vec<u8>],
    external_images: &[Vec<u8>],
) -> Result<GltfSceneData> {
    use std::collections::HashMap;

    let (doc, blob) =
        Doc::from_bytes(gltf_bytes).map_err(|e| PillError::from(format!("parse glTF: {e:?}")))?;

    // Resolve every buffer: embedded BIN blob, or caller-supplied external bytes.
    let mut buffers: Vec<&[u8]> = Vec::with_capacity(doc.buffers.len());
    for (i, buffer) in doc.buffers.iter().enumerate() {
        let bytes: &[u8] = match &buffer.uri {
            Some(_) => external_buffers
                .get(i)
                .map(|v| v.as_slice())
                .ok_or_else(|| PillError::from(format!("missing fetched bytes for buffer {i}")))?,
            None => blob.ok_or_else(|| {
                PillError::from("glTF has an embedded buffer but no blob".to_string())
            })?,
        };
        buffers.push(bytes);
    }

    // Decode every image to RGBA8, indexed by glTF image index.
    let mut decoded: Vec<DecodedImage> = Vec::with_capacity(doc.images.len());
    for (i, image) in doc.images.iter().enumerate() {
        let encoded: &[u8] = if image.uri.is_some() {
            external_images
                .get(i)
                .map(|v| v.as_slice())
                .filter(|v| !v.is_empty())
                .ok_or_else(|| PillError::from(format!("missing fetched bytes for image {i}")))?
        } else if let Some(bv) = image.buffer_view {
            let view = &doc.buffer_views[bv];
            &buffers[view.buffer][view.byte_offset..view.byte_offset + view.byte_length]
        } else {
            return Err(PillError::from(format!(
                "image {i} has neither uri nor buffer view"
            )));
        };
        decoded.push(decode_image(encoded)?);
    }

    // Textures, deduped by (image index, type): the same image bound as color (sRGB) vs normal
    // (linear) is a distinct GPU texture.
    let mut textures: Vec<GltfTextureData> = Vec::new();
    let mut tex_cache: HashMap<(usize, TextureType), usize> = HashMap::new();
    let image_of = |tex_index: usize| doc.textures[tex_index].source;

    let mut materials: Vec<GltfMaterialData> = Vec::with_capacity(doc.materials.len() + 1);
    for (idx, material) in doc.materials.iter().enumerate() {
        let pbr = &material.pbr_metallic_roughness;
        let albedo = pbr
            .base_color_texture
            .as_ref()
            .and_then(|t| image_of(t.index))
            .map(|img| {
                intern_tex(
                    &mut textures,
                    &mut tex_cache,
                    &decoded,
                    img,
                    TextureType::Color,
                )
            });
        let normal = material
            .normal_texture
            .as_ref()
            .and_then(|t| image_of(t.index))
            .map(|img| {
                intern_tex(
                    &mut textures,
                    &mut tex_cache,
                    &decoded,
                    img,
                    TextureType::Normal,
                )
            });
        let metallic_roughness = pbr
            .metallic_roughness_texture
            .as_ref()
            .and_then(|t| image_of(t.index))
            .map(|img| {
                intern_tex(
                    &mut textures,
                    &mut tex_cache,
                    &decoded,
                    img,
                    TextureType::MetallicRoughness,
                )
            });
        let emissive = material
            .emissive_texture
            .as_ref()
            .and_then(|t| image_of(t.index))
            .map(|img| {
                intern_tex(
                    &mut textures,
                    &mut tex_cache,
                    &decoded,
                    img,
                    TextureType::Emissive,
                )
            });

        materials.push(GltfMaterialData {
            name: format!("gltf_mat_{idx}"),
            albedo,
            normal,
            metallic_roughness,
            emissive,
            base_color: pbr.base_color_factor,
            metallic: pbr.metallic_factor,
            roughness: pbr.roughness_factor,
        });
    }

    // Fallback material for primitives that reference no material.
    let default_material = materials.len();
    materials.push(GltfMaterialData {
        name: "gltf_mat_default".to_string(),
        albedo: None,
        normal: None,
        metallic_roughness: None,
        emissive: None,
        base_color: [1.0; 4],
        metallic: 0.0,
        roughness: 0.5,
    });

    // Meshes: walk the scene graph, baking each node's world transform into the vertices, one RMSH
    // blob per primitive.
    let mut meshes: Vec<GltfMeshData> = Vec::new();
    let scene = doc
        .scenes
        .get(doc.scene)
        .or_else(|| doc.scenes.first())
        .ok_or_else(|| PillError::from("glTF has no scene".to_string()))?;
    let mut stack: Vec<(usize, convert::Mat4)> = scene
        .nodes
        .iter()
        .map(|&n| (n, convert::IDENTITY))
        .collect();
    while let Some((node_idx, parent)) = stack.pop() {
        let node = &doc.nodes[node_idx];
        let world = convert::mat4_mul(parent, node_transform(node));
        if let Some(mesh_idx) = node.mesh {
            for (prim_idx, prim) in doc.meshes[mesh_idx].primitives.iter().enumerate() {
                let positions = to_vec3(read_floats(&doc, &buffers, prim.attributes.position)?);
                let normals = to_vec3(read_floats(&doc, &buffers, prim.attributes.normal)?);
                let uvs = to_vec2(read_floats(&doc, &buffers, prim.attributes.texcoord_0)?);
                let tangents = match prim.attributes.tangent {
                    Some(acc) => to_vec4(read_floats_at(&doc, &buffers, acc)?),
                    None => Vec::new(),
                };
                let indices = read_indices(&doc, &buffers, prim.indices)?;

                let vertices =
                    convert::build_vertices(&positions, &normals, &uvs, &tangents, &indices, world);
                meshes.push(GltfMeshData {
                    name: format!("gltf_mesh_{node_idx}_{prim_idx}"),
                    rmsh: convert::rmsh_bytes(&vertices, &indices),
                    material: prim.material.unwrap_or(default_material),
                });
            }
        }
        for &child in &node.children {
            stack.push((child, world));
        }
    }

    Ok(GltfSceneData {
        textures,
        materials,
        meshes,
    })
}

// --- Accessor reading (goth-gltf is low-level; we read the typed arrays out of the buffers) ---

/// Reads a required float-attribute accessor (POSITION/NORMAL/TEXCOORD/...) as a flat `Vec<f32>`.
fn read_floats(doc: &Doc, buffers: &[&[u8]], accessor: Option<usize>) -> Result<Vec<f32>> {
    let accessor = accessor
        .ok_or_else(|| PillError::from("primitive missing a required attribute".to_string()))?;
    read_floats_at(doc, buffers, accessor)
}

fn read_floats_at(doc: &Doc, buffers: &[&[u8]], accessor_idx: usize) -> Result<Vec<f32>> {
    let acc = &doc.accessors[accessor_idx];
    let components = acc.accessor_type.num_components();
    let (buf, base, stride, comp_size) = accessor_layout(doc, buffers, accessor_idx)?;
    let mut out = Vec::with_capacity(acc.count * components);
    for i in 0..acc.count {
        let elem = base + i * stride;
        for c in 0..components {
            let o = elem + c * comp_size;
            let value = match acc.component_type {
                ComponentType::Float => f32::from_le_bytes(buf[o..o + 4].try_into().unwrap()),
                ComponentType::UnsignedByte => norm_u(buf[o] as f32, 255.0, acc.normalized),
                ComponentType::UnsignedShort => norm_u(
                    u16::from_le_bytes(buf[o..o + 2].try_into().unwrap()) as f32,
                    65535.0,
                    acc.normalized,
                ),
                ComponentType::Byte => norm_s(buf[o] as i8 as f32, 127.0, acc.normalized),
                ComponentType::Short => norm_s(
                    i16::from_le_bytes(buf[o..o + 2].try_into().unwrap()) as f32,
                    32767.0,
                    acc.normalized,
                ),
                ComponentType::UnsignedInt => {
                    u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()) as f32
                }
            };
            out.push(value);
        }
    }
    Ok(out)
}

/// Reads an index accessor as `u32` (glTF allows u8/u16/u32 indices).
fn read_indices(doc: &Doc, buffers: &[&[u8]], accessor: Option<usize>) -> Result<Vec<u32>> {
    let accessor_idx =
        accessor.ok_or_else(|| PillError::from("primitive missing indices".to_string()))?;
    let acc = &doc.accessors[accessor_idx];
    let (buf, base, stride, _) = accessor_layout(doc, buffers, accessor_idx)?;
    let mut out = Vec::with_capacity(acc.count);
    for i in 0..acc.count {
        let o = base + i * stride;
        let value = match acc.component_type {
            ComponentType::UnsignedInt => u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()),
            ComponentType::UnsignedShort => {
                u16::from_le_bytes(buf[o..o + 2].try_into().unwrap()) as u32
            }
            ComponentType::UnsignedByte => buf[o] as u32,
            other => {
                return Err(PillError::from(format!(
                    "unsupported index component type {other:?}"
                )))
            }
        };
        out.push(value);
    }
    Ok(out)
}

/// Resolves (buffer slice, byte base offset, element stride, component byte size) for an accessor.
fn accessor_layout<'a>(
    doc: &Doc,
    buffers: &[&'a [u8]],
    accessor_idx: usize,
) -> Result<(&'a [u8], usize, usize, usize)> {
    let acc = &doc.accessors[accessor_idx];
    let bv = acc
        .buffer_view
        .ok_or_else(|| PillError::from("accessor without a buffer view".to_string()))?;
    let view = &doc.buffer_views[bv];
    let buf = *buffers
        .get(view.buffer)
        .ok_or_else(|| PillError::from("accessor references missing buffer".to_string()))?;
    let comp_size = acc.component_type.byte_size();
    let element = acc.accessor_type.num_components() * comp_size;
    let stride = view.byte_stride.unwrap_or(element);
    Ok((buf, view.byte_offset + acc.byte_offset, stride, comp_size))
}

fn norm_u(v: f32, max: f32, normalized: bool) -> f32 {
    if normalized {
        v / max
    } else {
        v
    }
}

fn norm_s(v: f32, max: f32, normalized: bool) -> f32 {
    if normalized {
        // glTF signed-normalized dequant: max(c / MAX, -1.0). The divisor is the type's positive
        // max, so values can't exceed +1.0 — only the low end needs the clamp (spec 3.6.2.1.2).
        (v / max).max(-1.0)
    } else {
        v
    }
}

fn to_vec3(flat: Vec<f32>) -> Vec<[f32; 3]> {
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}
fn to_vec2(flat: Vec<f32>) -> Vec<[f32; 2]> {
    flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
}
fn to_vec4(flat: Vec<f32>) -> Vec<[f32; 4]> {
    flat.chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}

/// Node local transform → row-major Mat4 (matching `pill_assets::convert`).
fn node_transform(
    node: &goth_gltf::Node<goth_gltf::default_extensions::Extensions>,
) -> convert::Mat4 {
    match node.transform() {
        NodeTransform::Matrix(m) => {
            // glTF matrices are column-major; group into columns for mat4_from_cols.
            convert::mat4_from_cols([
                [m[0], m[1], m[2], m[3]],
                [m[4], m[5], m[6], m[7]],
                [m[8], m[9], m[10], m[11]],
                [m[12], m[13], m[14], m[15]],
            ])
        }
        NodeTransform::Set {
            translation: [tx, ty, tz],
            rotation: [x, y, z, w],
            scale: [sx, sy, sz],
        } => {
            // T * R(quat) * S, row-major.
            let r00 = 1.0 - 2.0 * (y * y + z * z);
            let r01 = 2.0 * (x * y - z * w);
            let r02 = 2.0 * (x * z + y * w);
            let r10 = 2.0 * (x * y + z * w);
            let r11 = 1.0 - 2.0 * (x * x + z * z);
            let r12 = 2.0 * (y * z - x * w);
            let r20 = 2.0 * (x * z - y * w);
            let r21 = 2.0 * (y * z + x * w);
            let r22 = 1.0 - 2.0 * (x * x + y * y);
            [
                [r00 * sx, r01 * sy, r02 * sz, tx],
                [r10 * sx, r11 * sy, r12 * sz, ty],
                [r20 * sx, r21 * sy, r22 * sz, tz],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }
    }
}

// --- Texture decode (lean zune codecs) ---

struct DecodedImage {
    rgba: Vec<u8>,
    w: u32,
    h: u32,
}

fn decode_image(bytes: &[u8]) -> Result<DecodedImage> {
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        decode_jpeg(bytes)
    } else if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        decode_png(bytes)
    } else {
        Err(PillError::from(
            "unsupported texture image (expected JPEG or PNG)".to_string(),
        ))
    }
}

fn decode_jpeg(bytes: &[u8]) -> Result<DecodedImage> {
    let mut decoder = zune_jpeg::JpegDecoder::new(bytes);
    let pixels = decoder
        .decode()
        .map_err(|e| PillError::from(format!("decode JPEG: {e:?}")))?;
    let (w, h) = decoder
        .dimensions()
        .ok_or_else(|| PillError::from("JPEG has no dimensions".to_string()))?;
    let components = decoder
        .get_output_colorspace()
        .map(|c| c.num_components())
        .unwrap_or(3);
    Ok(DecodedImage {
        rgba: expand_to_rgba(pixels, components),
        w: w as u32,
        h: h as u32,
    })
}

fn decode_png(bytes: &[u8]) -> Result<DecodedImage> {
    use zune_png::zune_core::result::DecodingResult;
    let mut decoder = zune_png::PngDecoder::new(bytes);
    let result = decoder
        .decode()
        .map_err(|e| PillError::from(format!("decode PNG: {e:?}")))?;
    let (w, h) = decoder
        .get_dimensions()
        .ok_or_else(|| PillError::from("PNG has no dimensions".to_string()))?;
    let components = decoder
        .get_colorspace()
        .map(|c| c.num_components())
        .unwrap_or(4);
    let px8 = match result {
        DecodingResult::U8(v) => v,
        _ => return Err(PillError::from("16-bit PNG not supported".to_string())),
    };
    Ok(DecodedImage {
        rgba: expand_to_rgba(px8, components),
        w: w as u32,
        h: h as u32,
    })
}

/// Expands decoded pixels (luma / luma-alpha / RGB / RGBA) to RGBA8. Takes ownership so the
/// already-RGBA case (the common one) is a move, not a full-image copy.
fn expand_to_rgba(pixels: Vec<u8>, components: usize) -> Vec<u8> {
    match components {
        3 => pixels
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        2 => pixels
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        1 => pixels.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        _ => pixels, // already RGBA (4) or unknown — pass through
    }
}

fn intern_tex(
    textures: &mut Vec<GltfTextureData>,
    cache: &mut std::collections::HashMap<(usize, TextureType), usize>,
    images: &[DecodedImage],
    image_idx: usize,
    kind: TextureType,
) -> usize {
    if let Some(&existing) = cache.get(&(image_idx, kind)) {
        return existing;
    }
    let role = match kind {
        TextureType::Color => "albedo",
        TextureType::Normal => "normal",
        TextureType::MetallicRoughness => "metallic_roughness",
        TextureType::Emissive => "emissive",
    };
    let di = &images[image_idx];
    let idx = textures.len();
    textures.push(GltfTextureData {
        name: format!("gltf_{role}_{image_idx}"),
        kind,
        rtex: convert::rtex_bytes(di.w, di.h, &di.rgba),
    });
    cache.insert((image_idx, kind), idx);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reads back the position + texcoord of vertex `i` from an RMSH blob.
    // RMSH = b"RMSH" + u32 ver + u32 vcount + u32 icount + CookedVertex[]; CookedVertex is
    // position[3] + texture_coordinates[2] + normal[3] + tangent[3] + bitangent[3] = 14 f32.
    fn vertex(rmsh: &[u8], i: usize) -> ([f32; 3], [f32; 2]) {
        let base = 16 + i * 14 * 4;
        let f =
            |o: usize| f32::from_le_bytes(rmsh[base + o * 4..base + o * 4 + 4].try_into().unwrap());
        ([f(0), f(1), f(2)], [f(3), f(4)])
    }

    // Exercises the paths Sponza never hits: a node `matrix` transform, u32 indices, and a
    // normalized-u8 TEXCOORD. Expected values are computed by hand.
    #[test]
    fn parses_matrix_node_u32_indices_normalized_uv() {
        // Buffer: positions(3×vec3 f32) | normals(3×vec3 f32) | uv(3×vec2 u8) | pad | indices(3×u32)
        let mut buf = Vec::new();
        for p in [[1f32, 0., 0.], [0., 1., 0.], [0., 0., 1.]] {
            for v in p {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        for _ in 0..3 {
            for v in [0f32, 0., 1.] {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf.extend_from_slice(&[0, 0, 255, 0, 0, 255]); // uv u8: (0,0) (255,0) (0,255)
        buf.extend_from_slice(&[0, 0]); // pad to 4-byte alignment for u32 indices
        for i in [0u32, 1, 2] {
            buf.extend_from_slice(&i.to_le_bytes());
        }
        assert_eq!(buf.len(), 92);

        // Node matrix (column-major): scale 2 + translate (10,20,30).
        let json = r#"{
          "asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],
          "nodes":[{"mesh":0,"matrix":[2,0,0,0, 0,2,0,0, 0,0,2,0, 10,20,30,1]}],
          "meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},"indices":3}]}],
          "buffers":[{"uri":"b.bin","byteLength":92}],
          "bufferViews":[
            {"buffer":0,"byteOffset":0,"byteLength":36},
            {"buffer":0,"byteOffset":36,"byteLength":36},
            {"buffer":0,"byteOffset":72,"byteLength":6},
            {"buffer":0,"byteOffset":80,"byteLength":12}],
          "accessors":[
            {"bufferView":0,"componentType":5126,"type":"VEC3","count":3},
            {"bufferView":1,"componentType":5126,"type":"VEC3","count":3},
            {"bufferView":2,"componentType":5121,"type":"VEC2","count":3,"normalized":true},
            {"bufferView":3,"componentType":5125,"type":"SCALAR","count":3}],
          "materials":[],"textures":[],"images":[]
        }"#;

        let scene = parse_gltf_scene(json.as_bytes(), &[buf], &[]).expect("parse");
        assert_eq!(scene.meshes.len(), 1);
        let rmsh = &scene.meshes[0].rmsh;
        assert_eq!(&rmsh[0..4], b"RMSH");
        assert_eq!(u32::from_le_bytes(rmsh[8..12].try_into().unwrap()), 3); // vertex count
        assert_eq!(u32::from_le_bytes(rmsh[12..16].try_into().unwrap()), 3); // index count

        let approx = |a: [f32; 3], b: [f32; 3]| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-4);
        // matrix: world = scale 2 then translate (10,20,30)
        let (p0, uv0) = vertex(rmsh, 0);
        let (p1, uv1) = vertex(rmsh, 1);
        let (p2, uv2) = vertex(rmsh, 2);
        assert!(approx(p0, [12.0, 20.0, 30.0]), "p0 = {p0:?}");
        assert!(approx(p1, [10.0, 22.0, 30.0]), "p1 = {p1:?}");
        assert!(approx(p2, [10.0, 20.0, 32.0]), "p2 = {p2:?}");
        // normalized u8 → [0,1]
        assert!(
            (uv0[0]).abs() < 1e-4 && (uv0[1]).abs() < 1e-4,
            "uv0 = {uv0:?}"
        );
        assert!(
            (uv1[0] - 1.0).abs() < 1e-4 && uv1[1].abs() < 1e-4,
            "uv1 = {uv1:?}"
        );
        assert!(
            uv2[0].abs() < 1e-4 && (uv2[1] - 1.0).abs() < 1e-4,
            "uv2 = {uv2:?}"
        );

        // indices preserved (u32 path)
        let icount = 3usize;
        let ibase = 16 + 3 * 14 * 4;
        let indices: Vec<u32> = (0..icount)
            .map(|i| u32::from_le_bytes(rmsh[ibase + i * 4..ibase + i * 4 + 4].try_into().unwrap()))
            .collect();
        assert_eq!(indices, vec![0, 1, 2]);
    }
}
