//! Shared mesh/texture byte-format conversion — the single source of the cooked formats.
//!
//! Pure (no filesystem, no glTF/image deps): the vertex assembly, tangent generation, matrix math
//! and `RMSH`/`RTEX` writers used by both the build-time cooker (`GlbToCookedMesh`) and the
//! engine's runtime glTF loader.

use bytemuck::{Pod, Zeroable};

/// Cooked vertex layout. Must stay bit-for-bit identical to `pill_engine::resources::MeshVertex`
/// (the engine casts `RMSH` bytes straight to that type).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CookedVertex {
    pub position: [f32; 3],
    pub texture_coordinates: [f32; 2],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
}

pub type Mat4 = [[f32; 4]; 4];

pub const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Builds cooked vertices for one primitive, baking `world` into positions/normals/tangents.
/// If `tangents` is empty, tangents/bitangents are derived from positions+UVs.
pub fn build_vertices(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    tangents: &[[f32; 4]],
    indices: &[u32],
    world: Mat4,
) -> Vec<CookedVertex> {
    let mut vertices: Vec<CookedVertex> = (0..positions.len())
        .map(|i| {
            let n = transform_normal(world, normals[i]);
            let (tx, ty, tz, sign) = tangents
                .get(i)
                .map(|t| (t[0], t[1], t[2], t[3]))
                .unwrap_or((1.0, 0.0, 0.0, 1.0));
            let t_world = transform_normal(world, [tx, ty, tz]);
            let bx = (n[1] * t_world[2] - n[2] * t_world[1]) * sign;
            let by = (n[2] * t_world[0] - n[0] * t_world[2]) * sign;
            let bz = (n[0] * t_world[1] - n[1] * t_world[0]) * sign;
            CookedVertex {
                position: transform_point(world, positions[i]),
                texture_coordinates: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                normal: n,
                tangent: t_world,
                bitangent: [bx, by, bz],
            }
        })
        .collect();

    if tangents.is_empty() {
        compute_tangents(&mut vertices, indices);
    }
    vertices
}

/// RMSH v1: `b"RMSH"` + u32 version=1 + u32 vertex_count + u32 index_count + vertices + indices.
pub fn rmsh_bytes(vertices: &[CookedVertex], indices: &[u32]) -> Vec<u8> {
    let vertex_bytes: &[u8] = bytemuck::cast_slice(vertices);
    let index_bytes: &[u8] = bytemuck::cast_slice(indices);
    let mut out = Vec::with_capacity(16 + vertex_bytes.len() + index_bytes.len());
    out.extend_from_slice(b"RMSH");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(vertices.len() as u32).to_le_bytes());
    out.extend_from_slice(&(indices.len() as u32).to_le_bytes());
    out.extend_from_slice(vertex_bytes);
    out.extend_from_slice(index_bytes);
    out
}

/// RTEX v1: `b"RTEX"` + u32 version=1 + u32 width + u32 height + raw RGBA8.
pub fn rtex_bytes(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + rgba.len());
    out.extend_from_slice(b"RTEX");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&w.to_le_bytes());
    out.extend_from_slice(&h.to_le_bytes());
    out.extend_from_slice(rgba);
    out
}

// --- Tangent generation (Lengyel) ---

pub fn compute_tangents(vertices: &mut [CookedVertex], indices: &[u32]) {
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

// --- Vector / matrix math ---

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

pub fn mat4_mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut c = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            c[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

/// Builds a row-major Mat4 from glTF's column-major 4×4 (as returned by `Transform::matrix()`).
pub fn mat4_from_cols(m: [[f32; 4]; 4]) -> Mat4 {
    [
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ]
}

pub fn transform_point(m: Mat4, p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
        m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
        m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3],
    ]
}

pub fn transform_normal(m: Mat4, n: [f32; 3]) -> [f32; 3] {
    // Normals transform by the inverse-transpose of the upper-left 3x3.
    // For uniform or orthogonal scaling this equals the 3x3 itself (re-normalized).
    let r = [
        m[0][0] * n[0] + m[0][1] * n[1] + m[0][2] * n[2],
        m[1][0] * n[0] + m[1][1] * n[1] + m[1][2] * n[2],
        m[2][0] * n[0] + m[2][1] * n[1] + m[2][2] * n[2],
    ];
    normalize3(r)
}
