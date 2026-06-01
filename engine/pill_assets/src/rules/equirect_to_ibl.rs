use std::f32::consts::PI;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use super::studio_equirect::{linear_to_srgb, srgb_to_linear, write_rtex, write_rtex_hdr_mip};
use crate::Rule;

pub struct EquirectToIBL;

impl Rule for EquirectToIBL {
    fn name(&self) -> &'static str {
        "equirect_to_ibl"
    }

    fn input_glob(&self) -> &'static str {
        "**/*_equirect.cooked_tex"
    }

    fn output_for(&self, input: &Path) -> PathBuf {
        // Anchor output is the diffuse IBL; build() also writes specular IBL + BRDF LUT.
        let stem = input.file_stem().unwrap().to_str().unwrap();
        let base = stem.strip_suffix("_equirect").unwrap_or(stem);
        input.with_file_name(format!("{base}_diffuse_ibl.cooked_tex"))
    }

    fn build(&self, input: &Path, output: &Path) -> Result<()> {
        let equirect = Equirect::load(input)?;
        let dir = output.parent().unwrap();
        let stem = input.file_stem().unwrap().to_str().unwrap();
        let base = stem.strip_suffix("_equirect").unwrap_or(stem);

        // 1. Diffuse irradiance (32×16 sRGB) — anchor output
        let irr = compute_irradiance(&equirect, 32, 16);
        write_rtex(output, 32, 16, &irr)?;

        // 2. Specular prefilter HDR mip chain (RTEX v4, Rgba32Float).
        // 5 mip levels; mip i roughness = i/4, matching shader roughness * MAX_REFLECTION_LOD(4).
        const MIP_ROUGHNESS: [f32; 5] = [0.04, 0.25, 0.5, 0.75, 1.0];
        let spec_path = dir.join(format!("{base}_specular_ibl.cooked_tex"));
        let mut spec_mips: Vec<Vec<f32>> = Vec::with_capacity(5);
        for (mip, &r) in MIP_ROUGHNESS.iter().enumerate() {
            let w = (128u32 >> mip).max(1);
            let h = (64u32 >> mip).max(1);
            spec_mips.push(compute_specular_prefilter(&equirect, w, h, r));
        }
        write_rtex_hdr_mip(&spec_path, 128, 64, &spec_mips)?;

        // 3. BRDF LUT (256×256 linear) — split-sum preintegration, environment-independent
        let lut = compute_brdf_lut(256, 256);
        let lut_path = dir.join("brdf_lut.cooked_tex");
        write_rtex(&lut_path, 256, 256, &lut)?;

        Ok(())
    }
}

// --- Equirect helper ---

struct Equirect {
    pixels: Vec<f32>, // flat RGBA f32 linear
    w: u32,
    h: u32,
}

impl Equirect {
    fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path).with_context(|| format!("read {path:?}"))?;
        if data.len() < 16 || &data[0..4] != b"RTEX" {
            bail!("invalid RTEX in {path:?}");
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let w = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let h = u32::from_le_bytes(data[12..16].try_into().unwrap());
        let pixels = if version == 2 {
            // v2: Rgba32Float (4 bytes per channel, f32 LE)
            data[16..]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        } else {
            // v1: Rgba8UnormSrgb — decode to linear f32 at load time
            data[16..]
                .iter()
                .map(|&b| srgb_to_linear(b as f32 / 255.0))
                .collect()
        };
        Ok(Self { pixels, w, h })
    }

    // Sample at a world-space direction; nearest-neighbor, returns linear HDR values.
    fn sample(&self, dir: [f32; 3]) -> [f32; 3] {
        let (u, v) = dir_to_equirect_uv(dir);
        let x = ((u * self.w as f32) as u32).min(self.w - 1);
        let y = ((v * self.h as f32) as u32).min(self.h - 1);
        let i = ((y * self.w + x) * 4) as usize;
        [self.pixels[i], self.pixels[i + 1], self.pixels[i + 2]]
    }
}

// --- Coordinate helpers ---

fn equirect_uv_to_dir(u: f32, v: f32) -> [f32; 3] {
    let az = (u - 0.5) * 2.0 * PI; // consistent with dir_to_equirect_uv: u = 0.5 + az/(2π)
    let elev = (0.5 - v) * PI;
    let cos_e = elev.cos();
    [cos_e * az.cos(), elev.sin(), cos_e * az.sin()]
}

fn dir_to_equirect_uv(dir: [f32; 3]) -> (f32, f32) {
    let [x, y, z] = normalize(dir);
    let az = z.atan2(x);
    let elev = y.clamp(-1.0, 1.0).asin();
    let u = (0.5 + az / (2.0 * PI)).rem_euclid(1.0);
    let v = (0.5 - elev / PI).clamp(0.0, 1.0);
    (u, v)
}

// --- Vector math ---

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-8);
    [v[0] / len, v[1] / len, v[2] / len]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// Build an orthonormal tangent frame (t, b) from normal n.
fn tangent_frame(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let up = if n[1].abs() < 0.9 {
        [0.0f32, 1.0, 0.0]
    } else {
        [1.0f32, 0.0, 0.0]
    };
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    (t, b)
}

// Transform v from tangent space (where n=[0,0,1]) to world space.
fn tangent_to_world(v: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    let (t, b) = tangent_frame(n);
    normalize([
        t[0] * v[0] + b[0] * v[1] + n[0] * v[2],
        t[1] * v[0] + b[1] * v[1] + n[1] * v[2],
        t[2] * v[0] + b[2] * v[1] + n[2] * v[2],
    ])
}

// --- Hammersley quasi-random sequence ---

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * 2.328_306_4e-10 // / 2^32
}

fn hammersley(i: u32, n: u32) -> (f32, f32) {
    (i as f32 / n as f32, radical_inverse_vdc(i))
}

// GGX importance sampling: returns half-vector in world space around n.
fn importance_sample_ggx(xi: (f32, f32), roughness: f32, n: [f32; 3]) -> [f32; 3] {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.0;
    let cos_t = ((1.0 - xi.1) / (1.0 + (a * a - 1.0) * xi.1))
        .max(0.0)
        .sqrt();
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    let h_local = [sin_t * phi.cos(), sin_t * phi.sin(), cos_t];
    tangent_to_world(h_local, n)
}

// --- Diffuse irradiance: cosine-weighted Riemann sum ---

fn compute_irradiance(eq: &Equirect, out_w: u32, out_h: u32) -> Vec<u8> {
    const N_PHI: u32 = 64;
    const N_THETA: u32 = 32;
    let dphi = 2.0 * PI / N_PHI as f32;
    let dtheta = 0.5 * PI / N_THETA as f32;

    let mut out = Vec::with_capacity((out_w * out_h * 4) as usize);
    for y in 0..out_h {
        for x in 0..out_w {
            let u = (x as f32 + 0.5) / out_w as f32;
            let v = (y as f32 + 0.5) / out_h as f32;
            let n = equirect_uv_to_dir(u, v);

            let mut irr = [0.0f32; 3];
            let mut wt = 0.0f32;

            for j in 0..N_THETA {
                for i in 0..N_PHI {
                    let phi = (i as f32 + 0.5) * dphi;
                    let theta = (j as f32 + 0.5) * dtheta;
                    let sin_t = theta.sin();
                    let cos_t = theta.cos();
                    let local = [sin_t * phi.cos(), sin_t * phi.sin(), cos_t];
                    let world = tangent_to_world(local, n);
                    let sample = eq.sample(world);
                    let w = cos_t * sin_t;
                    irr[0] += sample[0] * w;
                    irr[1] += sample[1] * w;
                    irr[2] += sample[2] * w;
                    wt += w;
                }
            }

            let s = PI / wt;
            out.push(linear_to_srgb(irr[0] * s));
            out.push(linear_to_srgb(irr[1] * s));
            out.push(linear_to_srgb(irr[2] * s));
            out.push(255u8);
        }
    }
    out
}

// --- Specular prefilter: GGX importance sampling, HDR linear output ---

fn compute_specular_prefilter(eq: &Equirect, out_w: u32, out_h: u32, roughness: f32) -> Vec<f32> {
    const N_SAMPLES: u32 = 256;

    let mut out = Vec::with_capacity((out_w * out_h * 4) as usize);
    for y in 0..out_h {
        for x in 0..out_w {
            let u = (x as f32 + 0.5) / out_w as f32;
            let v = (y as f32 + 0.5) / out_h as f32;
            let n = equirect_uv_to_dir(u, v);

            let mut col = [0.0f32; 3];
            let mut wt = 0.0f32;

            for i in 0..N_SAMPLES {
                let xi = hammersley(i, N_SAMPLES);
                let h = importance_sample_ggx(xi, roughness, n);
                let vdoth = dot(n, h).max(0.0);
                let l = normalize([
                    2.0 * vdoth * h[0] - n[0],
                    2.0 * vdoth * h[1] - n[1],
                    2.0 * vdoth * h[2] - n[2],
                ]);
                let ndotl = dot(n, l).max(0.0);
                if ndotl > 0.0 {
                    let s = eq.sample(l);
                    col[0] += s[0] * ndotl;
                    col[1] += s[1] * ndotl;
                    col[2] += s[2] * ndotl;
                    wt += ndotl;
                }
            }

            let s = 1.0 / wt.max(1e-6);
            out.push(col[0] * s);
            out.push(col[1] * s);
            out.push(col[2] * s);
            out.push(1.0f32);
        }
    }
    out
}

// --- BRDF LUT: GGX split-sum preintegration (Karis / UE4) ---
// Stored linear (not sRGB): R = F0 scale, G = F0 bias.

fn geometry_schlick_ggx_ibl(ndotv: f32, roughness: f32) -> f32 {
    let k = roughness * roughness / 2.0;
    ndotv / (ndotv * (1.0 - k) + k)
}

fn geometry_smith_ibl(ndotv: f32, ndotl: f32, roughness: f32) -> f32 {
    geometry_schlick_ggx_ibl(ndotv, roughness) * geometry_schlick_ggx_ibl(ndotl, roughness)
}

fn compute_brdf_lut(w: u32, h: u32) -> Vec<u8> {
    const N: u32 = 1024;

    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            // x = NdotV (0=grazing, 1=normal incidence)
            // y = roughness (0=smooth, 1=rough) — y=0 is top of texture
            let ndotv = (x as f32 + 0.5) / w as f32;
            let roughness = (y as f32 + 0.5) / h as f32;
            let roughness = roughness.max(0.04);

            let v = [(1.0 - ndotv * ndotv).max(0.0).sqrt(), 0.0f32, ndotv];
            let n = [0.0f32, 0.0, 1.0];

            let mut scale = 0.0f32;
            let mut bias = 0.0f32;

            for i in 0..N {
                let xi = hammersley(i, N);
                let h_v = importance_sample_ggx(xi, roughness, n);
                let vdoth = dot(v, h_v).max(0.0);
                let l = normalize([
                    2.0 * vdoth * h_v[0] - v[0],
                    2.0 * vdoth * h_v[1] - v[1],
                    2.0 * vdoth * h_v[2] - v[2],
                ]);
                let ndotl = l[2].max(0.0); // n = [0,0,1]
                let ndoth = h_v[2].max(0.0);

                if ndotl > 0.0 {
                    let g_vis = geometry_smith_ibl(ndotv, ndotl, roughness) * vdoth
                        / (ndoth * ndotv.max(0.001));
                    let fc = (1.0 - vdoth).powi(5);
                    scale += (1.0 - fc) * g_vis;
                    bias += fc * g_vis;
                }
            }

            scale = (scale / N as f32).clamp(0.0, 1.0);
            bias = (bias / N as f32).clamp(0.0, 1.0);

            // Linear bytes (not sRGB — BRDF coefficients are in linear space)
            out.push((scale * 255.0 + 0.5) as u8);
            out.push((bias * 255.0 + 0.5) as u8);
            out.push(0u8);
            out.push(255u8);
        }
    }
    out
}
