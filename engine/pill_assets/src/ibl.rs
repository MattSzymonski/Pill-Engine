//! Image-based lighting bake — the single source of the IBL math.
//!
//! Pure CPU, no filesystem: used at build time (the `EquirectToIBL` / `BrdfLut` rules) and at
//! runtime (games baking from a fetched HDR, including on WASM). Inputs/outputs are plain f32
//! RGBA buffers; callers handle I/O and GPU upload.

use std::f32::consts::PI;

use anyhow::{anyhow, bail, Result};

use crate::util::{
    color::srgb_to_linear,
    vec_math::{dot3, normalize3, tangent_frame},
};

/// Diffuse irradiance map size (Rgba32Float).
pub const DIFFUSE_SIZE: (u32, u32) = (32, 16);
/// Specular prefilter base mip size (Rgba32Float, 5 mips).
pub const SPECULAR_SIZE: (u32, u32) = (128, 64);

/// Decodes a Radiance `.hdr` equirectangular panorama and resizes it to 512×256 RGBA f32.
/// (Runtime equivalent of the build-time HDR decode step.) Bilinear downsample is plenty for a
/// low-frequency environment map.
pub fn equirect_from_hdr(bytes: &[u8]) -> Result<Equirect> {
    let mut decoder = zune_hdr::HdrDecoder::new(bytes);
    let rgb: Vec<f32> = decoder.decode().map_err(|e| anyhow!("decode HDR: {e:?}"))?;
    let (src_w, src_h) = decoder
        .get_dimensions()
        .ok_or_else(|| anyhow!("HDR has no dimensions"))?;

    let (out_w, out_h) = (512usize, 256usize);
    let mut pixels = Vec::with_capacity(out_w * out_h * 4);
    let sample = |x: usize, y: usize, c: usize| rgb[(y * src_w + x) * 3 + c];
    for oy in 0..out_h {
        let fy = ((oy as f32 + 0.5) / out_h as f32 * src_h as f32 - 0.5).max(0.0);
        let y0 = (fy as usize).min(src_h - 1);
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = fy - y0 as f32;
        for ox in 0..out_w {
            let fx = ((ox as f32 + 0.5) / out_w as f32 * src_w as f32 - 0.5).max(0.0);
            let x0 = (fx as usize).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = fx - x0 as f32;
            let mut rgba = [0.0f32, 0.0, 0.0, 1.0];
            for (c, out) in rgba.iter_mut().take(3).enumerate() {
                let top = sample(x0, y0, c) + (sample(x1, y0, c) - sample(x0, y0, c)) * wx;
                let bot = sample(x0, y1, c) + (sample(x1, y1, c) - sample(x0, y1, c)) * wx;
                *out = top + (bot - top) * wy;
            }
            pixels.extend_from_slice(&rgba);
        }
    }
    Ok(Equirect {
        pixels,
        width: out_w as u32,
        height: out_h as u32,
    })
}

/// Bakes IBL maps from an Rgba32Float equirect panorama.
/// Returns (diffuse `DIFFUSE_SIZE`, specular mips from `SPECULAR_SIZE`), all Rgba32Float. The
/// BRDF LUT is environment-independent and engine-owned.
pub fn bake_all(equirect: &Equirect) -> (Vec<f32>, Vec<Vec<f32>>) {
    let diffuse = compute_irradiance(equirect, DIFFUSE_SIZE.0, DIFFUSE_SIZE.1);
    let specular_mips = bake_specular_mips(equirect);
    (diffuse, specular_mips)
}

/// Specular prefilter mip chain (5 levels from `SPECULAR_SIZE`), Rgba32Float HDR.
/// Mip i roughness = i/4, matching shader roughness * MAX_REFLECTION_LOD(4).
pub fn bake_specular_mips(equirect: &Equirect) -> Vec<Vec<f32>> {
    const MIP_ROUGHNESS: [f32; 5] = [0.04, 0.25, 0.5, 0.75, 1.0];
    let mut specular_mips: Vec<Vec<f32>> = Vec::with_capacity(MIP_ROUGHNESS.len());
    for (mip_level, &roughness) in MIP_ROUGHNESS.iter().enumerate() {
        let width = (SPECULAR_SIZE.0 >> mip_level).max(1);
        let height = (SPECULAR_SIZE.1 >> mip_level).max(1);
        specular_mips.push(compute_specular_prefilter(
            equirect, width, height, roughness,
        ));
    }
    specular_mips
}

// --- Equirect ---

/// Equirect panorama as flat RGBA f32 (linear).
pub struct Equirect {
    pub pixels: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl Equirect {
    /// Decodes an RTEX v2 (Rgba32Float) or v1 (sRGB RGBA8) equirect into linear f32.
    pub fn from_rtex(data: &[u8]) -> Result<Self> {
        if data.len() < 16 || &data[0..4] != b"RTEX" {
            bail!("invalid RTEX");
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let width = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let height = u32::from_le_bytes(data[12..16].try_into().unwrap());
        let pixels = if version == 2 {
            // v2: Rgba32Float (4 bytes per channel, f32 LE)
            let (texels, _) = data[16..].as_chunks::<4>();
            texels
                .iter()
                .map(|&bytes| f32::from_le_bytes(bytes))
                .collect()
        } else {
            // v1: Rgba8UnormSrgb — decode to linear f32 at load time
            data[16..]
                .iter()
                .map(|&byte| srgb_to_linear(byte as f32 / 255.0))
                .collect()
        };
        Ok(Self {
            pixels,
            width,
            height,
        })
    }

    // Sample at a world-space direction; nearest-neighbor, returns linear HDR values.
    pub fn sample(&self, direction: [f32; 3]) -> [f32; 3] {
        let (u_coordinate, v_coordinate) = dir_to_equirect_uv(direction);
        let x = ((u_coordinate * self.width as f32) as u32).min(self.width - 1);
        let y = ((v_coordinate * self.height as f32) as u32).min(self.height - 1);
        let pixel_index = ((y * self.width + x) * 4) as usize;
        [
            self.pixels[pixel_index],
            self.pixels[pixel_index + 1],
            self.pixels[pixel_index + 2],
        ]
    }
}

// --- Coordinate helpers ---

fn equirect_uv_to_dir(u_coordinate: f32, v_coordinate: f32) -> [f32; 3] {
    let azimuth = (u_coordinate - 0.5) * 2.0 * PI; // consistent with dir_to_equirect_uv: u = 0.5 + azimuth/(2π)
    let elevation = (0.5 - v_coordinate) * PI;
    let cos_elevation = elevation.cos();
    [
        cos_elevation * azimuth.cos(),
        elevation.sin(),
        cos_elevation * azimuth.sin(),
    ]
}

fn dir_to_equirect_uv(direction: [f32; 3]) -> (f32, f32) {
    let [x, y, z] = normalize3(direction);
    let azimuth = z.atan2(x);
    let elevation = y.clamp(-1.0, 1.0).asin();
    let u_coordinate = (0.5 + azimuth / (2.0 * PI)).rem_euclid(1.0);
    let v_coordinate = (0.5 - elevation / PI).clamp(0.0, 1.0);
    (u_coordinate, v_coordinate)
}

// --- Vector math ---

// Transform vector from tangent space (where normal=[0,0,1]) to world space.
fn tangent_to_world(vector: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let (tangent, bitangent) = tangent_frame(normal);
    normalize3([
        tangent[0] * vector[0] + bitangent[0] * vector[1] + normal[0] * vector[2],
        tangent[1] * vector[0] + bitangent[1] * vector[1] + normal[1] * vector[2],
        tangent[2] * vector[0] + bitangent[2] * vector[1] + normal[2] * vector[2],
    ])
}

// --- Hammersley quasi-random sequence ---

fn radical_inverse_van_der_corput(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * 2.328_306_4e-10 // / 2^32
}

fn hammersley(index: u32, count: u32) -> (f32, f32) {
    (
        index as f32 / count as f32,
        radical_inverse_van_der_corput(index),
    )
}

// GGX importance sampling: returns half-vector in world space around normal.
fn importance_sample_ggx(sample_point: (f32, f32), roughness: f32, normal: [f32; 3]) -> [f32; 3] {
    let alpha = roughness * roughness;
    let phi = 2.0 * PI * sample_point.0;
    let cos_theta = ((1.0 - sample_point.1) / (1.0 + (alpha * alpha - 1.0) * sample_point.1))
        .max(0.0)
        .sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let half_vector_local = [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta];
    tangent_to_world(half_vector_local, normal)
}

// --- Diffuse irradiance: cosine-weighted Riemann sum ---

pub fn compute_irradiance(equirect: &Equirect, output_width: u32, output_height: u32) -> Vec<f32> {
    const PHI_SAMPLE_COUNT: u32 = 64;
    const THETA_SAMPLE_COUNT: u32 = 32;
    let delta_phi = 2.0 * PI / PHI_SAMPLE_COUNT as f32;
    let delta_theta = 0.5 * PI / THETA_SAMPLE_COUNT as f32;

    let mut output = Vec::with_capacity((output_width * output_height * 4) as usize);
    for y in 0..output_height {
        for x in 0..output_width {
            let u_coordinate = (x as f32 + 0.5) / output_width as f32;
            let v_coordinate = (y as f32 + 0.5) / output_height as f32;
            let normal = equirect_uv_to_dir(u_coordinate, v_coordinate);

            let mut irradiance = [0.0f32; 3];
            let mut total_weight = 0.0f32;

            for theta_index in 0..THETA_SAMPLE_COUNT {
                for phi_index in 0..PHI_SAMPLE_COUNT {
                    let phi = (phi_index as f32 + 0.5) * delta_phi;
                    let theta = (theta_index as f32 + 0.5) * delta_theta;
                    let sin_theta = theta.sin();
                    let cos_theta = theta.cos();
                    let local = [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta];
                    let world = tangent_to_world(local, normal);
                    let sample = equirect.sample(world);
                    let weight = cos_theta * sin_theta;
                    irradiance[0] += sample[0] * weight;
                    irradiance[1] += sample[1] * weight;
                    irradiance[2] += sample[2] * weight;
                    total_weight += weight;
                }
            }

            // Cosine-weighted average radiance = E/π, so the shader's irradiance * albedo is Lambert
            let normalization = 1.0 / total_weight;
            output.push(irradiance[0] * normalization);
            output.push(irradiance[1] * normalization);
            output.push(irradiance[2] * normalization);
            output.push(1.0);
        }
    }
    output
}

// --- Specular prefilter: GGX importance sampling, HDR linear output ---

pub fn compute_specular_prefilter(
    equirect: &Equirect,
    output_width: u32,
    output_height: u32,
    roughness: f32,
) -> Vec<f32> {
    const SAMPLE_COUNT: u32 = 256;

    let mut output = Vec::with_capacity((output_width * output_height * 4) as usize);
    for y in 0..output_height {
        for x in 0..output_width {
            let u_coordinate = (x as f32 + 0.5) / output_width as f32;
            let v_coordinate = (y as f32 + 0.5) / output_height as f32;
            let normal = equirect_uv_to_dir(u_coordinate, v_coordinate);

            let mut color = [0.0f32; 3];
            let mut total_weight = 0.0f32;

            for sample_index in 0..SAMPLE_COUNT {
                let sample_point = hammersley(sample_index, SAMPLE_COUNT);
                let half_vector = importance_sample_ggx(sample_point, roughness, normal);
                let normal_dot_half = dot3(normal, half_vector).max(0.0);
                let light_direction = normalize3([
                    2.0 * normal_dot_half * half_vector[0] - normal[0],
                    2.0 * normal_dot_half * half_vector[1] - normal[1],
                    2.0 * normal_dot_half * half_vector[2] - normal[2],
                ]);
                let normal_dot_light = dot3(normal, light_direction).max(0.0);
                if normal_dot_light > 0.0 {
                    let sampled = equirect.sample(light_direction);
                    color[0] += sampled[0] * normal_dot_light;
                    color[1] += sampled[1] * normal_dot_light;
                    color[2] += sampled[2] * normal_dot_light;
                    total_weight += normal_dot_light;
                }
            }

            let normalization = 1.0 / total_weight.max(1e-6);
            output.push(color[0] * normalization);
            output.push(color[1] * normalization);
            output.push(color[2] * normalization);
            output.push(1.0f32);
        }
    }
    output
}

// --- BRDF LUT: GGX split-sum preintegration (Karis / UE4) ---
// Stored linear (not sRGB): R = F0 scale, G = F0 bias.

fn geometry_schlick_ggx_ibl(normal_dot_view: f32, roughness: f32) -> f32 {
    let k = roughness * roughness / 2.0;
    normal_dot_view / (normal_dot_view * (1.0 - k) + k)
}

fn geometry_smith_ibl(normal_dot_view: f32, normal_dot_light: f32, roughness: f32) -> f32 {
    geometry_schlick_ggx_ibl(normal_dot_view, roughness)
        * geometry_schlick_ggx_ibl(normal_dot_light, roughness)
}

pub fn compute_brdf_lut(width: u32, height: u32) -> Vec<f32> {
    const SAMPLE_COUNT: u32 = 1024;

    let mut output = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            // x = NdotV (0=grazing, 1=normal incidence)
            // y = roughness (0=smooth, 1=rough) — y=0 is top of texture
            let normal_dot_view = (x as f32 + 0.5) / width as f32;
            let roughness = (y as f32 + 0.5) / height as f32;
            let roughness = roughness.max(0.04);

            let view_direction = [
                (1.0 - normal_dot_view * normal_dot_view).max(0.0).sqrt(),
                0.0f32,
                normal_dot_view,
            ];
            let normal = [0.0f32, 0.0, 1.0];

            let mut scale = 0.0f32;
            let mut bias = 0.0f32;

            for sample_index in 0..SAMPLE_COUNT {
                let sample_point = hammersley(sample_index, SAMPLE_COUNT);
                let half_vector = importance_sample_ggx(sample_point, roughness, normal);
                let view_dot_half = dot3(view_direction, half_vector).max(0.0);
                let light_direction = normalize3([
                    2.0 * view_dot_half * half_vector[0] - view_direction[0],
                    2.0 * view_dot_half * half_vector[1] - view_direction[1],
                    2.0 * view_dot_half * half_vector[2] - view_direction[2],
                ]);
                let normal_dot_light = light_direction[2].max(0.0); // normal = [0,0,1]
                let normal_dot_half = half_vector[2].max(0.0);

                if normal_dot_light > 0.0 {
                    let geometry_visibility =
                        geometry_smith_ibl(normal_dot_view, normal_dot_light, roughness)
                            * view_dot_half
                            / (normal_dot_half * normal_dot_view.max(0.001));
                    let fresnel_coefficient = (1.0 - view_dot_half).powi(5);
                    scale += (1.0 - fresnel_coefficient) * geometry_visibility;
                    bias += fresnel_coefficient * geometry_visibility;
                }
            }

            scale = (scale / SAMPLE_COUNT as f32).clamp(0.0, 1.0);
            bias = (bias / SAMPLE_COUNT as f32).clamp(0.0, 1.0);

            output.push(scale);
            output.push(bias);
            output.push(0.0);
            output.push(1.0);
        }
    }
    output
}
