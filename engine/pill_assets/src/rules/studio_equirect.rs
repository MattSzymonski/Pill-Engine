use std::f32::consts::PI;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::Rule;

// Gradient (linear, inverse-Reinhard compensated: c/(1+c) → target on screen).
const TOP: [f32; 3] = [0.90, 0.90, 1.20]; // Reinhard → [0.47, 0.47, 0.55] cool blue ceiling
const HORIZON: [f32; 3] = [0.43, 0.43, 0.46]; // Reinhard → [0.30, 0.30, 0.32] neutral mid
const BOTTOM: [f32; 3] = [0.040, 0.040, 0.050]; // Reinhard → [0.04, 0.04, 0.05] dark floor

// Key light: warm tungsten, upper-left, ~40° above horizon.
const KEY_U: f32 = 0.10;
const KEY_V: f32 = 0.28;
const KEY_INTENSITY: [f32; 3] = [6.0, 4.8, 2.5]; // Reinhard → [0.86, 0.83, 0.71] warm gold
const KEY_SIGMA: f32 = 0.0875;

// Rim light: cool blue, upper-right, ~45° above horizon.
const RIM_U: f32 = 0.60;
const RIM_V: f32 = 0.25;
const RIM_INTENSITY: [f32; 3] = [1.5, 2.5, 5.0]; // Reinhard → [0.60, 0.71, 0.83] cool blue
const RIM_SIGMA: f32 = 0.100;

pub struct StudioEquirect;

impl Rule for StudioEquirect {
    fn name(&self) -> &'static str {
        "studio_equirect"
    }

    fn input_glob(&self) -> &'static str {
        "**/*.studio"
    }

    fn output_for(&self, input: &Path) -> PathBuf {
        let stem = input.file_stem().unwrap().to_str().unwrap();
        input.with_file_name(format!("{stem}_equirect.cooked_tex"))
    }

    fn build(&self, _input: &Path, output: &Path) -> Result<()> {
        let (w, h) = (512u32, 256u32);
        let pixels = generate_equirect(w, h);
        write_rtex_hdr(output, w, h, &pixels)
    }
}

fn generate_equirect(w: u32, h: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let v = (y as f32 + 0.5) / h as f32;
            let elev = (0.5 - v) * PI;
            let sin_e = elev.sin();
            let t = (sin_e + 1.0) * 0.5;
            let mut col = if t < 0.5 {
                lerp3(BOTTOM, HORIZON, t * 2.0)
            } else {
                lerp3(HORIZON, TOP, (t - 0.5) * 2.0)
            };
            let dku = u - KEY_U;
            let dku = (dku - dku.round()).abs(); // wrap U to [-0.5, 0.5]
            let dkv = v - KEY_V;
            let w_key = (-(dku * dku + dkv * dkv) / (2.0 * KEY_SIGMA * KEY_SIGMA)).exp();
            col[0] = col[0].max(KEY_INTENSITY[0] * w_key);
            col[1] = col[1].max(KEY_INTENSITY[1] * w_key);
            col[2] = col[2].max(KEY_INTENSITY[2] * w_key);
            let dru = u - RIM_U;
            let dru = (dru - dru.round()).abs();
            let drv = v - RIM_V;
            let w_rim = (-(dru * dru + drv * drv) / (2.0 * RIM_SIGMA * RIM_SIGMA)).exp();
            col[0] = col[0].max(RIM_INTENSITY[0] * w_rim);
            col[1] = col[1].max(RIM_INTENSITY[1] * w_rim);
            col[2] = col[2].max(RIM_INTENSITY[2] * w_rim);
            out.push(col[0]);
            out.push(col[1]);
            out.push(col[2]);
            out.push(1.0);
        }
    }
    out
}

// --- shared utilities ---

// RTEX v2: Rgba32Float (4 bytes per channel, f32 LE), linear HDR.
pub(crate) fn write_rtex_hdr(path: &Path, w: u32, h: u32, rgba_f32: &[f32]) -> Result<()> {
    let mut out = Vec::with_capacity(16 + rgba_f32.len() * 4);
    out.extend_from_slice(b"RTEX");
    out.extend_from_slice(&2u32.to_le_bytes()); // version 2 = Rgba32Float HDR
    out.extend_from_slice(&w.to_le_bytes());
    out.extend_from_slice(&h.to_le_bytes());
    for &v in rgba_f32 {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &out).with_context(|| format!("write_rtex_hdr {path:?}"))?;
    Ok(())
}

// RTEX v4: Rgba32Float mip chain for HDR specular prefilter.
// Header: RTEX | 4 | w | h | mip_count | mip0_f32_pixels | mip1_f32_pixels | ...
pub(crate) fn write_rtex_hdr_mip(path: &Path, w: u32, h: u32, mips: &[Vec<f32>]) -> Result<()> {
    let mut out = Vec::new();
    out.extend_from_slice(b"RTEX");
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&w.to_le_bytes());
    out.extend_from_slice(&h.to_le_bytes());
    out.extend_from_slice(&(mips.len() as u32).to_le_bytes());
    for mip in mips {
        for &v in mip {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, &out).with_context(|| format!("write_rtex_hdr_mip {path:?}"))?;
    Ok(())
}

// RTEX v1: Rgba8UnormSrgb (legacy LDR IBL outputs).
pub(crate) fn write_rtex(path: &Path, w: u32, h: u32, rgba: &[u8]) -> Result<()> {
    let mut out = Vec::with_capacity(16 + rgba.len());
    out.extend_from_slice(b"RTEX");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&w.to_le_bytes());
    out.extend_from_slice(&h.to_le_bytes());
    out.extend_from_slice(rgba);
    std::fs::write(path, &out).with_context(|| format!("write_rtex {path:?}"))?;
    Ok(())
}

pub(crate) fn linear_to_srgb(x: f32) -> u8 {
    let c = x.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

pub(crate) fn srgb_to_linear(x: f32) -> f32 {
    if x <= 0.040_45 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

pub(crate) fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}
