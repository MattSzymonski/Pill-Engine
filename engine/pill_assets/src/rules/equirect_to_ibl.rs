use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use super::studio_equirect::{write_rtex, write_rtex_hdr_mip};
use crate::ibl::{self, Equirect};
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
        let equirect = load_equirect(input)?;
        let dir = output.parent().unwrap();
        let stem = input.file_stem().unwrap().to_str().unwrap();
        let base = stem.strip_suffix("_equirect").unwrap_or(stem);

        // 1. Diffuse irradiance (32×16 sRGB) — anchor output
        let irr = ibl::compute_irradiance(&equirect, 32, 16);
        write_rtex(output, 32, 16, &irr)?;

        // 2. Specular prefilter HDR mip chain (RTEX v4, Rgba32Float).
        let spec_path = dir.join(format!("{base}_specular_ibl.cooked_tex"));
        let spec_mips = ibl::bake_specular_mips(&equirect);
        write_rtex_hdr_mip(&spec_path, 128, 64, &spec_mips)?;

        // 3. BRDF LUT (256×256 linear) — split-sum preintegration, environment-independent
        let lut = ibl::compute_brdf_lut(256, 256);
        let lut_path = dir.join("brdf_lut.cooked_tex");
        write_rtex(&lut_path, 256, 256, &lut)?;

        Ok(())
    }
}

/// Loads a cooked equirect (RTEX v1 sRGB or v2 Rgba32Float) into the linear-f32 `Equirect` the
/// bake math operates on.
fn load_equirect(path: &Path) -> Result<Equirect> {
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
            .map(|&b| ibl::srgb_to_linear(b as f32 / 255.0))
            .collect()
    };
    Ok(Equirect { pixels, w, h })
}
