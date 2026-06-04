//! Fetches the multi-file Sponza glTF over HTTP and hands the bytes to the engine's runtime glTF
//! loader. Mirrors three.js's GLTFLoader: fetch `Sponza.gltf`, then resolve+fetch `Sponza.bin` and
//! every texture relative to the base URL.
//!
//! All glTF parsing, image decoding and mesh/texture byte-format conversion lives in the engine
//! (`pill_engine`'s `gltf_loading` feature). This module is just I/O + orchestration: the fetch is
//! the only platform-specific part (async `fetch` on WASM, blocking `ureq` on native).

use pill_engine::game::{
    bake_all, equirect_from_hdr, gltf_resource_uris, parse_gltf_scene, GltfSceneData,
};

use crate::fetch;

pub struct SponzaCpu {
    pub scene: GltfSceneData,
    pub equirect: (Vec<f32>, u32, u32), // RGBA f32 background panorama (from fetched HDR)
    pub ibl: (Vec<f32>, Vec<Vec<f32>>, Vec<f32>), // bake::bake_all output: diffuse, specular mips, brdf_lut
}

#[cfg(not(target_arch = "wasm32"))]
pub fn base_url() -> &'static str {
    fetch::SPONZA_BASE
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_native() -> Result<SponzaCpu, String> {
    let gltf_bytes = fetch::get(&format!("{}Sponza.gltf", fetch::SPONZA_BASE))?;
    let uris = gltf_resource_uris(&gltf_bytes).map_err(|e| e.to_string())?;

    let total = 2 + uris.buffers.len() + uris.images.len();
    let mut progress = Progress::new(total);
    progress.tick(gltf_bytes.len());

    let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(uris.buffers.len());
    for uri in &uris.buffers {
        let bytes = match uri {
            Some(uri) => fetch::get(&format!("{}{}", fetch::SPONZA_BASE, uri))?,
            None => Vec::new(), // embedded BIN; engine resolves it from the blob
        };
        progress.tick(bytes.len());
        buffers.push(bytes);
    }

    let mut images: Vec<Vec<u8>> = Vec::with_capacity(uris.images.len());
    for uri in &uris.images {
        let bytes = match uri {
            Some(uri) => fetch::get(&format!("{}{}", fetch::SPONZA_BASE, uri))?,
            None => Vec::new(), // embedded in a buffer view; engine resolves it
        };
        progress.tick(bytes.len());
        images.push(bytes);
    }

    let scene = parse_gltf_scene(&gltf_bytes, &buffers, &images).map_err(|e| e.to_string())?;

    let hdr = fetch::get(fetch::HDR_URL)?;
    progress.tick(hdr.len());

    finish(scene, &hdr)
}

#[cfg(target_arch = "wasm32")]
pub async fn load_wasm() -> Result<SponzaCpu, String> {
    let gltf_bytes = fetch::get(&format!("{}Sponza.gltf", fetch::SPONZA_BASE)).await?;
    let uris = gltf_resource_uris(&gltf_bytes).map_err(|e| e.to_string())?;

    let total = 2 + uris.buffers.len() + uris.images.len();
    let mut progress = Progress::new(total);
    progress.tick(gltf_bytes.len());

    let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(uris.buffers.len());
    for uri in &uris.buffers {
        let bytes = match uri {
            Some(uri) => fetch::get(&format!("{}{}", fetch::SPONZA_BASE, uri)).await?,
            None => Vec::new(), // embedded BIN; engine resolves it from the blob
        };
        progress.tick(bytes.len());
        buffers.push(bytes);
    }

    let mut images: Vec<Vec<u8>> = Vec::with_capacity(uris.images.len());
    for uri in &uris.images {
        let bytes = match uri {
            Some(uri) => fetch::get(&format!("{}{}", fetch::SPONZA_BASE, uri)).await?,
            None => Vec::new(),
        };
        progress.tick(bytes.len());
        images.push(bytes);
    }

    let scene = parse_gltf_scene(&gltf_bytes, &buffers, &images).map_err(|e| e.to_string())?;

    let hdr = fetch::get(fetch::HDR_URL).await?;
    progress.tick(hdr.len());

    finish(scene, &hdr)
}

/// Bakes IBL from the fetched HDR and packages the result.
fn finish(scene: GltfSceneData, hdr_bytes: &[u8]) -> Result<SponzaCpu, String> {
    let equirect = equirect_from_hdr(hdr_bytes).map_err(|e| e.to_string())?;
    let ibl = bake_all(&equirect.0, equirect.1, equirect.2);
    log::info!(
        "Sponza: parsed {} meshes, {} materials, {} textures",
        scene.meshes.len(),
        scene.materials.len(),
        scene.textures.len(),
    );
    Ok(SponzaCpu {
        scene,
        equirect,
        ibl,
    })
}

// --- Progress (log only, fully derived from the document) ---

struct Progress {
    done: usize,
    total: usize,
    bytes: usize,
}

impl Progress {
    fn new(total: usize) -> Self {
        Self {
            done: 0,
            total,
            bytes: 0,
        }
    }
    fn tick(&mut self, last: usize) {
        self.done += 1;
        self.bytes += last;
        log::info!(
            "Sponza: file {}/{}, +{} KB, {:.1} MB so far",
            self.done,
            self.total,
            last / 1024,
            self.bytes as f64 / (1024.0 * 1024.0),
        );
    }
}
