/// Loads the studio equirect panorama embedded at compile time from the build script's OUT_DIR.
/// Returns RGBA f32 pixels + (width, height).
///
/// The IBL bake itself (`bake_all`) is shared engine code — see `pill_engine::game::bake_all`.
pub fn load_equirect() -> (Vec<f32>, u32, u32) {
    // Blob layout written by build.rs: [u32 width][u32 height][f32 RGBA pixels...], all little-endian.
    let raw: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/equirect.f32bin"));
    let w = u32::from_le_bytes(raw[0..4].try_into().unwrap());
    let h = u32::from_le_bytes(raw[4..8].try_into().unwrap());
    let pixel_bytes = &raw[8..];
    // Reinterpret every 4 bytes as one f32 (chunks_exact(4) walks the blob 4 bytes at a time).
    let pixels: Vec<f32> = pixel_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    (pixels, w, h)
}
