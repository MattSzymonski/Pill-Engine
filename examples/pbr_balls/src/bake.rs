use std::f32::consts::PI;

// Procedural studio panorama for this example. The IBL bake itself (`bake_all`) is shared engine
// code — see `pill_engine::game::bake_all`.

// Gradient (linear, inverse-Reinhard compensated: c/(1+c) → target on screen).
const TOP: [f32; 3] = [0.90, 0.90, 1.20]; // Reinhard → [0.47, 0.47, 0.55] cool blue ceiling
const HORIZON: [f32; 3] = [0.43, 0.43, 0.46]; // Reinhard → [0.30, 0.30, 0.32] neutral mid
const BOTTOM: [f32; 3] = [0.040, 0.040, 0.050]; // Reinhard → [0.04, 0.04, 0.05] dark floor

// Key light: warm tungsten, upper-left, ~40° above horizon.
const KEY_U: f32 = 0.10;
const KEY_V: f32 = 0.28;
const KEY_INTENSITY: [f32; 3] = [6.0, 4.8, 2.5]; // Reinhard → [0.86, 0.83, 0.71] warm gold

// Rim light: cool blue, upper-right, ~45° above horizon.
const RIM_U: f32 = 0.60;
const RIM_V: f32 = 0.25;
const RIM_INTENSITY: [f32; 3] = [1.5, 2.5, 5.0]; // Reinhard → [0.60, 0.71, 0.83] cool blue

const LIGHT_K: f32 = 8.0; // glare falloff exponent
const KEY_SKY: [f32; 3] = [0.010, 0.060, 0.200]; // 0.2 * vec3(0.05, 0.3, 1.) — cool blue haze away from key

/// Generates the studio equirect panorama as Rgba32Float pixels (512×256 linear HDR).
pub fn generate() -> (Vec<f32>, u32, u32) {
    let (w, h) = (512u32, 256u32);
    let key_dir = equirect_uv_to_dir(KEY_U, KEY_V);
    let rim_dir = equirect_uv_to_dir(RIM_U, RIM_V);
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let v = (y as f32 + 0.5) / h as f32;
            let pixel_dir = equirect_uv_to_dir(u, v);
            let t = (pixel_dir[1] + 1.0) * 0.5; // pixel_dir[1] = sin(elev) ∈ [-1, 1]
            let mut col = if t < 0.5 {
                lerp3(BOTTOM, HORIZON, t * 2.0)
            } else {
                lerp3(HORIZON, TOP, (t - 0.5) * 2.0)
            };
            // key: warm glare + cool sky haze (FF2023 sun glare formula)
            let sun = dot(pixel_dir, key_dir).clamp(0.0, 1.0);
            let sky = 1.0 - sun.powf(0.5);
            col[0] += KEY_INTENSITY[0] * sun.powf(LIGHT_K) + KEY_SKY[0] * sky;
            col[1] += KEY_INTENSITY[1] * sun.powf(LIGHT_K) + KEY_SKY[1] * sky;
            col[2] += KEY_INTENSITY[2] * sun.powf(LIGHT_K) + KEY_SKY[2] * sky;
            // rim: glare only
            let w_rim = dot(pixel_dir, rim_dir).clamp(0.0, 1.0).powf(LIGHT_K);
            col[0] += RIM_INTENSITY[0] * w_rim;
            col[1] += RIM_INTENSITY[1] * w_rim;
            col[2] += RIM_INTENSITY[2] * w_rim;
            out.push(col[0]);
            out.push(col[1]);
            out.push(col[2]);
            out.push(1.0);
        }
    }
    (out, w, h)
}

fn equirect_uv_to_dir(u: f32, v: f32) -> [f32; 3] {
    let az = (u - 0.5) * 2.0 * PI; // consistent with dir_to_equirect_uv: u = 0.5 + az/(2π)
    let elev = (0.5 - v) * PI;
    let cos_e = elev.cos();
    [cos_e * az.cos(), elev.sin(), cos_e * az.sin()]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}
