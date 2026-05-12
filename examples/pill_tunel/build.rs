use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use pill_assets::{Pipeline, Rule};

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("res");

    let rules = [
        TextureRule {
            name: "pill_albedo",
            output: "pill_color.png",
            resolution: 1024,
            fill: image::Rgba([128, 128, 128, 255]),
            pixel: |obj, axis, split| {
                let base = if obj[axis] < split {
                    [0.55f32, 0.06, 0.08] // crimson
                } else {
                    [0.10f32, 0.10, 0.72] // cobalt
                };
                let amp = if obj[axis] < split { 0.06 } else { 0.02 };
                let v = fbm([obj[0] * 3.0, obj[1] * 3.0, obj[2] * 3.0], 4) * amp;
                image::Rgba([
                    ((base[0] + v).clamp(0.0, 1.0) * 255.0) as u8,
                    ((base[1] + v).clamp(0.0, 1.0) * 255.0) as u8,
                    ((base[2] + v).clamp(0.0, 1.0) * 255.0) as u8,
                    255,
                ])
            },
        },
        TextureRule {
            name: "pill_normal_fbm",
            output: "pill_normal.png",
            resolution: 1024,
            fill: image::Rgba([128, 128, 255, 255]),
            pixel: |obj, axis, split| {
                if obj[axis] >= split {
                    return image::Rgba([128, 128, 255, 255]);
                }
                // FBM gradient in object space as tangent-space approximation
                let p = [obj[0] * 6.0, obj[1] * 6.0, obj[2] * 6.0];
                let amp = 0.035f32;
                let [gx, gy, _] = fbm_grad(p, 3);
                let nx = (-gx * amp).clamp(-1.0, 1.0);
                let ny = (-gy * amp).clamp(-1.0, 1.0);
                let nz = (1.0 - nx * nx - ny * ny).sqrt().max(0.0);
                image::Rgba([
                    ((nx * 0.5 + 0.5) * 255.0) as u8,
                    ((ny * 0.5 + 0.5) * 255.0) as u8,
                    ((nz * 0.5 + 0.5) * 255.0) as u8,
                    255,
                ])
            },
        },
    ];

    // Track outputs so a missing file forces a re-run
    for r in &rules {
        println!("cargo:rerun-if-changed={}", root.join("textures/generated").join(r.output).display());
    }

    let pipeline = Pipeline {
        root,
        rules: rules.into_iter().map(|r| Box::new(r) as Box<dyn Rule>).collect(),
    };
    let stats = pipeline.run().expect("game asset pipeline failed");
    println!("cargo:rerun-if-changed=build.rs");
    for p in &stats.discovered {
        println!("cargo:rerun-if-changed={}", p.display());
    }
}

// ── Rules ─────────────────────────────────────────────────────────────────────

struct TextureRule {
    name:       &'static str,
    output:     &'static str, // filename under textures/generated/
    resolution: u32,
    fill:       image::Rgba<u8>,
    pixel:      fn([f32; 3], usize, f32) -> image::Rgba<u8>,
}

impl Rule for TextureRule {
    fn name(&self)       -> &'static str { self.name }
    fn input_glob(&self) -> &'static str { "models/pill.obj" }

    fn output_for(&self, input: &Path) -> PathBuf {
        input.parent().unwrap().parent().unwrap()
            .join("textures/generated")
            .join(self.output)
    }

    fn build(&self, input: &Path, output: &Path) -> Result<()> {
        let models = load_models(input)?;
        let (axis, split) = long_axis_and_split(&models);
        let mut img = image::RgbaImage::from_pixel(self.resolution, self.resolution, self.fill);
        for model in &models {
            let mesh = &model.mesh;
            if mesh.texcoords.is_empty() {
                bail!("{input:?}: model '{}' has no UV coordinates", model.name);
            }
            for_each_tri(mesh, |uv, pos| {
                rasterize_triangle(&mut img, uv, pos, |obj| (self.pixel)(obj, axis, split));
            });
        }
        img.save(output).with_context(|| format!("failed to save {output:?}"))?;
        Ok(())
    }
}

// ── Shared helpers ────────────────────────────────────────────────────────────

fn load_models(input: &Path) -> Result<Vec<tobj::Model>> {
    let (models, _) = tobj::load_obj(
        input,
        &tobj::LoadOptions { triangulate: true, single_index: true, ..Default::default() },
    )
    .with_context(|| format!("failed to load {input:?}"))?;
    Ok(models)
}

fn long_axis_and_split(models: &[tobj::Model]) -> (usize, f32) {
    let mut mn = [f32::MAX; 3];
    let mut mx = [f32::MIN; 3];
    for m in models {
        for i in (0..m.mesh.positions.len()).step_by(3) {
            for a in 0..3 {
                mn[a] = mn[a].min(m.mesh.positions[i + a]);
                mx[a] = mx[a].max(m.mesh.positions[i + a]);
            }
        }
    }
    let spans = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
    let axis = spans
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    (axis, (mn[axis] + mx[axis]) * 0.5)
}

fn for_each_tri(mesh: &tobj::Mesh, mut f: impl FnMut([[f32; 2]; 3], [[f32; 3]; 3])) {
    let n = mesh.indices.len() / 3;
    for t in 0..n {
        let i = [
            mesh.indices[t * 3] as usize,
            mesh.indices[t * 3 + 1] as usize,
            mesh.indices[t * 3 + 2] as usize,
        ];
        let uv = [
            [mesh.texcoords[i[0] * 2], mesh.texcoords[i[0] * 2 + 1]],
            [mesh.texcoords[i[1] * 2], mesh.texcoords[i[1] * 2 + 1]],
            [mesh.texcoords[i[2] * 2], mesh.texcoords[i[2] * 2 + 1]],
        ];
        let pos = [
            [mesh.positions[i[0]*3], mesh.positions[i[0]*3+1], mesh.positions[i[0]*3+2]],
            [mesh.positions[i[1]*3], mesh.positions[i[1]*3+1], mesh.positions[i[1]*3+2]],
            [mesh.positions[i[2]*3], mesh.positions[i[2]*3+1], mesh.positions[i[2]*3+2]],
        ];
        f(uv, pos);
    }
}

// ── Rasterizer ────────────────────────────────────────────────────────────────

fn rasterize_triangle<F>(
    img: &mut image::RgbaImage,
    uv: [[f32; 2]; 3],
    pos: [[f32; 3]; 3],
    pixel_fn: F,
) where
    F: Fn([f32; 3]) -> image::Rgba<u8>,
{
    let res = img.width();
    let rf = res as f32;
    // V flipped: UV origin is bottom-left, image origin is top-left
    let to_px = |u: [f32; 2]| -> [f32; 2] { [u[0] * rf, (1.0 - u[1]) * rf] };
    let p = [to_px(uv[0]), to_px(uv[1]), to_px(uv[2])];

    let min_x = (p[0][0].min(p[1][0]).min(p[2][0]).floor().max(0.0) as u32).min(res - 1);
    let max_x = (p[0][0].max(p[1][0]).max(p[2][0]).ceil() as u32).min(res - 1);
    let min_y = (p[0][1].min(p[1][1]).min(p[2][1]).floor().max(0.0) as u32).min(res - 1);
    let max_y = (p[0][1].max(p[1][1]).max(p[2][1]).ceil() as u32).min(res - 1);

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if let Some(bary) = barycentric([x as f32 + 0.5, y as f32 + 0.5], p[0], p[1], p[2]) {
                let obj = [
                    bary[0]*pos[0][0] + bary[1]*pos[1][0] + bary[2]*pos[2][0],
                    bary[0]*pos[0][1] + bary[1]*pos[1][1] + bary[2]*pos[2][1],
                    bary[0]*pos[0][2] + bary[1]*pos[1][2] + bary[2]*pos[2][2],
                ];
                img.put_pixel(x, y, pixel_fn(obj));
            }
        }
    }
}

fn barycentric(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> Option<[f32; 3]> {
    let edge = |p1: [f32; 2], p2: [f32; 2], q: [f32; 2]| -> f32 {
        (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0])
    };
    let denom = edge(a, b, c);
    if denom.abs() < f32::EPSILON { return None; }
    let wa = edge(b, c, p) / denom;
    let wb = edge(c, a, p) / denom;
    let wc = edge(a, b, p) / denom;
    if wa >= 0.0 && wb >= 0.0 && wc >= 0.0 { Some([wa, wb, wc]) } else { None }
}

// ── Noise ─────────────────────────────────────────────────────────────────────

fn fbm(p: [f32; 3], octaves: u32) -> f32 {
    let mut v = 0.0f32;
    let mut amp = 0.5f32;
    let mut q = p;
    for _ in 0..octaves {
        v += amp * value_noise3(q);
        q = [q[0] * 2.02, q[1] * 2.02, q[2] * 2.02];
        amp *= 0.5;
    }
    v
}

fn fbm_grad(p: [f32; 3], octaves: u32) -> [f32; 3] {
    let h = 0.005f32;
    let v = fbm(p, octaves);
    [
        (fbm([p[0]+h, p[1], p[2]], octaves) - v) / h,
        (fbm([p[0], p[1]+h, p[2]], octaves) - v) / h,
        (fbm([p[0], p[1], p[2]+h], octaves) - v) / h,
    ]
}

fn value_noise3(p: [f32; 3]) -> f32 {
    let ix = p[0].floor() as i32;
    let iy = p[1].floor() as i32;
    let iz = p[2].floor() as i32;
    let fx = p[0] - ix as f32;
    let fy = p[1] - iy as f32;
    let fz = p[2] - iz as f32;
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let uz = fz * fz * (3.0 - 2.0 * fz);

    let hash = |x: i32, y: i32, z: i32| -> f32 {
        let n = x.wrapping_mul(1619)
            .wrapping_add(y.wrapping_mul(31337))
            .wrapping_add(z.wrapping_mul(1013904223));
        let n = n.wrapping_mul(n.wrapping_mul(n).wrapping_mul(60493));
        (n as f32) / (i32::MAX as f32) * 0.5 + 0.5
    };
    let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);

    lerp(
        lerp(
            lerp(hash(ix,   iy,   iz), hash(ix+1, iy,   iz), ux),
            lerp(hash(ix,   iy+1, iz), hash(ix+1, iy+1, iz), ux),
            uy,
        ),
        lerp(
            lerp(hash(ix,   iy,   iz+1), hash(ix+1, iy,   iz+1), ux),
            lerp(hash(ix,   iy+1, iz+1), hash(ix+1, iy+1, iz+1), ux),
            uy,
        ),
        uz,
    )
}
