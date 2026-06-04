use pill_engine::{define_component, game::*};

use crate::loader::{self, SponzaCpu};

define_component!(OrbitCamera {
    yaw: f32,
    pitch: f32,
    radius: f32,
});

pub struct Game {}

// On WASM the model is fetched asynchronously: the spawn_local task (which cannot hold &mut Engine,
// since the event loop owns it) deposits a finished CPU payload here, and `sponza_drain_system`
// picks it up on the next frame and performs the GPU uploads + entity build.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING: std::cell::RefCell<Option<SponzaCpu>> = const { std::cell::RefCell::new(None) };
    static DONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// Orbit around a point at mid-height of the atrium so the camera circles inside the hall.
const ORBIT_CENTER_Y: f32 = 5.0;

fn orbit_camera_system(engine: &mut Engine) -> Result<()> {
    let input = engine.get_global_component_mut::<InputComponent>()?;
    let mouse_delta = input.get_mouse_delta();
    let scroll_delta = input.get_mouse_scroll_delta();
    let left_mouse_button_held = input.get_mouse_button(MouseButton::Left);

    for (_, transform, orbit) in
        engine.iterate_two_components_mut::<TransformComponent, OrbitCamera>()?
    {
        if left_mouse_button_held {
            orbit.yaw -= mouse_delta.x * 0.3;
            orbit.pitch = (orbit.pitch + mouse_delta.y * 0.3).clamp(-80.0, 80.0);
        }
        orbit.radius = (orbit.radius - scroll_delta.y * 0.8).clamp(2.0, 40.0);

        let pitch_radians = orbit.pitch.to_radians();
        let yaw_radians = orbit.yaw.to_radians();
        let radius_cos_pitch = orbit.radius * pitch_radians.cos();
        transform.set_position(Vector3f::new(
            radius_cos_pitch * yaw_radians.sin(),
            orbit.radius * pitch_radians.sin() + ORBIT_CENTER_Y,
            radius_cos_pitch * yaw_radians.cos(),
        ));
    }
    Ok(())
}

// Drains the fetched payload into the engine once loading finishes (WASM only). Idempotent.
#[cfg(target_arch = "wasm32")]
fn sponza_drain_system(engine: &mut Engine) -> Result<()> {
    if DONE.with(|d| d.get()) {
        return Ok(());
    }
    let cpu = PENDING.with(|p| p.borrow_mut().take());
    if let Some(cpu) = cpu {
        let scene = engine.get_active_scene_handle()?;
        spawn_assets(engine, scene, cpu)?;
        DONE.with(|d| d.set(true));
    }
    Ok(())
}

// Uploads the CPU payload to the GPU and builds one entity per primitive. Shared by the native
// inline path and the WASM drain system.
fn spawn_assets(engine: &mut Engine, scene: SceneHandle, cpu: SponzaCpu) -> Result<()> {
    // --- IBL (background + diffuse/specular/brdf, same layout as pbr_helmet) ---
    let (equirect, eq_w, eq_h) = &cpu.equirect;
    let background = engine.create_gpu_texture_f32("sponza_equirect", equirect, *eq_w, *eq_h)?;
    let (diffuse, specular_mips, brdf_lut) = &cpu.ibl;
    let diffuse_handle = engine.create_gpu_texture_f32("sponza_diffuse_ibl", diffuse, 32, 16)?;
    let specular_handle =
        engine.create_gpu_mipped_texture_f32("sponza_specular_ibl", specular_mips, 128, 64)?;
    let brdf_handle = engine.create_gpu_texture_f32("sponza_brdf_lut", brdf_lut, 256, 256)?;
    {
        let render_state = engine.get_global_component_mut::<RenderStateComponent>()?;
        render_state.background = background;
        render_state.ibl_diffuse = diffuse_handle;
        render_state.ibl_specular = specular_handle;
        render_state.ibl_brdf_lut = brdf_handle;
    }

    // --- Textures ---
    let mut texture_handles: Vec<TextureHandle> = Vec::with_capacity(cpu.scene.textures.len());
    for texture in &cpu.scene.textures {
        texture_handles.push(engine.add_resource(Texture::from_bytes(
            &texture.name,
            texture.kind,
            &texture.rtex,
        ))?);
    }

    // --- Materials ---
    let mut material_handles: Vec<PBRMaterialHandle> =
        Vec::with_capacity(cpu.scene.materials.len());
    for material in &cpu.scene.materials {
        let mut pbr = PBRMaterial::new(&material.name)
            .albedo(Color::new(
                material.base_color[0],
                material.base_color[1],
                material.base_color[2],
            ))
            .metallic(material.metallic)
            .roughness(material.roughness);
        if let Some(i) = material.albedo {
            pbr = pbr.albedo_texture(texture_handles[i]);
        }
        if let Some(i) = material.normal {
            pbr = pbr.normal_texture(texture_handles[i]);
        }
        if let Some(i) = material.metallic_roughness {
            pbr = pbr.metallic_roughness_texture(texture_handles[i]);
        }
        if let Some(i) = material.emissive {
            pbr = pbr.emissive_texture(texture_handles[i]);
        }
        material_handles.push(engine.add_resource(pbr)?);
    }

    // --- Meshes + entities (world transform already baked into the vertices) ---
    for mesh in &cpu.scene.meshes {
        let mesh_handle =
            engine.add_resource(Mesh::from_cooked_mesh_bytes(&mesh.name, &mesh.rmsh)?)?;
        engine
            .build_entity(scene)
            .with_component(TransformComponent::builder().build())
            .with_component(
                PbrRenderableComponent::builder()
                    .mesh(&mesh_handle)
                    .pbr_material(&material_handles[mesh.material])
                    .build(),
            )
            .build();
    }

    log::info!("Sponza: uploaded {} meshes to GPU", cpu.scene.meshes.len());
    Ok(())
}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let scene = engine.create_scene("sponza")?;
        engine.set_active_scene(scene)?;

        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        engine.register_component::<PbrRenderableComponent>(scene)?;
        engine.register_component::<OrbitCamera>(scene)?;

        // Build the camera up front so the window renders (clear color + sky once IBL lands)
        // while the ~50 MB model downloads.
        engine
            .build_entity(scene)
            .with_component(
                TransformComponent::builder()
                    .position(Vector3f::new(-9.0, ORBIT_CENTER_Y, 0.0))
                    .build(),
            )
            .with_component(
                CameraComponent::builder()
                    .enabled(true)
                    .fov(60.0)
                    .clear_color(Color::new(0.02, 0.02, 0.03))
                    .look_at(Some(Vector3f::new(0.0, ORBIT_CENTER_Y, 0.0)))
                    .build(),
            )
            .with_component(OrbitCamera {
                yaw: 90.0,
                pitch: 5.0,
                radius: 9.0,
            })
            .build();

        engine.add_system("orbit_camera", orbit_camera_system)?;

        // Kick off the runtime fetch (the whole point of this example).
        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(async {
                match loader::load_wasm().await {
                    Ok(cpu) => PENDING.with(|p| *p.borrow_mut() = Some(cpu)),
                    Err(e) => log::error!("Sponza load failed: {e}"),
                }
            });
            engine.add_system("sponza_drain", sponza_drain_system)?;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            log::info!("Sponza: loading from {}", loader::base_url());
            let cpu = loader::load_native().map_err(PillError::from)?;
            spawn_assets(engine, scene, cpu)?;
        }

        Ok(())
    }
}
