//! Falling pills (from the demo's "Build" shot) rendered through `PassChain::PostFx`: a rain of
//! clear-coated pills around a slowly turning hero pill, with fog, an emissive orb and the
//! post-processing passes configured in `RenderStateComponent`.

use pill_engine::{define_component, game::*};

use crate::falling_pills::{falling_pills_system, spawn_falling_pills, FallingPill};
use crate::mesh::{make_pill_mesh_bytes, make_sphere_mesh_bytes};
use crate::tex_gen;

// Deep navy, also the fog color, so distant pills dissolve into the background
const BACKGROUND_COLOR: [f32; 3] = [0.003, 0.004, 0.0175];
const CAMERA_POSITION: Vector3f = Vector3f::new(0.0, 1.8, -3.25);
const HERO_POSITION: Vector3f = Vector3f::new(0.0, 1.3, 0.0);

define_component!(HeroPill {
    spin: f32, // degrees/s around Y
});

// Slow sideways sway around the hero pill, so camera motion feeds motion blur
define_component!(SwayCamera {
    amplitude: f32, // units along X
    frequency: f32, // radians/s
});

pub struct Game {}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        engine.toggle_system("audio_system", UpdatePhase::PostGame, false)?;

        let scene = engine.create_scene("falling_pills")?;
        engine.set_active_scene(scene)?;
        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<MeshRenderingComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        engine.register_component::<FallingPill>(scene)?;
        engine.register_component::<HeroPill>(scene)?;
        engine.register_component::<SwayCamera>(scene)?;
        engine.add_system("falling_pills", falling_pills_system)?;
        engine.add_system("hero_pill", hero_pill_system)?;
        engine.add_system("sway_camera", sway_camera_system)?;

        setup_render_state(engine)?;

        // --- Resources ---
        let pill_mesh = engine.add_resource(Mesh::from_cooked_mesh_bytes(
            "pill",
            &make_pill_mesh_bytes(
                tex_gen::CAP_STACKS,
                tex_gen::SLICES,
                tex_gen::HEIGHT,
                tex_gen::EXTRUDE,
            ),
        )?)?;
        let sphere_mesh = engine.add_resource(Mesh::from_cooked_mesh_bytes(
            "sphere",
            &make_sphere_mesh_bytes(24, 32),
        )?)?;
        let floor_mesh = engine.add_resource(Mesh::cube("floor", 1.0))?;

        let pill_albedo = engine.add_resource(Texture::from_bytes_owned(
            "pill_albedo",
            TextureType::Color,
            tex_gen::albedo(),
        ))?;
        let pill_normal = engine.add_resource(Texture::from_bytes_owned(
            "pill_normal",
            TextureType::Normal,
            tex_gen::normal_map(),
        ))?;
        let pill_roughness = engine.add_resource(Texture::from_bytes_owned(
            "pill_roughness",
            TextureType::MetallicRoughness,
            tex_gen::roughness(),
        ))?;

        // The demo's pill: metallic, greasy roughness map, glossy clear coat on top
        let pill_material = engine.add_resource(
            Material::new("pill")
                .albedo_texture(pill_albedo)
                .normal_texture(pill_normal)
                .metallic_roughness_texture(pill_roughness)
                .metallic(1.0)
                .roughness(1.0)
                .clearcoat(0.3, 0.1),
        )?;
        // Every 8th falling pill is a frosted glass shell, drawn by the translucent pass
        let glass_material = engine.add_resource(
            Material::new("glass_pill")
                .albedo(Color::new(0.75, 0.85, 1.0))
                .roughness(0.15)
                .clearcoat(1.0, 0.05)
                .opacity(0.35),
        )?;
        let orb_material = engine.add_resource(
            Material::new("orb")
                .albedo(Color::new(0.0, 0.0, 0.0))
                .emissive(Color::new(40.0, 22.0, 9.0)),
        )?;
        let floor_material = engine.add_resource(
            Material::new("floor")
                .albedo(Color::new(0.05, 0.05, 0.06))
                .roughness(0.35),
        )?;

        // --- Entities ---
        engine
            .build_entity(scene)
            .with_component(
                TransformComponent::builder()
                    .position(HERO_POSITION)
                    .rotation(Vector3f::new(0.0, 0.0, 70.0))
                    .scale(Vector3f::splat(0.3))
                    .build(),
            )
            .with_component(
                MeshRenderingComponent::builder()
                    .mesh(&pill_mesh)
                    .material(&pill_material)
                    .build(),
            )
            .with_component(HeroPill { spin: 25.0 })
            .build();

        spawn_falling_pills(engine, scene, &pill_mesh, |index| {
            if index % 8 == 0 {
                glass_material
            } else {
                pill_material
            }
        });

        // Emissive orb: the bloom source
        engine
            .build_entity(scene)
            .with_component(
                TransformComponent::builder()
                    .position(Vector3f::new(-0.9, 2.0, 1.4))
                    .scale(Vector3f::splat(0.06))
                    .build(),
            )
            .with_component(
                MeshRenderingComponent::builder()
                    .mesh(&sphere_mesh)
                    .material(&orb_material)
                    .build(),
            )
            .build();

        // Floor the pills fall through: contact shadows for SSAO
        engine
            .build_entity(scene)
            .with_component(
                TransformComponent::builder()
                    .position(Vector3f::new(0.0, -1.0, 0.0))
                    .scale(Vector3f::new(60.0, 0.02, 60.0))
                    .build(),
            )
            .with_component(
                MeshRenderingComponent::builder()
                    .mesh(&floor_mesh)
                    .material(&floor_material)
                    .build(),
            )
            .build();

        let mut camera_transform = TransformComponent::builder()
            .position(CAMERA_POSITION)
            .build();
        camera_transform.look_at(HERO_POSITION);
        engine
            .build_entity(scene)
            .with_component(camera_transform)
            .with_component(
                CameraComponent::builder()
                    .enabled(true)
                    .fov(38.0)
                    .range(0.05..60.0)
                    .build(),
            )
            .with_component(SwayCamera {
                amplitude: 0.8,
                frequency: 0.5,
            })
            .build();

        Ok(())
    }
}

/// Post-processing chain, studio IBL, lights and per-pass parameters.
fn setup_render_state(engine: &mut Engine) -> Result<()> {
    let diffuse = engine.load_cooked_texture(
        "studio_diffuse_ibl",
        include_bytes!(concat!(env!("OUT_DIR"), "/ibl_diffuse.rtex")),
    )?;
    let specular = engine.load_cooked_texture(
        "studio_specular_ibl",
        include_bytes!(concat!(env!("OUT_DIR"), "/ibl_specular.rtex")),
    )?;

    let render_state = engine.get_global_component_mut::<RenderStateComponent>()?;
    render_state.pass_chain = PassChain::PostFx;
    render_state.ibl_diffuse = diffuse;
    render_state.ibl_specular = specular;
    render_state.ibl_intensity = 0.5;
    render_state.background = None;
    render_state.background_color = BACKGROUND_COLOR;
    render_state.fog_color = BACKGROUND_COLOR;
    render_state.fog_density = 0.174;
    // Camera looks +Z: warm key from upper camera-left, cool rim from behind, faint bounce from below
    render_state.lights = [
        DirectionalLight {
            direction: [0.45, -0.55, 0.70],
            color: [1.0, 0.92, 0.82],
            intensity: 2.4,
        },
        DirectionalLight {
            direction: [-0.30, -0.35, -0.89],
            color: [0.45, 0.55, 0.8],
            intensity: 1.6,
        },
        DirectionalLight {
            direction: [0.0, 1.0, 0.0],
            color: [0.8, 0.9, 1.0],
            intensity: 0.15,
        },
    ];
    // Post-processing values of the demo's Crew2 shot (half-shutter motion blur, soft glow,
    // slight pincushion, corner fringing, cool-shadow grade, fine grain)
    render_state.tonemap = Tonemap::lottes();
    render_state.motion_blur = 0.5;
    // Contact shadows where pills pass over the floor (darkens the IBL term only)
    render_state.ssao = Ssao {
        strength: 1.0,
        radius: 0.5,
        intensity: 1.5,
        power: 1.5,
    };
    // Focus on the hero pill; pills between it and the camera fall into the near layer
    render_state.depth_of_field = DepthOfField {
        focus_distance: 3.3,
        focus_scale: 2.0,
        near_blur: 2.0,
        near_cut: 0.5,
        anamorphic_squeeze: 0.5,
    };
    render_state.bloom = Bloom {
        strength: 0.04,
        radius: 0.005,
    };
    render_state.lens = Lens {
        distortion: 0.01,
        chromatic_aberration: 0.002,
        vignette: 0.6,
        grain_strength: 0.00375,
        grade_gamma: [1.0, 1.0, 0.8],
        grade_edge0: -0.05,
        grade_edge1: 0.9,
        grade_mix: 1.0,
        ..Lens::default()
    };
    Ok(())
}

fn hero_pill_system(engine: &mut Engine) -> Result<()> {
    let delta_time = engine.get_global_component::<TimeComponent>()?.delta_time;
    for (_, transform, hero) in
        engine.iterate_two_components_mut::<TransformComponent, HeroPill>()?
    {
        let mut rotation = transform.rotation;
        rotation.y += hero.spin * delta_time;
        transform.set_rotation(rotation);
    }
    Ok(())
}

fn sway_camera_system(engine: &mut Engine) -> Result<()> {
    let time = engine.get_global_component::<TimeComponent>()?.time;
    for (_, transform, sway) in
        engine.iterate_two_components_mut::<TransformComponent, SwayCamera>()?
    {
        let offset = sway.amplitude * (sway.frequency * time).sin();
        transform.set_position(CAMERA_POSITION + Vector3f::new(offset, 0.0, 0.0));
        transform.look_at(HERO_POSITION);
    }
    Ok(())
}
