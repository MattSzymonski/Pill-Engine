use crate::{
    config::RENDERING_SYSTEM,
    ecs::{
        CameraAspectRatio, CameraComponent, EntityHandle, PbrRenderableComponent,
        RenderStateComponent, TransformComponent,
    },
    engine::Engine,
    graphics::{PassPBRStatic, PassTonemap, RenderQueueItem, RendererTargetDesc},
};

use pill_core::{warn, EngineError, LogContext, PillSlotMapKey, PillStyle, RendererError, Timer};
use wgpu;

use pill_core::{ErrorContext, Result};
use web_time::Instant;

pub fn rendering_system(engine: &mut Engine) -> Result<()> {
    let mut timer = Timer::new();
    timer.begin_context("rendering_system update");

    // First-frame bootstrap: install default pass chain
    let boot_done = engine
        .get_global_component::<RenderStateComponent>()?
        .boot_done;

    if !boot_done {
        let (bg, diff, spec, lut, bg_color, fog_density) = {
            let rs = engine.get_global_component::<RenderStateComponent>()?;
            (
                rs.background,
                rs.ibl_diffuse,
                rs.ibl_specular,
                rs.ibl_brdf_lut,
                rs.bg_color,
                rs.fog_density,
            )
        };
        let (w, h) = engine.renderer.get_surface_size();
        let hdr = engine.renderer.create_render_target(RendererTargetDesc {
            name: "hdr_target".to_string(),
            format: wgpu::TextureFormat::Rgba16Float,
            width: w,
            height: h,
        })?;
        #[cfg_attr(not(feature = "ui"), allow(unused_mut))]
        let mut passes: Vec<Box<dyn crate::graphics::Pass>> = vec![
            Box::new(
                PassPBRStatic::new(Some(hdr))
                    .with_background(bg, bg_color)
                    .with_ibl(diff, spec, lut)
                    .with_fog(bg_color, fog_density),
            ),
            Box::new(PassTonemap::new(hdr)),
        ];
        #[cfg(feature = "ui")]
        {
            use crate::{ecs::EguiComponent, graphics::PassEgui};
            let client = engine
                .get_global_component::<EguiComponent>()?
                .egui_client
                .clone();
            let window = engine.renderer.get_window();
            passes.push(Box::new(PassEgui::new(window, client)));
        }
        engine.renderer.set_passes(passes)?;
        engine
            .get_global_component_mut::<RenderStateComponent>()?
            .boot_done = true;
        return Ok(());
    }

    timer.record("Get active camera");

    let active_scene_handle = engine.scene_manager.get_active_scene_handle()?;
    let mut active_camera_entity_handle_result: Option<EntityHandle> = None;

    {
        let active_scene = engine.scene_manager.get_active_scene_mut()?;

        for (entity_handle, camera_component) in
            active_scene.get_one_component_iterator_mut::<CameraComponent>()?
        {
            if camera_component.enabled {
                if let CameraAspectRatio::Automatic(_) = camera_component.aspect {
                    let aspect_ratio =
                        engine.window_size.width as f32 / engine.window_size.height as f32;
                    camera_component.aspect = CameraAspectRatio::Automatic(aspect_ratio);
                }
                active_camera_entity_handle_result = Some(entity_handle);
                break;
            }
        }
    }

    let active_camera_entity_handle =
        active_camera_entity_handle_result.ok_or_else(|| -> pill_core::PillError {
            pill_core::PillError::from(EngineError::NoActiveCamera)
        })?;

    timer.record("Clear render queue");

    engine.render_queue.clear();
    engine.render_queue.reserve(200000);

    timer.record("Prepare render queue");

    let mut _matrix_calculation_duration: f32 = 0.0;
    let mut add_to_render_queue_duration: f32 = 0.0;

    // GPU fast-path: copy raw pos/rot/scale during the cache-warm sequential scan — NO trig,
    // no matrix build. The vertex shader composes model = T·R·S on the GPU. [Aaltonen GDC 2023]
    for (entity_handle, transform, pbr_renderable_component) in engine
        .scene_manager
        .get_two_component_iterator_mut::<TransformComponent, PbrRenderableComponent>(
        active_scene_handle,
    )? {
        let add_to_render_queue_start_time = Instant::now();
        if let Some(render_queue_key) = pbr_renderable_component.render_queue_key {
            let p = transform.position;
            let r = transform.rotation;
            let s = transform.scale;
            engine.render_queue.push(RenderQueueItem {
                key: render_queue_key,
                entity_index: entity_handle.data().index,
                position: [p.x, p.y, p.z, 0.0],
                rotation: [r.x, r.y, r.z, 0.0],
                scale: [s.x, s.y, s.z, 0.0],
            });
        } else {
            warn!(LogContext::Rendering => "Invalid render queue key");
            continue;
        }
        add_to_render_queue_duration +=
            add_to_render_queue_start_time.elapsed().as_secs_f32() * 1000.0;
    }

    timer.record(format!(
        "Matrix calculation {} ms",
        _matrix_calculation_duration
    ));
    timer.record(format!(
        "Add to render queue {} ms",
        add_to_render_queue_duration
    ));

    timer.record("Sort render queue");

    // pdqsort by key — measured faster here than hand-rolled radix/counting sort, because the
    // scene has few distinct keys and the builtin handles many-equal-elements partitioning well.
    engine.render_queue.sort_unstable_by_key(|item| item.key);

    timer.record("Get component storages");

    let active_scene = engine.scene_manager.get_active_scene_mut()?;
    let camera_component_storage = active_scene
        .get_component_storage::<CameraComponent>()
        .context(format!(
            "{}: Cannot get active {}",
            "rendering_system".specific_object_style(),
            "Camera".general_object_style()
        ))?;
    let transform_component_storage = active_scene
        .get_component_storage::<TransformComponent>()
        .context(format!(
            "{}: Cannot get {}",
            "rendering_system".specific_object_style(),
            "TransformComponents".specific_object_style()
        ))
        .unwrap();

    timer.begin_context("Render");

    // Render
    let delta_time = engine.frame_delta_time;

    let render_result = engine.renderer.render(
        active_camera_entity_handle,
        &engine.render_queue,
        camera_component_storage,
        transform_component_storage,
        delta_time,
        &mut timer,
        &engine.resource_manager,
    );
    match render_result {
        Ok(_) => {
            timer.end_context()?; // end "Render"
            timer.end_context()?; // end "rendering_system update"
            engine.system_manager.update_system_timer(
                RENDERING_SYSTEM.name,
                RENDERING_SYSTEM.update_phase,
                timer,
            )?;
            Ok(())
        }
        Err(error) => match error.downcast_ref::<RendererError>() {
            Some(RendererError::SurfaceLost) => {
                timer.end_context()?; // end "Render"
                timer.end_context()?; // end "rendering_system update"
                engine.system_manager.update_system_timer(
                    RENDERING_SYSTEM.name,
                    RENDERING_SYSTEM.update_phase,
                    timer,
                )?;
                engine.renderer.resize(engine.window_size);
                Ok(())
            }
            Some(RendererError::SurfaceOutOfMemory) => {
                panic!("Critical: Renderer error, system out of memory");
            }
            _ => Err(error),
        },
    }
}
