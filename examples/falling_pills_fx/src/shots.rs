//! The demo's falling-pills takes, cut every `SHOT_DURATION` seconds: camera pose, fov, depth of
//! field and pill direction per shot (values from the demo's cam.seq.ron, cues Crew2 and Build).

use pill_engine::{define_global_component, game::*};

use crate::falling_pills::FallingPillsDirection;

pub(crate) const SHOT_DURATION: f32 = 10.0;

struct Shot {
    camera_position: Vector3f,
    look_at: Vector3f,
    fov: f32,
    rising: bool,
    depth_of_field: DepthOfField,
}

// Build: side view; Crew2: from below the rain, looking up
const SIDE_POSITION: Vector3f = Vector3f::new(0.0, 1.8, -3.25);
const SIDE_LOOK_AT: Vector3f = Vector3f::new(0.0, 1.3, 0.0);
const SIDE_DEPTH_OF_FIELD: DepthOfField = DepthOfField {
    focus_distance: 1.5,
    focus_scale: 1.0,
    near_blur: 1.0,
    near_cut: 1.5,
    anamorphic_squeeze: 0.5,
};

const SHOTS: [Shot; 3] = [
    // Crew2: look up, pills fall towards the camera
    Shot {
        camera_position: Vector3f::new(0.46, -0.68, -1.2),
        look_at: Vector3f::new(0.48, 2.32, -1.16),
        fov: 38.0,
        rising: false,
        depth_of_field: DepthOfField {
            focus_distance: 2.5,
            focus_scale: 2.0,
            near_blur: 2.0,
            near_cut: 1.5,
            anamorphic_squeeze: 0.5,
        },
    },
    // Build, side view: pills fall
    Shot {
        camera_position: SIDE_POSITION,
        look_at: SIDE_LOOK_AT,
        fov: 35.0,
        rising: false,
        depth_of_field: SIDE_DEPTH_OF_FIELD,
    },
    // Build, side view: pills rise
    Shot {
        camera_position: SIDE_POSITION,
        look_at: SIDE_LOOK_AT,
        fov: 35.0,
        rising: true,
        depth_of_field: SIDE_DEPTH_OF_FIELD,
    },
];

define_global_component!(ShotCycle {
    // Index into SHOTS of the shot on screen; None before the first frame
    current: Option<usize>,
});

/// Cuts to the next shot every SHOT_DURATION seconds.
pub(crate) fn shot_system(engine: &mut Engine) -> Result<()> {
    let time = engine.get_global_component::<TimeComponent>()?.time;
    let index = (time / SHOT_DURATION) as usize % SHOTS.len();
    let cycle = engine.get_global_component_mut::<ShotCycle>()?;
    if cycle.current == Some(index) {
        return Ok(());
    }
    cycle.current = Some(index);
    let shot = &SHOTS[index];

    for (_, transform, camera) in
        engine.iterate_two_components_mut::<TransformComponent, CameraComponent>()?
    {
        transform.set_position(shot.camera_position);
        transform.look_at(shot.look_at);
        camera.fov = shot.fov;
    }
    engine
        .get_global_component_mut::<FallingPillsDirection>()?
        .rising = shot.rising;
    let render_state = engine.get_global_component_mut::<RenderStateComponent>()?;
    render_state.depth_of_field = shot.depth_of_field;
    // A cut: no motion blur between the two camera poses
    render_state.reset_motion_history();
    Ok(())
}
