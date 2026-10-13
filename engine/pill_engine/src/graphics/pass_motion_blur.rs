use crate::config::{HDR_TARGET_FORMAT, MOTION_BLUR_FRAGMENT_SHADER};
use crate::ecs::RenderStateComponent;
use crate::graphics::fullscreen::{
    create_fullscreen_pipeline, draw_fullscreen, linear_clamp_sampler, sampler_entry, target_view,
    texture_entry, uniform_buffer, uniform_entry, Binding,
};
use crate::graphics::projection::depth_unpack;
use crate::graphics::{FrameTarget, Pass, PassTextureHandle, PillRenderer, Pipeline, WorldQuery};

use pill_core::Result;

/// Mirrors `MotionBlurParams` in motion_blur_fragment.hlsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MotionBlurParams {
    velocity_scale: f32,
    min_speed: f32,
    depth_softness: f32,
    max_samples: u32,
    depth_mul: f32,
    depth_add: f32,
    _padding: [f32; 2],
}

struct MotionBlurState {
    pipeline: Pipeline,
    sampler: wgpu::Sampler,
    params_buffer: wgpu::Buffer,
}

/// Blurs `color` along the per-pixel motion vectors of the PBR pass's velocity target, scaled by
/// `RenderStateComponent::motion_blur` (shutter fraction, 0 = copy).
pub struct PassMotionBlur {
    color: PassTextureHandle,
    velocity: PassTextureHandle,
    depth: PassTextureHandle,
    output: PassTextureHandle,
    /// Pixels; slower pixels are copied unblurred.
    pub min_speed: f32,
    /// Sample weight falloff per unit of view-space depth difference to the center pixel.
    pub depth_softness: f32,
    pub max_samples: u32,
    state: Option<MotionBlurState>,
}

impl PassMotionBlur {
    pub fn new(
        color: PassTextureHandle,
        velocity: PassTextureHandle,
        depth: PassTextureHandle,
        output: PassTextureHandle,
    ) -> Self {
        Self {
            color,
            velocity,
            depth,
            output,
            min_speed: 0.5,
            depth_softness: 2.0,
            max_samples: 16,
            state: None,
        }
    }
}

impl Pass for PassMotionBlur {
    fn get_label(&self) -> &str {
        "pass_motion_blur"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let pipeline = create_fullscreen_pipeline(
            renderer,
            "pass_motion_blur",
            MOTION_BLUR_FRAGMENT_SHADER,
            vec![
                texture_entry(0, true),
                texture_entry(1, true),
                texture_entry(2, false),
                uniform_entry(3),
                sampler_entry(4),
            ],
            HDR_TARGET_FORMAT,
            None,
        )?;
        self.state = Some(MotionBlurState {
            pipeline,
            sampler: linear_clamp_sampler(renderer),
            params_buffer: uniform_buffer::<MotionBlurParams>(renderer, "pass_motion_blur_params"),
        });
        Ok(())
    }

    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        _frame_target: &FrameTarget,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        let state = self
            .state
            .as_ref()
            .expect("PassMotionBlur: state not initialized — call init() before draw()");
        let render_state = world.get_global::<RenderStateComponent>()?;
        let (camera, _) = world.active_camera()?;
        let [depth_mul, depth_add] = depth_unpack(camera.range.start, camera.range.end);
        let params = MotionBlurParams {
            velocity_scale: render_state.motion_blur,
            min_speed: self.min_speed,
            depth_softness: self.depth_softness,
            max_samples: self.max_samples,
            depth_mul,
            depth_add,
            _padding: [0.0; 2],
        };
        renderer
            .get_queue()
            .write_buffer(&state.params_buffer, 0, bytemuck::bytes_of(&params));
        draw_fullscreen(
            encoder,
            renderer,
            "pass_motion_blur",
            &state.pipeline,
            &[
                Binding::Target(self.color),
                Binding::Target(self.velocity),
                Binding::Target(self.depth),
                Binding::Buffer(&state.params_buffer),
                Binding::Sampler(&state.sampler),
            ],
            target_view(renderer, self.output, "pass_motion_blur")?,
        )
    }
}
