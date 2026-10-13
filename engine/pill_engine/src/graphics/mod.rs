#![cfg_attr(debug_assertions, allow(dead_code, unused_imports))]

#[cfg(feature = "headless")]
mod dummy_renderer;
mod fullscreen;
mod pass_bloom;
#[cfg(feature = "ui")]
mod pass_egui;
mod pass_fxaa;
mod pass_lens;
mod pass_motion_blur;
mod pass_pbr_opaque;
mod pass_tonemap;
mod projection;
mod render_queue;
mod renderer;
mod scene_uniform;

// --- Use ---

pub use renderer::{
    camera_forward, camera_view_projection, BufferDesc, FrameTarget, Pass, PassTextureHandle,
    PillRenderer, Pipeline, PipelineDesc, Renderer, RendererMaterialHandle, RendererMeshHandle,
    RendererTargetDesc, RendererTextureHandle, ShaderDesc, TargetSize, TextureData, WorldQuery,
};

#[cfg(feature = "headless")]
pub use self::dummy_renderer::DummyRenderer;
pub use pass_bloom::PassBloom;
#[cfg(feature = "ui")]
pub use pass_egui::PassEgui;
pub use pass_fxaa::PassFxaa;
pub use pass_lens::PassLens;
pub use pass_motion_blur::PassMotionBlur;
pub use pass_pbr_opaque::{PassPBROpaque, PassPBRTranslucent};
pub use pass_tonemap::PassTonemap;
pub(crate) use scene_uniform::SceneUniform;

pub use render_queue::{
    compose_render_queue_key, decompose_render_queue_key, RenderQueueKey, RenderQueueKeyFields,
    RENDER_QUEUE_KEY_MESH_INDEX,
};
