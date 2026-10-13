use crate::{
    ecs::components::{GlobalComponent, GlobalComponentStorage},
    graphics::PassTextureHandle,
};

use pill_core::PillTypeMapKey;

/// HDR → display mapping applied by the tonemap pass.
#[derive(Clone, Copy, Debug)]
pub enum Tonemap {
    /// Linear pass-through; values above 1 clip.
    None,
    /// Extended Reinhard c·(1 + c/W²)/(1 + c): `white_point` W maps to 1.0; INFINITY = plain c/(1 + c).
    Reinhard { white_point: f32 },
}

/// Directional light; `direction` points from the light toward the scene.
#[derive(Clone, Copy, Debug)]
pub struct DirectionalLight {
    pub direction: [f32; 3],
    /// Linear RGB, multiplied by `intensity`.
    pub color: [f32; 3],
    pub intensity: f32,
}

/// Engine-owned (always present, cannot be removed or constructed by games); games set the
/// public environment, IBL, fog and tonemap fields.
pub struct RenderStateComponent {
    pub(crate) boot_done: bool,
    /// Scene color target (HDR_TARGET_FORMAT) shared by scene passes; resolved by the tonemap pass.
    pub(crate) hdr_target: PassTextureHandle,
    /// Scene depth target (DEPTH_TARGET_FORMAT) shared by scene passes.
    pub(crate) depth_target: PassTextureHandle,
    /// Engine default IBL maps (diffuse, specular, BRDF LUT): used when a game-set handle is invalid.
    pub(crate) default_ibl: [PassTextureHandle; 3],
    /// Environment (equirect) image drawn behind the scene; `None` draws `background_color`.
    pub background: Option<PassTextureHandle>,
    /// Solid background color used when `background` is `None` (linear RGB).
    pub background_color: [f32; 3],
    pub ibl_diffuse: PassTextureHandle,
    pub ibl_specular: PassTextureHandle,
    pub ibl_brdf_lut: PassTextureHandle,
    /// Exponential-squared fog: color is mixed toward `fog_color` by 1 - exp(-density² · distance²);
    /// density 0 disables fog.
    pub fog_color: [f32; 3],
    pub fog_density: f32,
    /// Scales both IBL terms (diffuse irradiance and specular reflection); 0 = direct light only.
    pub ibl_intensity: f32,
    /// Key, rim and bounce lights of the PBR passes.
    pub lights: [DirectionalLight; 3],
    pub tonemap: Tonemap,
}

impl RenderStateComponent {
    /// Creates the component in its pre-boot state; handles are replaced by `create_default_resources`.
    pub(crate) fn new() -> Self {
        Self {
            boot_done: false,
            hdr_target: PassTextureHandle::default(),
            depth_target: PassTextureHandle::default(),
            default_ibl: [PassTextureHandle::default(); 3],
            background: None,
            background_color: [0.15, 0.15, 0.15],
            ibl_diffuse: PassTextureHandle::default(),
            ibl_specular: PassTextureHandle::default(),
            ibl_brdf_lut: PassTextureHandle::default(),
            fog_color: [0.0, 0.0, 0.0],
            fog_density: 0.0,
            ibl_intensity: 1.0,
            // Key behind the camera upper-left, cool rim front upper-right, neutral bounce from below
            lights: [
                DirectionalLight {
                    direction: [0.38, -0.38, -0.84],
                    color: [1.0, 0.98, 0.95],
                    intensity: 2.2,
                },
                DirectionalLight {
                    direction: [-0.50, 0.50, 0.71],
                    color: [0.6, 0.65, 1.0],
                    intensity: 0.8,
                },
                DirectionalLight {
                    direction: [0.0, -1.0, 0.0],
                    color: [0.8, 0.8, 0.8],
                    intensity: 0.3,
                },
            ],
            tonemap: Tonemap::Reinhard {
                white_point: f32::INFINITY,
            },
        }
    }
}

impl RenderStateComponent {
    /// Scene color target (HDR_TARGET_FORMAT) for game-defined passes.
    pub fn hdr_target(&self) -> PassTextureHandle {
        self.hdr_target
    }

    /// Scene depth target (DEPTH_TARGET_FORMAT) for game-defined passes.
    pub fn depth_target(&self) -> PassTextureHandle {
        self.depth_target
    }
}

impl PillTypeMapKey for RenderStateComponent {
    type Storage = GlobalComponentStorage<RenderStateComponent>;
}

impl GlobalComponent for RenderStateComponent {}
