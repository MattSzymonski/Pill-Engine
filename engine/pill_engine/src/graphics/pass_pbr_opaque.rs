use crate::config::{
    AMBIENT_TARGET_FORMAT, BACKGROUND_FRAGMENT_SHADER, BACKGROUND_VERTEX_SHADER,
    DEFAULT_EQUIRECT_FALLBACK_PIXEL, DEPTH_TARGET_FORMAT, HDR_TARGET_FORMAT,
    PBR_OPAQUE_FRAGMENT_SHADER, PBR_OPAQUE_VERTEX_SHADER, VELOCITY_TARGET_FORMAT,
};
use crate::{
    ecs::{
        CameraComponent, ComponentStorage, MeshRenderingComponent, RenderStateComponent,
        TransformComponent,
    },
    graphics::{
        camera_forward, camera_view_projection, decompose_render_queue_key, BufferDesc,
        FrameTarget, Pass, PassTextureHandle, PillRenderer, Pipeline, PipelineDesc, RenderQueueKey,
        RendererMaterialHandle, RendererMeshHandle, SceneUniform, ShaderDesc, TextureData,
        WorldQuery,
    },
    renderer::resources::{
        material_params_layout_entries, material_textures_layout_entries, RendererMaterial,
        RendererMesh, Vertex,
    },
};
use pill_core::{error, LogContext, Matrix4f, PillSlotMapKey, Result, Vector3f};
use std::num::NonZeroU32;

/// Camera uniform layout: position (vec4) + view-projection matrices (mat4x4) of this and the
/// previous frame (motion vectors).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    position: [f32; 4],
    view_projection_matrix: [[f32; 4]; 4],
    previous_view_projection_matrix: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            position: [0.0; 4],
            view_projection_matrix: Matrix4f::IDENTITY.to_cols_array_2d(),
            previous_view_projection_matrix: Matrix4f::IDENTITY.to_cols_array_2d(),
        }
    }

    /// Recomputes view and projection from ECS components; called once per frame before GPU upload.
    fn update_data(
        &mut self,
        camera_component: &CameraComponent,
        transform_component: &TransformComponent,
    ) {
        self.position = [
            transform_component.position.x,
            transform_component.position.y,
            transform_component.position.z,
            0.0,
        ];

        self.view_projection_matrix =
            camera_view_projection(camera_component, transform_component).to_cols_array_2d();
    }
}

/// Raw transform of one draw: position, rotation (radians), scale; xyz of each row.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DrawTransform {
    position: [f32; 4],
    rotation: [f32; 4],
    scale: [f32; 4],
}

impl DrawTransform {
    fn from(transform: &TransformComponent) -> Self {
        Self {
            position: [
                transform.position.x,
                transform.position.y,
                transform.position.z,
                0.0,
            ],
            // TransformComponent stores degrees; the vertex shader expects radians
            rotation: [
                transform.rotation.x.to_radians(),
                transform.rotation.y.to_radians(),
                transform.rotation.z.to_radians(),
                0.0,
            ],
            scale: [transform.scale.x, transform.scale.y, transform.scale.z, 0.0],
        }
    }
}

/// Per-draw storage entry (96 B): this and last frame's transform. GPU builds model = T·R·S, MVP
/// and the motion vector.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PerDrawStd140 {
    current: DrawTransform,
    previous: DrawTransform,
}

/// Last frame's transform of an entity slot and the frame it was drawn in.
#[derive(Clone, Copy)]
struct PreviousTransform {
    transform: DrawTransform,
    frame: u64,
}

// Per-draw data lives in one storage buffer indexed by instance id (no per-draw dynamic offsets).
const PER_DRAW_STRIDE_BYTES: usize = std::mem::size_of::<PerDrawStd140>();

pub const MATERIAL_BIND_GROUP_GLOBALS: usize = 0;
pub const MATERIAL_BIND_GROUP_TEXTURES: usize = 1;
pub const MATERIAL_BIND_GROUP_PARAMS: usize = 2;
pub const MATERIAL_BIND_GROUP_PERDRAW: usize = 3;

/// One instanced draw: a run of equal render keys (same material and mesh) in the sorted
/// per-draw array; instances `first..first + count` of the storage buffer.
struct DrawBatch {
    material_handle: RendererMaterialHandle,
    mesh_handle: RendererMeshHandle,
    first: u32,
    count: u32,
}

/// Camera uniform for the background sub-draw (view direction basis + fov, and last frame's
/// view-projection for the sky's motion vectors).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BgCameraUbo {
    right: [f32; 3],
    tan_half_fov: f32,
    up: [f32; 3],
    aspect: f32,
    fwd: [f32; 3],
    _pad: f32,
    bg_color: [f32; 3],
    _pad2: f32,
    previous_view_projection: [[f32; 4]; 4],
}

/// GPU-side state for the background sub-draw within the PBR render pass.
struct BgSubState {
    pipeline: Pipeline,
    camera_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    // Equirect handle the current bind_group was built from; rebuild only when it changes.
    cached_bg: PassTextureHandle,
    // White 1×1 equirect drawn (tinted by background_color) when no environment image is set
    fallback_bg: PassTextureHandle,
}

/// GPU-side pass state initialized in `Pass::init`, read every `Pass::draw`.
struct PassPBROpaqueState {
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    scene_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    pipeline: Pipeline,
    ibl_sampler: wgpu::Sampler,
    prefilter_sampler: wgpu::Sampler,
    brdf_lut_sampler: wgpu::Sampler,
    per_draw_buffer: wgpu::Buffer,
    per_draw_bind_group: wgpu::BindGroup,
    bg: BgSubState,
    // IBL handles the current globals_bind_group was built from; rebuild only when these change.
    cached_ibl: [PassTextureHandle; 3],
}

/// View of a game-set RenderStateComponent texture; an invalid handle logs and uses `fallback`.
/// Callers cache by handle, so the error is logged once per handle value.
fn view_or_fallback(
    renderer: &dyn PillRenderer,
    handle: PassTextureHandle,
    fallback: PassTextureHandle,
    label: &str,
) -> Result<wgpu::TextureView> {
    if let Some(view) = renderer.get_texture_view(handle) {
        return Ok(view);
    }
    error!(LogContext::Rendering => "RenderStateComponent.{} is not a valid texture; using the default", label);
    renderer
        .get_texture_view(fallback)
        .ok_or_else(|| format!("PassPBROpaque: default {label} texture missing").into())
}

/// Builds the globals bind group (camera UBO + IBL irradiance/prefilter/BRDF-LUT views + scene UBO).
/// Shared by init() and the per-frame rebuild-on-change path in draw().
#[allow(clippy::too_many_arguments)]
fn build_globals_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera_buffer: &wgpu::Buffer,
    scene_buffer: &wgpu::Buffer,
    irradiance_view: &wgpu::TextureView,
    prefilter_view: &wgpu::TextureView,
    brdf_lut_view: &wgpu::TextureView,
    ibl_sampler: &wgpu::Sampler,
    prefilter_sampler: &wgpu::Sampler,
    brdf_lut_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pass_pbr_opaque_globals_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(irradiance_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(ibl_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(prefilter_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(prefilter_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(brdf_lut_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(brdf_lut_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: scene_buffer.as_entire_binding(),
            },
        ],
    })
}

/// Builds the background bind group (camera UBO + equirect view + sampler).
fn build_bg_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera_buffer: &wgpu::Buffer,
    equirect_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pass_pbr_opaque_bg_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(equirect_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Default PBR pass: GGX + IBL, one instanced draw per material/mesh batch.
/// Background, IBL and fog are read from RenderStateComponent every frame.
/// Optional extra outputs: screen-space motion vectors (`with_velocity_target`, UV units per frame)
/// and the indirect (IBL) term alone (`with_ambient_target`, for SSAO).
pub struct PassPBROpaque {
    // Translucent mode (PassPBRTranslucent): draws only materials with opacity < 1, back to front,
    // blended over the color target, depth-tested against the opaque depth without writing it
    translucent: bool,
    // Opaque mode with a PassPBRTranslucent after it: leave opacity < 1 materials to that pass
    exclude_translucent: bool,
    color_target: PassTextureHandle,
    depth_target: PassTextureHandle,
    velocity_target: Option<PassTextureHandle>,
    ambient_target: Option<PassTextureHandle>,
    // Per-draw storage capacity; at most one instance per entity
    max_instances: usize,
    // [diffuse, specular, brdf_lut] for the first frame's bind group
    initial_ibl: [PassTextureHandle; 3],
    capacity_warned: bool,
    // Motion history: last frame's transform per entity slot and camera view-projection
    frame: u64,
    previous_transforms: Vec<Option<PreviousTransform>>,
    previous_view_projection: Option<Matrix4f>,
    motion_history_epoch: u32,
    // Per-frame scratch, capacity reused across frames; f32 = squared camera distance (translucent sort)
    draws: Vec<(RenderQueueKey, f32, PerDrawStd140)>,
    batches: Vec<DrawBatch>,
    staging_buffer: Vec<u8>,
    state: Option<PassPBROpaqueState>,
}

impl PassPBROpaque {
    /// `max_instances` sizes the per-draw buffer; `initial_ibl` ([diffuse, specular, brdf_lut]) are
    /// the engine defaults, also the fallback for invalid game handles. Call `Pass::init` before drawing.
    pub fn new(
        color_target: PassTextureHandle,
        depth_target: PassTextureHandle,
        max_instances: usize,
        initial_ibl: [PassTextureHandle; 3],
    ) -> Self {
        let max_instances = max_instances.max(1);
        Self {
            translucent: false,
            exclude_translucent: false,
            color_target,
            depth_target,
            velocity_target: None,
            ambient_target: None,
            max_instances,
            initial_ibl,
            capacity_warned: false,
            frame: 0,
            previous_transforms: Vec::new(),
            previous_view_projection: None,
            motion_history_epoch: 0,
            draws: Vec::with_capacity(max_instances),
            batches: Vec::new(),
            staging_buffer: Vec::with_capacity(max_instances * PER_DRAW_STRIDE_BYTES),
            state: None,
        }
    }

    pub fn max_instances(&self) -> usize {
        self.max_instances
    }

    pub fn initial_ibl(&self) -> [PassTextureHandle; 3] {
        self.initial_ibl
    }

    /// Skips materials with opacity < 1, for chains where a `PassPBRTranslucent` draws them.
    pub fn without_translucent(mut self) -> Self {
        self.exclude_translucent = true;
        self
    }

    /// Also writes motion vectors (VELOCITY_TARGET_FORMAT): UV offset from last frame's position.
    pub fn with_velocity_target(mut self, target: PassTextureHandle) -> Self {
        self.velocity_target = Some(target);
        self
    }

    /// Also writes the fogged IBL term alone (AMBIENT_TARGET_FORMAT), which SSAO darkens.
    pub fn with_ambient_target(mut self, target: PassTextureHandle) -> Self {
        self.ambient_target = Some(target);
        self
    }

    /// Color targets in shader output order (color, velocity, ambient); absent outputs are `None`.
    fn color_targets(&self, blend: wgpu::BlendState) -> [Option<wgpu::ColorTargetState>; 3] {
        [
            Some(wgpu::ColorTargetState {
                format: HDR_TARGET_FORMAT,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            self.velocity_target.map(|_| wgpu::ColorTargetState {
                format: VELOCITY_TARGET_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            self.ambient_target.map(|_| wgpu::ColorTargetState {
                format: AMBIENT_TARGET_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ]
    }
}

/// Returns the initialized pass state; panics in debug if `init` was not called.
fn get_state(pass: &mut PassPBROpaque) -> &mut PassPBROpaqueState {
    debug_assert!(pass.state.is_some());
    pass.state
        .as_mut()
        .expect("PassPBROpaque: state not initialized — call init() before draw()")
}

/// Whether the material of a render key has opacity < 1; a missing material counts as opaque.
fn is_translucent(world: &WorldQuery<'_>, key: RenderQueueKey) -> bool {
    let key_fields = decompose_render_queue_key(key);
    let Some(version) = NonZeroU32::new(key_fields.material_version.into()) else {
        return false;
    };
    let handle = RendererMaterialHandle::new(key_fields.material_index.into(), version);
    world
        .resources
        .get_resource::<RendererMaterial>(&handle)
        .is_ok_and(|material| material.translucent)
}

/// Translucent PBR: materials with opacity < 1, sorted back to front and alpha-blended over the
/// HDR target after the opaque pass, depth-tested against (not writing) the opaque depth. Shares
/// PassPBROpaque's shaders; writes no motion vectors.
pub struct PassPBRTranslucent(PassPBROpaque);

impl PassPBRTranslucent {
    /// Same targets and IBL defaults as the opaque pass it follows.
    pub fn new(
        color_target: PassTextureHandle,
        depth_target: PassTextureHandle,
        max_instances: usize,
        initial_ibl: [PassTextureHandle; 3],
    ) -> Self {
        let mut pass = PassPBROpaque::new(color_target, depth_target, max_instances, initial_ibl);
        pass.translucent = true;
        Self(pass)
    }
}

impl Pass for PassPBRTranslucent {
    fn get_label(&self) -> &str {
        self.0.get_label()
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        self.0.init(renderer)
    }

    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        frame_target: &FrameTarget,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        self.0.draw(encoder, renderer, frame_target, world)
    }
}

impl Pass for PassPBROpaque {
    fn get_label(&self) -> &str {
        if self.translucent {
            "pass_pbr_translucent"
        } else {
            "pass_pbr_opaque"
        }
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let bind_groups: Vec<Vec<wgpu::BindGroupLayoutEntry>> = vec![
            // 0: globals (camera + IBL irradiance + prefilter + BRDF LUT + scene)
            vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            // 1: material textures (base_color, normal, metallic_roughness, emissive)
            material_textures_layout_entries(),
            // 2: material params
            material_params_layout_entries(),
            // 3: per-draw storage array — no dynamic offset, no 256-B Metal alignment (Apple Metal Spec §4.1.1).
            vec![wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        ];

        let color_targets = self.color_targets(if self.translucent {
            wgpu::BlendState::ALPHA_BLENDING
        } else {
            wgpu::BlendState::REPLACE
        });
        let desc = PipelineDesc {
            label: Some("pass_pbr_opaque_pipeline"),
            vs: ShaderDesc {
                source: PBR_OPAQUE_VERTEX_SHADER,
                entry_func: "vs_main",
            },
            ps: ShaderDesc {
                source: PBR_OPAQUE_FRAGMENT_SHADER,
                entry_func: "fs_main",
            },
            vertex_buffers: &[RendererMesh::data_layout_descriptor()],
            bind_groups,
            targets: &color_targets,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_TARGET_FORMAT,
                depth_write_enabled: !self.translucent,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
                unclipped_depth: false,
            },
        };

        let pipeline = renderer.create_pipeline(desc)?;

        let camera_buffer = {
            let device = renderer.get_device();
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pass_pbr_opaque_camera_buffer"),
                size: std::mem::size_of::<CameraUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let scene_buffer = renderer.create_buffer(BufferDesc {
            label: Some("pass_pbr_opaque_scene_buffer"),
            byte_size: std::mem::size_of::<SceneUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })?;

        let per_draw_buffer = renderer.create_buffer(BufferDesc {
            label: Some("pass_pbr_opaque_per_draw"),
            byte_size: (PER_DRAW_STRIDE_BYTES * self.max_instances) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })?;
        let per_draw_bind_group = {
            let layout = &pipeline.bind_group_layouts[MATERIAL_BIND_GROUP_PERDRAW];
            let device = renderer.get_device();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pass_pbr_opaque_per_draw_bind_group"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: per_draw_buffer.as_entire_binding(),
                }],
            })
        };

        let ibl_sampler = {
            let device = renderer.get_device();
            device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            })
        };

        // Initial IBL bind group from the handles given at construction; draw() rebuilds it when
        // RenderStateComponent's handles change.
        let [irr_h, pre_h, lut_h] = self.initial_ibl;
        let view = |handle, label: &str| -> Result<wgpu::TextureView> {
            renderer
                .get_texture_view(handle)
                .ok_or_else(|| format!("PassPBROpaque: initial {label} texture missing").into())
        };
        let irradiance_view = view(irr_h, "ibl_diffuse")?;
        let prefilter_view = view(pre_h, "ibl_specular")?;
        let brdf_lut_view = view(lut_h, "ibl_brdf_lut")?;

        let prefilter_sampler = {
            let device = renderer.get_device();
            device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                lod_min_clamp: 0.0,
                lod_max_clamp: 16.0,
                ..Default::default()
            })
        };

        // The LUT's U axis is NdotV, not an angle: clamp instead of wrapping
        let brdf_lut_sampler = renderer
            .get_device()
            .create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

        let globals_bind_group = build_globals_bind_group(
            renderer.get_device(),
            &pipeline.bind_group_layouts[MATERIAL_BIND_GROUP_GLOBALS],
            &camera_buffer,
            &scene_buffer,
            &irradiance_view,
            &prefilter_view,
            &brdf_lut_view,
            &ibl_sampler,
            &prefilter_sampler,
            &brdf_lut_sampler,
        );
        let cached_ibl = [irr_h, pre_h, lut_h];

        // Background sub-draw; draw() rebinds its equirect when RenderStateComponent.background
        // changes, starting from a 1px fallback.
        let bg = {
            let bg_bind_groups = vec![vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ]];
            // Same attachments as the scene draw; the sky writes color and velocity, ambient stays 0
            let mut bg_targets = color_targets.clone();
            if let Some(ambient) = &mut bg_targets[2] {
                ambient.write_mask = wgpu::ColorWrites::empty();
            }
            let bg_pipeline = renderer.create_pipeline(PipelineDesc {
                label: Some("pass_pbr_opaque_bg_pipeline"),
                vs: ShaderDesc {
                    source: BACKGROUND_VERTEX_SHADER,
                    entry_func: "vs_main",
                },
                ps: ShaderDesc {
                    source: BACKGROUND_FRAGMENT_SHADER,
                    entry_func: "fs_main",
                },
                vertex_buffers: &[],
                bind_groups: bg_bind_groups,
                targets: &bg_targets,
                // LessEqual + write disabled: fills only pixels where depth == 1.0 (far plane).
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_TARGET_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                    unclipped_depth: false,
                },
            })?;
            let equirect_fallback_h = renderer.create_texture_from_pixels(
                "bg_equirect_fallback",
                TextureData {
                    mips: &[DEFAULT_EQUIRECT_FALLBACK_PIXEL],
                    width: 1,
                    height: 1,
                    format: wgpu::TextureFormat::Rgba32Float,
                },
            )?;
            let equirect_view = renderer
                .get_texture_view(equirect_fallback_h)
                .ok_or("PassPBROpaque: background fallback texture missing")?;
            let bg_sampler = renderer
                .get_device()
                .create_sampler(&wgpu::SamplerDescriptor {
                    address_mode_u: wgpu::AddressMode::Repeat, // equirect wraps in U
                    address_mode_v: wgpu::AddressMode::ClampToEdge, // poles clamp in V
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                });
            let bg_camera_buf = renderer
                .get_device()
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pass_pbr_opaque_bg_camera"),
                    size: std::mem::size_of::<BgCameraUbo>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            let bg_bind_group = build_bg_bind_group(
                renderer.get_device(),
                &bg_pipeline.bind_group_layouts[0],
                &bg_camera_buf,
                &equirect_view,
                &bg_sampler,
            );
            BgSubState {
                pipeline: bg_pipeline,
                camera_buf: bg_camera_buf,
                bind_group: bg_bind_group,
                sampler: bg_sampler,
                cached_bg: equirect_fallback_h,
                fallback_bg: equirect_fallback_h,
            }
        };

        self.state = Some(PassPBROpaqueState {
            camera_uniform: CameraUniform::new(),
            camera_buffer,
            scene_buffer,
            globals_bind_group,
            pipeline,
            ibl_sampler,
            prefilter_sampler,
            brdf_lut_sampler,
            per_draw_buffer,
            per_draw_bind_group,
            bg,
            cached_ibl,
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
        // Query per-entity storages + the global render state (bg/IBL/fog) this pass needs.
        let transform_components = world.query::<TransformComponent>()?;
        let mesh_rendering_components = world.query::<MeshRenderingComponent>()?;
        let render_state = world.get_global::<RenderStateComponent>()?;

        // Rebuild the IBL globals bind group only when the component's handles change (steady state: 0).
        {
            let ibl = [
                render_state.ibl_diffuse,
                render_state.ibl_specular,
                render_state.ibl_brdf_lut,
            ];
            if get_state(self).cached_ibl != ibl {
                let fallback = self.initial_ibl;
                let irradiance_view =
                    view_or_fallback(renderer, ibl[0], fallback[0], "ibl_diffuse")?;
                let prefilter_view =
                    view_or_fallback(renderer, ibl[1], fallback[1], "ibl_specular")?;
                let brdf_lut_view =
                    view_or_fallback(renderer, ibl[2], fallback[2], "ibl_brdf_lut")?;
                let device = renderer.get_device();
                let state = get_state(self);
                state.globals_bind_group = build_globals_bind_group(
                    device,
                    &state.pipeline.bind_group_layouts[MATERIAL_BIND_GROUP_GLOBALS],
                    &state.camera_buffer,
                    &state.scene_buffer,
                    &irradiance_view,
                    &prefilter_view,
                    &brdf_lut_view,
                    &state.ibl_sampler,
                    &state.prefilter_sampler,
                    &state.brdf_lut_sampler,
                );
                state.cached_ibl = ibl;
            }
        }

        // Rebuild the background bind group only when the component's equirect handle changes.
        {
            let bg_h = render_state
                .background
                .unwrap_or(get_state(self).bg.fallback_bg);
            if get_state(self).bg.cached_bg != bg_h {
                let fallback = get_state(self).bg.fallback_bg;
                let equirect_view = view_or_fallback(renderer, bg_h, fallback, "background")?;
                let device = renderer.get_device();
                let state = get_state(self);
                state.bg.bind_group = build_bg_bind_group(
                    device,
                    &state.bg.pipeline.bind_group_layouts[0],
                    &state.bg.camera_buf,
                    &equirect_view,
                    &state.bg.sampler,
                );
                state.bg.cached_bg = bg_h;
            }
        }

        let (active_camera_component, active_camera_transform) = world.active_camera()?;

        // Motion history: a reset (camera cut) makes this frame's previous transforms = current
        self.frame += 1;
        if self.motion_history_epoch != render_state.motion_history_epoch {
            self.motion_history_epoch = render_state.motion_history_epoch;
            self.previous_transforms.clear();
            self.previous_view_projection = None;
        }
        let view_projection =
            camera_view_projection(active_camera_component, active_camera_transform);
        let previous_view_projection = self
            .previous_view_projection
            .replace(view_projection)
            .unwrap_or(view_projection);

        // Update camera and scene uniforms. MVP = viewProjection·model is computed in the vertex
        // shader, so the CPU never multiplies per-entity matrices.
        {
            let state = get_state(self);
            state
                .camera_uniform
                .update_data(active_camera_component, active_camera_transform);
            state.camera_uniform.previous_view_projection_matrix =
                previous_view_projection.to_cols_array_2d();
            renderer.get_queue().write_buffer(
                &state.camera_buffer,
                0,
                bytemuck::bytes_of(&state.camera_uniform),
            );
            renderer.get_queue().write_buffer(
                &state.scene_buffer,
                0,
                bytemuck::bytes_of(&SceneUniform::from_render_state(render_state)),
            );
        }

        // Collect (render key, per-draw) pairs of this pass's materials. Opaque: sorting by key groups
        // them by material, then mesh (depth-tested, so order doesn't matter). Translucent: back to
        // front; equal neighbouring keys still share an instanced draw, which keeps that order.
        self.draws.clear();
        let transforms = &transform_components.data;
        let mesh_renderings = &mesh_rendering_components.data;
        if self.previous_transforms.len() < transforms.len() {
            self.previous_transforms.resize(transforms.len(), None);
        }
        for (slot, (slot_transform, slot_mesh_rendering)) in
            transforms.iter().zip(mesh_renderings.iter()).enumerate()
        {
            let Some(transform) = slot_transform else {
                continue;
            };
            let Some(mesh_rendering) = slot_mesh_rendering else {
                continue;
            };
            let Some(key) = mesh_rendering.render_queue_key else {
                continue;
            };
            if (self.translucent || self.exclude_translucent)
                && is_translucent(world, key) != self.translucent
            {
                continue;
            }
            if self.draws.len() == self.max_instances {
                if !self.capacity_warned {
                    log::error!(
                        "PassPBROpaque: per-draw capacity {} reached; extra meshes are not drawn",
                        self.max_instances
                    );
                    self.capacity_warned = true;
                }
                break;
            }
            let current = DrawTransform::from(transform);
            // An entity not drawn last frame (new, re-shown or a reused slot) has no motion yet
            let previous = match self.previous_transforms[slot] {
                Some(previous) if previous.frame + 1 == self.frame => previous.transform,
                _ => current,
            };
            self.previous_transforms[slot] = Some(PreviousTransform {
                transform: current,
                frame: self.frame,
            });
            let distance_squared =
                (transform.position - active_camera_transform.position).length_squared();
            self.draws
                .push((key, distance_squared, PerDrawStd140 { current, previous }));
        }
        if self.translucent {
            self.draws.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        } else {
            self.draws.sort_unstable_by_key(|(key, _, _)| *key);
        }

        // One batch per run of equal keys; the key is decoded once per batch
        self.batches.clear();
        self.staging_buffer.clear();
        for (index, (key, _, per_draw)) in self.draws.iter().enumerate() {
            self.staging_buffer
                .extend_from_slice(bytemuck::bytes_of(per_draw));
            if index > 0 && self.draws[index - 1].0 == *key {
                if let Some(batch) = self.batches.last_mut() {
                    batch.count += 1;
                }
                continue;
            }
            let key_fields = decompose_render_queue_key(*key);
            self.batches.push(DrawBatch {
                material_handle: RendererMaterialHandle::new(
                    key_fields.material_index.into(),
                    NonZeroU32::new(key_fields.material_version.into()).unwrap(),
                ),
                mesh_handle: RendererMeshHandle::new(
                    key_fields.mesh_index.into(),
                    NonZeroU32::new(key_fields.mesh_version.into()).unwrap(),
                ),
                first: index as u32,
                count: 1,
            });
        }
        {
            let state_ref = self
                .state
                .as_ref()
                .expect("PassPBROpaque: state not initialized — call init() before draw()");
            renderer
                .get_queue()
                .write_buffer(&state_ref.per_draw_buffer, 0, &self.staging_buffer);
        }

        let depth_view = renderer
            .get_render_target_view(self.depth_target)
            .expect("PassPBROpaque: depth target missing");
        // The translucent pass draws over the opaque pass's color and depth
        let load = if self.translucent {
            wgpu::LoadOp::Load
        } else {
            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
        };
        let cleared = |handle: PassTextureHandle| {
            let view = renderer
                .get_render_target_view(handle)
                .expect("PassPBROpaque: color target missing");
            wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            }
        };
        let color_attachments = [
            Some(cleared(self.color_target)),
            self.velocity_target.map(cleared),
            self.ambient_target.map(cleared),
        ];

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pass_pbr_opaque_render_pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: if self.translucent {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(1.0)
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        let state_ref = self
            .state
            .as_ref()
            .expect("PassPBROpaque: state not initialized — call init() before draw()");
        render_pass.set_pipeline(&state_ref.pipeline.pipeline);
        // Per-draw array bound once; each batch's instance range selects its entries.
        render_pass.set_bind_group(
            MATERIAL_BIND_GROUP_PERDRAW as u32,
            &state_ref.per_draw_bind_group,
            &[],
        );
        render_pass.set_bind_group(
            MATERIAL_BIND_GROUP_GLOBALS as u32,
            &state_ref.globals_bind_group,
            &[],
        );
        // Batches are sorted by material, so each material is bound once
        let mut bound_material: Option<RendererMaterialHandle> = None;
        for batch in &self.batches {
            if bound_material != Some(batch.material_handle) {
                // A material destroyed this frame can still be referenced until its components'
                // deferred updates run; skip it instead of failing the frame
                let Ok(mat) = world
                    .resources
                    .get_resource::<RendererMaterial>(&batch.material_handle)
                else {
                    continue;
                };
                render_pass.set_bind_group(
                    MATERIAL_BIND_GROUP_TEXTURES as u32,
                    &mat.textures_bind_group,
                    &[],
                );
                render_pass.set_bind_group(
                    MATERIAL_BIND_GROUP_PARAMS as u32,
                    &mat.parameters_bind_group,
                    &[],
                );
                bound_material = Some(batch.material_handle);
            }

            let Ok(mesh) = world
                .resources
                .get_resource::<RendererMesh>(&batch.mesh_handle)
            else {
                continue;
            };
            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            // One instanced draw per batch: instance_index in [first, first+count) selects per-draw data.
            render_pass.draw_indexed(
                0..mesh.index_count,
                0,
                batch.first..(batch.first + batch.count),
            );
        }

        // Background sub-draw: within the same render pass, after opaque geometry.
        // LessEqual depth test + write_enabled=false: fills only pixels where depth == 1.0 (far plane = no geometry).
        if !self.translucent {
            let bg = &state_ref.bg;
            let fwd = camera_forward(active_camera_transform);
            let right = fwd.cross(Vector3f::Y).normalize();
            let up = right.cross(fwd);
            let ubo = BgCameraUbo {
                right: right.to_array(),
                tan_half_fov: (active_camera_component.fov.to_radians() / 2.0).tan(),
                up: up.to_array(),
                aspect: active_camera_component.aspect.get_value(),
                fwd: fwd.to_array(),
                _pad: 0.0,
                // Environment images are drawn as-is; the white fallback is tinted to the solid color
                bg_color: match render_state.background {
                    Some(_) => [1.0, 1.0, 1.0],
                    None => render_state.background_color,
                },
                _pad2: 0.0,
                previous_view_projection: previous_view_projection.to_cols_array_2d(),
            };
            renderer
                .get_queue()
                .write_buffer(&bg.camera_buf, 0, bytemuck::bytes_of(&ubo));
            render_pass.set_pipeline(&bg.pipeline.pipeline);
            render_pass.set_bind_group(0, &bg.bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}
