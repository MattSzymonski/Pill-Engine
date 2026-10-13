use crate::config::{
    FULLSCREEN_VERTEX_SHADER, MIP_DOWNSAMPLE_FRAGMENT_SHADER, MIP_DOWNSAMPLE_NORMAL_FRAGMENT_SHADER,
};
use crate::graphics::RendererTextureHandle;
use crate::resources::{Resource, ResourceStorage, TextureType};

use pill_core::{PillTypeMapKey, Result};

use std::cell::RefCell;
use std::collections::HashMap;

pub struct RendererTexture {
    pub name: String,
    pub texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl RendererTexture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        name: &str,
        rgba: &[u8],
        width: u32,
        height: u32,
        texture_type: TextureType,
    ) -> Result<Self> {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let format = match texture_type {
            TextureType::Color => wgpu::TextureFormat::Rgba8UnormSrgb,
            TextureType::Normal => wgpu::TextureFormat::Rgba8Unorm,
            TextureType::MetallicRoughness => wgpu::TextureFormat::Rgba8Unorm,
            TextureType::Emissive => wgpu::TextureFormat::Rgba8UnormSrgb,
        };

        let mip_count = mip_level_count(width, height);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(name),
            size,
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        // Write mip 0; the caller fills the rest with PillRenderer::generate_mipmaps
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Trilinear
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: (mip_count - 1) as f32,
            ..Default::default()
        });

        Ok(Self {
            name: name.to_string(),
            texture,
            texture_view,
            sampler,
        })
    }

    pub fn new_depth_texture(
        device: &wgpu::Device,
        surface_configuration: &wgpu::SurfaceConfiguration,
        label: &str,
    ) -> Result<Self> {
        let size = wgpu::Extent3d {
            width: surface_configuration.width,
            height: surface_configuration.height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Ok(Self {
            name: label.to_string(),
            texture,
            texture_view,
            sampler,
        })
    }

    /// Uploads a 2D texture; `mip_pixels[i]` holds mip `i`, sized
    /// `(base_width >> i) * (base_height >> i) * bytes_per_texel(format)`.
    pub fn new_from_pixels(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        name: &str,
        mip_pixels: &[&[u8]],
        base_width: u32,
        base_height: u32,
        format: wgpu::TextureFormat,
    ) -> Result<Self> {
        let mip_count = mip_pixels.len() as u32;
        let bytes_per_texel = format
            .block_copy_size(None)
            .expect("new_from_pixels: format must be a color format");

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(name),
            size: wgpu::Extent3d {
                width: base_width,
                height: base_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for (mip, &mip_data) in mip_pixels.iter().enumerate() {
            let mip = mip as u32;
            let mip_w = (base_width >> mip).max(1);
            let mip_h = (base_height >> mip).max(1);
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                mip_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(mip_w * bytes_per_texel),
                    rows_per_image: Some(mip_h),
                },
                wgpu::Extent3d {
                    width: mip_w,
                    height: mip_h,
                    depth_or_array_layers: 1,
                },
            );
        }

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 16.0,
            ..Default::default()
        });

        Ok(Self {
            name: name.to_string(),
            texture,
            texture_view,
            sampler,
        })
    }

    pub fn new_render_target(
        device: &wgpu::Device,
        label: &str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Result<Self> {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self {
            name: label.to_string(),
            texture,
            texture_view,
            sampler,
        })
    }
}

/// Full chain down to 1×1.
fn mip_level_count(width: u32, height: u32) -> u32 {
    u32::BITS - width.max(height).max(1).leading_zeros()
}

/// Fills mips 1.. of a texture from mip 0: each level is a full-screen draw sampling the previous
/// one bilinearly (wgpu has no glGenerateMipmap). Normal maps are renormalized per level.
/// Owned by the renderer, so shaders and layouts are built once and a pipeline once per
/// (format, renormalize).
pub(crate) struct MipGenerator {
    vertex_module: wgpu::ShaderModule,
    fragment_module: wgpu::ShaderModule,
    normal_fragment_module: wgpu::ShaderModule,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
    pipelines: RefCell<HashMap<(wgpu::TextureFormat, bool), wgpu::RenderPipeline>>,
}

impl MipGenerator {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let shader = |label: &str, source: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mip_downsample_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mip_downsample_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            vertex_module: shader("mip_downsample_vertex", FULLSCREEN_VERTEX_SHADER),
            fragment_module: shader("mip_downsample_fragment", MIP_DOWNSAMPLE_FRAGMENT_SHADER),
            normal_fragment_module: shader(
                "mip_downsample_normal_fragment",
                MIP_DOWNSAMPLE_NORMAL_FRAGMENT_SHADER,
            ),
            bind_group_layout,
            pipeline_layout,
            sampler,
            pipelines: RefCell::new(HashMap::new()),
        }
    }

    fn pipeline(
        &self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        renormalize: bool,
    ) -> wgpu::RenderPipeline {
        self.pipelines
            .borrow_mut()
            .entry((format, renormalize))
            .or_insert_with(|| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("mip_downsample_pipeline"),
                    layout: Some(&self.pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &self.vertex_module,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: if renormalize {
                            &self.normal_fragment_module
                        } else {
                            &self.fragment_module
                        },
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                })
            })
            .clone()
    }

    /// Fills every mip below 0 of `texture` (created with RENDER_ATTACHMENT usage).
    pub(crate) fn generate(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        renormalize: bool,
    ) {
        let mip_count = texture.mip_level_count();
        if mip_count <= 1 {
            return;
        }
        let pipeline = self.pipeline(device, texture.format(), renormalize);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mip_downsample_encoder"),
        });
        for mip in 1..mip_count {
            let source_view = texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: mip - 1,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let destination_view = texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mip_downsample_bind_group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mip_downsample_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &destination_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

impl PillTypeMapKey for RendererTexture {
    type Storage = ResourceStorage<RendererTexture>;
}

impl Resource for RendererTexture {
    type Handle = RendererTextureHandle;

    fn get_name(&self) -> String {
        self.name.clone()
    }
}
