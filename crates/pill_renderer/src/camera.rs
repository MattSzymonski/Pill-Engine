use cgmath::*;
use pill_engine::{game::TransformComponent, internal::CameraComponent};
use wgpu::util::DeviceExt;
use std::f32::consts::FRAC_PI_2;
use std::time::Duration;
use winit::dpi::PhysicalPosition;
use winit::event::*;
use anyhow::{Result, Context, Error};
use pill_engine::internal::RendererCameraHandle;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;


// --- Camera Uniform ---

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CameraUniform {
    pub(crate) position_matrix: [[f32; 4]; 4], // Camera translation
    pub(crate) view_projection_matrix: [[f32; 4]; 4], // Perspective manipulation
}

impl CameraUniform {
    pub fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            position_matrix: cgmath::Matrix4::identity().into(),
            view_projection_matrix: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_projection_matrix(&mut self, camera_component: &CameraComponent, transform_component: &TransformComponent) {
        self.view_projection_matrix = (CameraUniform::calculate_projection_matrix(camera_component) * CameraUniform::calculate_view_matrix(transform_component)).into();
    }

    fn calculate_view_matrix(transform_component: &TransformComponent) -> Matrix4::<f32> {
        let (sin_pitch, cos_pitch) = Deg(transform_component.rotation.x).sin_cos();
        let (sin_yaw, cos_yaw) = Deg(transform_component.rotation.y).sin_cos();
    
        let direction = Vector3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize();
        let position = cgmath::Point3::from_vec(transform_component.position);
    
        Matrix4::look_to_rh(
            position,
            direction,
            cgmath::Vector3::unit_y()
        )
    }
    
    fn calculate_projection_matrix(camera_component: &CameraComponent) -> Matrix4::<f32> {
        OPENGL_TO_WGPU_MATRIX * cgmath::perspective(
            Deg(camera_component.fov), 
            camera_component.aspect, 
            0.1,
            100.0
        )
    }
}



// --- Camera ---

#[derive(Debug)]
pub struct RendererCamera {
    pub(crate) uniform: CameraUniform,
    buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl RendererCamera {
    pub fn new(device: &wgpu::Device, camera_bind_group_layout: &wgpu::BindGroupLayout) -> Result<Self> {

        let uniform = CameraUniform::new();

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let camera = Self {
            uniform,
            buffer,
            bind_group,
        };

        Ok(camera)
    }

    pub fn update(&mut self, queue: &wgpu::Queue, camera_component: &CameraComponent, transform_component: &TransformComponent) {
        self.uniform.update_view_projection_matrix(camera_component, transform_component);
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[self.uniform]));
    }
}