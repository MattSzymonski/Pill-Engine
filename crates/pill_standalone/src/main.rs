#![cfg_attr(debug_assertions, allow(dead_code, unused_imports))]

use pill_engine::internal::*;
use pill_renderer;

use std::io::Write;
use log::{LevelFilter, debug, info};

use winit::{ // Import dependencies
    event::*, // Bring all public items into scope
    event_loop::{ControlFlow, EventLoop},
    window::{WindowBuilder},
};


fn main() {
    
    // Configure logging
    #[cfg(debug_assertions)]
    env_logger::Builder::new()
        .format(|buf, record| {
            writeln!(buf, "[{}] {} {}:{}: {}",
                record.level(),
                chrono::Local::now().format("%Y-%m-%dT%H:%M:%S"),
                record.file().unwrap_or("unknown"),
                record.line().unwrap_or(0),
                record.args()
            )
        })
        .filter_module("pill_standalone", LevelFilter::Debug)
        .filter_module("pill_engine", LevelFilter::Debug)
        .filter_module("pill_renderer", LevelFilter::Debug)
        .init();

    info!("Pill Standalone initializing");

    // Init window
    let event_loop = EventLoop::new();
    let title = env!("CARGO_PKG_NAME");
    let window = WindowBuilder::new().with_title(title).build(&event_loop).unwrap();
    let mut last_render_time = std::time::Instant::now();

    // Init engine
    let game: Box<dyn PillGame> = Box::new(pill_game::Game {});
    let renderer: Box<dyn PillRenderer> = Box::new(<pill_renderer::Renderer as PillRenderer>::new(&window));
    let mut engine = Engine::new(game, renderer);
    engine.initialize();

    // Run loop
    event_loop.run(move |event, _, control_flow|  { // Run function takes closure
        *control_flow = ControlFlow::Poll; 
        match event {


            Event::MainEventsCleared => {
                window.request_redraw();
            },

            // Raw events not associated with any specific window
            // Event::DeviceEvent {
            //     ref event,
            //     .. // We're not using device_id currently
            // } => {
            //     state.input(event);
            // }

            // Handle window events
            Event::WindowEvent {
                ref event,
                window_id,
            } 
            if window_id == window.id() => {
                match event {        

                    // Pass keyboard input to engine
                    WindowEvent::KeyboardInput { 
                        input,
                        .. // Skip other
                    } => { 
                        engine.pass_keyboard_key_input(&input);
                    },

                    // Pass mouse key input to engine
                    WindowEvent::MouseInput { 
                        button,
                        state,
                        .. // Skip other
                    } => { 
                        engine.pass_mouse_key_input( &button, &state);
                    },

                    // Pass mouse scroll input to engine
                    WindowEvent::MouseWheel { 
                        delta,
                        .. // Skip other
                    } => { 
                        engine.pass_mouse_wheel_input(&delta);
                    },

                    // Pass mouse motion input to engine
                    WindowEvent::CursorMoved {
                        position,
                        .. // Skip other
                    }=> { 
                        engine.pass_mouse_motion_input(&position);
                    },

                    // Close window
                    WindowEvent::CloseRequested => {
                        engine.shutdown();
                        *control_flow = ControlFlow::Exit
                    },

                    // Resize window
                    WindowEvent::Resized(physical_size) => {
                        engine.resize(*physical_size);
                    },



                    // Change window scale factor
                    // WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    //     state.resize(**new_inner_size);
                    // },


                    _ => {}
                }
            }

            // Handle redraw requests
            Event::RedrawRequested(_) => {
                let now = std::time::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                engine.update(dt);
            }
            _ => {}
        }
    });


}
