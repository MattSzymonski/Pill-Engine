use std::{collections::{vec_deque, VecDeque}, borrow::{Borrow, BorrowMut}};

use crate::game::Engine;
use anyhow::{Result, Context, Error};
use lazy_static::__Deref;
use winit::event::ElementState;

use super::{InputComponent, InputEvent, input_component::GlobalComponent};

pub fn input_system(engine: &mut Engine) -> Result<()> {

    while engine.input_queue.is_empty() == false {
        //let mut component = engine.take_global_component::<InputComponent>();
        
        let front_event = engine.input_queue.pop_front().unwrap();
        match front_event {
            InputEvent::KeyboardKey { key, state } => {
                //if component.is_some() {}
                //component.overwrite_prev_keys();
                match state {
                    ElementState::Pressed => { 
                        //component.press_key(key as usize); 
                    }
                    ElementState::Released => { 
                        //component.release_key(key as usize) 
                    }
                }
            }
            _ => ()
        }
        // let mut new_component = GlobalComponent::<InputComponent>::new();
        // new_component.set_component(component)?;
        // engine.insert_global_component::<InputComponent>(new_component)?;
    }
    
    Ok(())
}