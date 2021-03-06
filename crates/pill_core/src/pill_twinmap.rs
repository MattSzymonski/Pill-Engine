use std::collections::HashMap;
use core::hash::Hash;
use anyhow::{ Result, Context, Error };

// PillTwinMap is container storing bidirectional mapping from one type to another and vice versa
// From outside, in this container, in the same time key is value, and value is key
// This container does not support getting mutable. This is needed to assure that both alphas and betas in both maps are always the same
// Sometimes there is a need to search by value, in case of hashmap it has O(n) time complexity. PillTwinMap turns it to O(1), the drawback is doubled memory
// This container is suitable to store primitive, clonable types only
// E.g. <String, u32> <-> <u32, String> where both strings and u32 have the same values 

pub struct PillTwinMap<K: Eq + Hash + Clone, V: Eq + Hash + Clone> {
    key_value_map: HashMap<K, V>,
    value_key_map: HashMap<V, K>,
}

impl<K: Eq + Hash + Clone, V: Eq + Hash + Clone> PillTwinMap<K, V> {
    pub fn new() -> Self {
        Self {
            key_value_map: HashMap::<K, V>::new(),
            value_key_map: HashMap::<V, K>::new(),
        }
    }
    
    pub fn get_value(&self, key: &K) -> Option<&V> { 
        self.key_value_map.get(key)
    }

    pub fn get_key(&self, value: &V) -> Option<&K> { 
        self.value_key_map.get(value)
    }

    pub fn insert(&mut self, key: &K, value: &V) {
        self.key_value_map.insert( key.clone(), value.clone());
        self.value_key_map.insert(value.clone(), key.clone());
    }

    pub fn remove_by_key(&mut self, key: &K) {
        match self.key_value_map.remove_entry(key) {
            Some(v) => { self.value_key_map.remove_entry(&v.1); },
            None => {},
        };
    }

    pub fn remove_by_value(&mut self, value: &V) {
        match self.value_key_map.remove_entry(value) {
            Some(v) => { self.key_value_map.remove_entry(&v.1); },
            None => {},
        };
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.key_value_map.contains_key(key)
    }

    pub fn contains_value(&self, value: &V) -> bool {
        self.value_key_map.contains_key(value)
    }
}