use crate::params::Params;

use std::slice::{Iter, IterMut};
use std::slice::{Chunks, ChunksMut};

pub struct ParamsHeap {
    topology: Vec<usize>,
    buffer: Vec<f32>,
    bias_count: usize,
    weight_count: usize,
}

impl ParamsHeap {
    pub fn new(topology: Vec<usize>) -> Self {
        assert!(topology.len() > 1, "Wrong topology for FFNN was used, number of layers is less or equal to 1");
        let neuron_count: usize = topology.iter().sum();
        let bias_count = neuron_count - topology[0];

        let mut weight_count: usize = 0;

        let mut top_iter = topology.iter();
        let mut prev_layer = *top_iter.next().unwrap();
        for layer in top_iter {
            let layer = *layer;
            weight_count += layer * prev_layer;
            prev_layer = layer;
        }
        
        let parameter_count = weight_count + bias_count;
        let buffer: Vec<f32> = vec![0.0_f32; parameter_count];
        Self {
            topology,
            buffer,
            bias_count,
            weight_count,
        }
    }
    fn biases_buff(&self) -> &[f32] {
        &self.buffer[0..self.bias_count]
    }
    fn weights_buff(&self) -> &[f32] {
        &self.buffer[self.bias_count..self.weight_count + self.bias_count]
    }
    fn biases_buff_mut(&mut self) -> &mut [f32] {
        &mut self.buffer[0..self.bias_count]
    }
    fn weights_buff_mut(&mut self) -> &mut [f32] {
        &mut self.buffer[self.bias_count..self.weight_count + self.bias_count]
    }
}

impl Params for ParamsHeap {
    type BiasesIter<'a> = BiasIter<'a, 'a>;
    type BiasesIterMut<'a> = BiasIterMut<'a, 'a>;

    type WeightsRowIter<'a> = Chunks<'a, f32>;
    type WeightsIter<'a> = WeightsIter<'a, 'a>;

    type WeightsRowIterMut<'a> = ChunksMut<'a, f32>;
    type WeightsIterMut<'a> = WeightsIterMut<'a, 'a>;

    type BiasesBuff<'a> = Iter<'a, f32>;
    type BiasesBuffMut<'a> = IterMut<'a, f32>;

    type WeightsBuff<'a> = Iter<'a, f32>;
    type WeightsBuffMut<'a> = IterMut<'a, f32>;

    type RawParamIter<'a> = Iter<'a, f32>;
    type RawParamIterMut<'a> = IterMut<'a, f32>;

    fn biases_iter(&self) -> Self::BiasesIter<'_> {
        BiasIter::new(self.biases_buff(), &self.topology)
    }
    fn biases_iter_mut(&mut self) -> Self::BiasesIterMut<'_> {
        let bias_count = self.bias_count;
        let bias_buffer = &mut self.buffer[0..bias_count];
        BiasIterMut::new(bias_buffer, &self.topology)
    }
    fn weights_iter(&self) -> Self::WeightsIter<'_> {
        WeightsIter::new(self.weights_buff(), &self.topology)
    }

    fn weights_iter_mut(&mut self) -> Self::WeightsIterMut<'_> {
        let weights_buff = &mut self.buffer[self.bias_count..self.weight_count + self.bias_count];
        WeightsIterMut::new(weights_buff, &self.topology)
    }

    fn biases_buff(&self) -> Self::BiasesBuff<'_> {
        self.biases_buff().iter()
    }
    fn biases_buff_mut(&mut self) -> Self::BiasesBuffMut<'_> {
        self.biases_buff_mut().iter_mut()
    }

    fn weights_buff(&self) -> Self::WeightsBuff<'_> {
        self.weights_buff().iter()
    }
    fn weights_buff_mut(&mut self) -> Self::WeightsBuffMut<'_> {
        self.weights_buff_mut().iter_mut()
    }

    fn iter(&self) -> Self::RawParamIter<'_> {
        self.buffer.iter()
    }
    fn iter_mut(&mut self) -> Self::RawParamIterMut<'_> {
        self.buffer.iter_mut()
    }
}

pub struct BiasIter<'buf, 'top> {
    buffer: &'buf [f32],
    topology: &'top [usize],
}

impl<'buf, 'top> BiasIter<'buf, 'top> {
    fn new(bias_buffer: &'buf [f32], topology: &'top [usize]) -> Self {
        let (_, rest) = topology.split_first().expect("Wrong topology was provided");
        Self {
            buffer: bias_buffer,
            topology: rest
        }
    }
}

impl<'buf, 'top> Iterator for BiasIter<'buf, 'top> {
    type Item = &'buf [f32];

    fn next(&mut self) -> Option<Self::Item> {
        let topology = self.topology;
        let (&neurons_in_layer, rest_topology) = topology.split_first()?;

        debug_assert!(neurons_in_layer <= self.buffer.len(), "Buffer size is incompatible with given topology");

        let buffer = self.buffer;
        let (curr_data, rest_buffer) = buffer.split_at(neurons_in_layer);

        self.buffer = rest_buffer;
        self.topology = rest_topology;
        
        Some(curr_data)
    }
}


pub struct BiasIterMut<'buf, 'top> {
    buffer: &'buf mut [f32],
    topology: &'top [usize],
}

impl<'buf, 'top> BiasIterMut<'buf, 'top> {
    fn new(bias_buffer: &'buf mut [f32], topology: &'top [usize]) -> Self {
        let (_, rest) = topology.split_first().expect("Wrong topology was provided");
        Self {
            buffer: bias_buffer,
            topology: rest
        }
    }
}

impl<'buf, 'top> Iterator for BiasIterMut<'buf, 'top> {
    type Item = &'buf mut [f32];

    fn next(&mut self) -> Option<Self::Item> {
        let topology = self.topology;
        let (&neurons_in_layer, rest_topology) = topology.split_first()?;

        debug_assert!(neurons_in_layer <= self.buffer.len(), "Buffer size is incompatible with given topology");

        let buffer = std::mem::take(&mut self.buffer);
        let (curr_data, rest_buffer) = buffer.split_at_mut(neurons_in_layer);

        self.buffer = rest_buffer;
        self.topology = rest_topology;
        
        Some(curr_data)
    }
}

pub struct WeightsIter<'buf, 'top> {
    buffer: &'buf [f32],
    topology: &'top [usize],
    prev_layer_len: usize
}

impl<'buf, 'top> WeightsIter<'buf, 'top> { 
    fn new(weights_buffer: &'buf [f32], topology: &'top [usize]) -> Self {
        let (input_layer, rest) = topology.split_first().expect("Wrong topology was provided");
        Self {
            buffer: weights_buffer,
            topology: rest,
            prev_layer_len: *input_layer
        }
    }
}

impl<'buf, 'top> Iterator for WeightsIter<'buf, 'top> { 
    type Item = std::slice::Chunks<'buf, f32>;

    fn next(&mut self) -> Option<Self::Item> {
        let topology = self.topology;
        let (&neurons_in_layer, rest_topology) = topology.split_first()?;

        let weights_in_layer = neurons_in_layer * self.prev_layer_len;
        
        debug_assert!(weights_in_layer <= self.buffer.len(), "Buffer size is incompatible with given topology");
        
        let buffer = self.buffer;
        let (curr_data, rest_buffer) = buffer.split_at(weights_in_layer);
        
        self.buffer = rest_buffer;
        self.topology = rest_topology;
        
        let prev_layer_len = self.prev_layer_len;
        self.prev_layer_len = neurons_in_layer;

        debug_assert!(curr_data.len() % prev_layer_len == 0, "Unable to split buffer at given chunks");
        Some(curr_data.chunks(prev_layer_len))
    }
}


pub struct WeightsIterMut<'buf, 'top> {
    buffer: &'buf mut [f32],
    topology: &'top [usize],
    prev_layer_len: usize
}

impl<'buf, 'top> WeightsIterMut<'buf, 'top> { 
    fn new(weights_buffer: &'buf mut [f32], topology: &'top [usize]) -> Self {
        let (input_layer, rest) = topology.split_first().expect("Wrong topology was provided");
        Self {
            buffer: weights_buffer,
            topology: rest,
            prev_layer_len: *input_layer
        }
    }
}

impl<'buf, 'top> Iterator for WeightsIterMut<'buf, 'top> { 
    type Item = std::slice::ChunksMut<'buf, f32>;

    fn next(&mut self) -> Option<Self::Item> {
        let topology = self.topology;
        let (&neurons_in_layer, rest_topology) = topology.split_first()?;

        let weights_in_layer = neurons_in_layer * self.prev_layer_len;
        
        debug_assert!(weights_in_layer <= self.buffer.len(), "Buffer size is incompatible with given topology");
        
        let buffer = std::mem::take(&mut self.buffer);
        let (curr_data, rest_buffer) = buffer.split_at_mut(weights_in_layer);
        
        self.buffer = rest_buffer;
        self.topology = rest_topology;
        
        let prev_layer_len = self.prev_layer_len;
        self.prev_layer_len = neurons_in_layer;

        debug_assert!(curr_data.len() % prev_layer_len == 0, "Unable to split buffer at given chunks");
        Some(curr_data.chunks_mut(prev_layer_len))
    }
}
