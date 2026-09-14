use crate::params::Params;

use std::slice::{Iter, IterMut, Chunks, ChunksMut};

/// A [`Params`] implementation backed by a single contiguous `Vec<f32>`.
///
/// Memory layout:
///
/// ```text
/// [ biases ......... | weights ................................. ]
/// ```
///
/// Biases are stored layer-by-layer, with one `f32` per bias in each non-input
/// layer. Weights are stored layer-by-layer, and within a layer
/// neuron-by-neuron, with each neuron's incoming weights contiguous. This
/// keeps every parameter in one allocation and makes the flat iterators
/// trivial.
#[derive(Clone)]
pub struct ParamsHeap {
    topology: Vec<usize>,
    buffer: Vec<f32>,
    bias_count: usize,
    weight_count: usize,
}

impl ParamsHeap {
    /// Creates a new `ParamsHeap` for a network with the given layer sizes.
    ///
    /// `topology` lists the number of neurons in each layer, starting with
    /// the input layer, and must contain at least two entries. All
    /// parameters are initialized to `0.0`.
    ///
    /// # Panics
    ///
    /// Panics if `topology.len() <= 1`.
    pub fn new(topology: Vec<usize>) -> Self {
        assert!(
            topology.len() > 1,
            "topology must contain at least two layers, got {}",
            topology.len(),
        );
        
        let bias_count = topology[1..].iter().sum();

        let weight_count: usize = topology
            .windows(2)
            .map(|pair| pair[0] * pair[1])
            .sum();
        
        let buffer = vec![0.0; bias_count + weight_count];
        Self {
            topology,
            buffer,
            bias_count,
            weight_count,
        }
    }

    /// Returns the bias region as a contiguous slice.
    fn bias_slice(&self) -> &[f32] {
        &self.buffer[0..self.bias_count]
    }
    /// Returns the bias region as a mutable contiguous slice.
    fn bias_slice_mut(&mut self) -> &mut [f32] {
        &mut self.buffer[0..self.bias_count]
    }
    /// Returns the weight region as a contiguous slice.
    fn weight_slice(&self) -> &[f32] {
        &self.buffer[self.bias_count..self.weight_count + self.bias_count]
    }
    /// Returns the weight region as a mutable contiguous slice.
    fn weight_slice_mut(&mut self) -> &mut [f32] {
        &mut self.buffer[self.bias_count..self.weight_count + self.bias_count]
    }
}

impl Params for ParamsHeap {
    type BiasLayers<'a> = BiasIter<'a, 'a>;
    type BiasLayersMut<'a> = BiasIterMut<'a, 'a>;

    type WeightNeurons<'a> = Chunks<'a, f32>;
    type WeightLayers<'a> = WeightsIter<'a, 'a>;

    type WeightNeuronsMut<'a> = ChunksMut<'a, f32>;
    type WeightLayersMut<'a> = WeightsIterMut<'a, 'a>;

    type BiasesFlat<'a> = Iter<'a, f32>;
    type BiasesFlatMut<'a> = IterMut<'a, f32>;

    type WeightsFlat<'a> = Iter<'a, f32>;
    type WeightsFlatMut<'a> = IterMut<'a, f32>;

    type ParamsIter<'a> = Iter<'a, f32>;
    type ParamsIterMut<'a> = IterMut<'a, f32>;

    type LayerSizes<'a> = Iter<'a, usize>;

    fn bias_layers(&self) -> Self::BiasLayers<'_> {
        BiasIter::new(self.bias_slice(), &self.topology)
    }

    fn bias_layers_mut(&mut self) -> Self::BiasLayersMut<'_> {
        let bias_count = self.bias_count;
        let biases = &mut self.buffer[..bias_count];
        BiasIterMut::new(biases, &self.topology)
    }

    fn weight_layers(&self) -> Self::WeightLayers<'_> {
        WeightsIter::new(self.weight_slice(), &self.topology)
    }

    fn weight_layers_mut(&mut self) -> Self::WeightLayersMut<'_> {
        let bias_count = self.bias_count;
        let weights = &mut self.buffer[bias_count..];
        WeightsIterMut::new(weights, &self.topology)
    }

    fn biases_flat(&self) -> Self::BiasesFlat<'_> {
        self.bias_slice().iter()
    }
    fn biases_flat_mut(&mut self) -> Self::BiasesFlatMut<'_> {
        self.bias_slice_mut().iter_mut()
    }

    fn weights_flat(&self) -> Self::WeightsFlat<'_> {
        self.weight_slice().iter()
    }
    fn weights_flat_mut(&mut self) -> Self::WeightsFlatMut<'_> {
        self.weight_slice_mut().iter_mut()
    }

    fn params(&self) -> Self::ParamsIter<'_> {
        self.buffer.iter()
    }
    fn params_mut(&mut self) -> Self::ParamsIterMut<'_> {
        self.buffer.iter_mut()
    }

    fn layer_sizes(&self) -> Self::LayerSizes<'_> {
        self.topology.iter()
    }

    fn from_params(src: &impl Params) -> Self {
        let topology: Vec<usize> = src.layer_sizes().copied().collect();
        let mut result = Self::new(topology);
        result.copy_from(src);
        result
    }
}

/// Iterator over bias layers of a [`ParamsHeap`].
///
/// Yields one slice per non-input layer. The input layer is skipped; this is
/// why the constructor takes the full topology and discards its first entry.
#[derive(Clone)]
pub struct BiasIter<'buf, 'top> {
    buffer: &'buf [f32],
    topology: &'top [usize],
}

impl<'buf, 'top> BiasIter<'buf, 'top> {
    /// Creates a new iterator over `bias_buffer`, driven by `topology`.
    ///
    /// The first entry of `topology` (the input layer) is skipped.
    ///
    /// # Panics
    ///
    /// Panics if `topology` is empty.
    fn new(bias_buffer: &'buf [f32], topology: &'top [usize]) -> Self {
        let (_, rest) = topology
            .split_first()
            .expect("topology must contain at least one layer");
        Self {
            buffer: bias_buffer,
            topology: rest
        }
    }
}

impl<'buf, 'top> Iterator for BiasIter<'buf, 'top> {
    type Item = &'buf [f32];

    fn next(&mut self) -> Option<Self::Item> {
        let (&neurons_in_layer, rest_topology) = self.topology.split_first()?;

        let (curr_data, rest_buffer) = self.buffer.split_at(neurons_in_layer);

        self.buffer = rest_buffer;
        self.topology = rest_topology;
        
        Some(curr_data)
    }
}

/// Mutable counterpart of [`BiasIter`].
pub struct BiasIterMut<'buf, 'top> {
    buffer: &'buf mut [f32],
    topology: &'top [usize],
}

impl<'buf, 'top> BiasIterMut<'buf, 'top> {
    /// Creates a new mutable iterator over `bias_buffer`, driven by `topology`.
    ///
    /// The first entry of `topology` (the input layer) is skipped.
    ///
    /// # Panics
    ///
    /// Panics if `topology` is empty.
    fn new(bias_buffer: &'buf mut [f32], topology: &'top [usize]) -> Self {
        let (_, rest) = topology
            .split_first()
            .expect("topology must contain at least one layer");
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
