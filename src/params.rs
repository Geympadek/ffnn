pub mod heap_params;
pub mod stack_params;

/// High-level access to a feed-forward network's weights and biases.
///
/// Implementors decide how parameters are stored. This trait exposes:
/// - structure-preserving views: [`Params::bias_layers`], [`Params::weight_layers`]
/// - flat views: [`Params::biases_flat`], [`Params::weights_flat`], [`Params::params`]
/// - allocation helpers: `collect_*`
///
/// Layout convention:
/// - `bias_layers`: one item per non-input layer; each item is `&[f32]`.
/// - `weight_layers`: one item per non-input layer; each layer item is an
///   iterator over neurons; each neuron item is `&[f32]` of weights from the
///   previous layer.
/// - `layer_sizes`: sizes of all layers. For input=5, hidden=6, output=7,
///   this yields `[5, 6, 7]`.
pub trait Params {
    /// Iterator over bias layers. Each item is one layer's biases.
    type BiasLayers<'a>: Iterator<Item=&'a [f32]> where Self: 'a;
    /// Mutable iterator over bias layers. Each item is one layer's biases.
    type BiasLayersMut<'a>: Iterator<Item=&'a mut [f32]> where Self: 'a;

    /// Iterator over neurons in one weight layer.
    type WeightNeurons<'a>: Iterator<Item = &'a [f32]> where Self: 'a;
    /// Mutable iterator over neurons in one weight layer.
    type WeightNeuronsMut<'a>: Iterator<Item = &'a mut [f32]> where Self: 'a;
    
    /// Iterator over weight layers.
    ///
    /// This is a 3D-style view: layer -> neuron -> weight slice.
    type WeightLayers<'a>: Iterator<Item = Self::WeightNeurons<'a>> where Self: 'a;
    /// Mutable iterator over weight layers.
    ///
    /// This is a 3D-style view: layer -> neuron -> mutable weight slice.
    type WeightLayersMut<'a>: Iterator<Item = Self::WeightNeuronsMut<'a>> where Self: 'a;

    /// Flat iterator over all biases, ignoring layer structure.
    type BiasesFlat<'a>: Iterator<Item=&'a f32> where Self: 'a;
    /// Flat mutable iterator over all biases, ignoring layer structure.
    type BiasesFlatMut<'a>: Iterator<Item=&'a mut f32> where Self: 'a;

    /// Flat iterator over all weights, ignoring layer/neuron structure.
    type WeightsFlat<'a>: Iterator<Item=&'a f32> where Self: 'a;
    /// Flat mutable iterator over all weights, ignoring layer/neuron structure.
    type WeightsFlatMut<'a>: Iterator<Item=&'a mut f32> where Self: 'a;

    /// Flat iterator over every parameter: biases and weights
    type ParamsIter<'a>: Iterator<Item=&'a f32> where Self: 'a;
    /// Flat mutable iterator over every parameter: biases and weights
    type ParamsIterMut<'a>: Iterator<Item=&'a mut f32> where Self: 'a;
    
    /// Iterator over layer sizes (including input layer)
    type LayerSizes<'a>: Iterator<Item=&'a usize> where Self: 'a;
    
    /// Returns an iterator over bias layers.
    ///
    /// Each `next()` yields one non-input layer's biases as a slice.
    fn bias_layers(&self) -> Self::BiasLayers<'_>;

    /// Returns a mutable iterator over bias layers.
    ///
    /// Each `next()` yields one non-input layer's biases as a mutable slice.
    fn bias_layers_mut(&mut self) -> Self::BiasLayersMut<'_>;

    /// Returns an iterator over weight layers.
    ///
    /// This is a 3D-style view: layer -> neuron -> weight slice.
    fn weight_layers(&self) -> Self::WeightLayers<'_>;
    /// Returns a mutable iterator over weight layers.
    ///
    /// This is a 3D-style view: layer -> neuron -> mutable weight slice.
    fn weight_layers_mut(&mut self) -> Self::WeightLayersMut<'_>;

    /// Returns a flat iterator over all biases.
    fn biases_flat(&self) -> Self::BiasesFlat<'_>;
    /// Returns a mutable flat iterator over all biases.
    fn biases_flat_mut(&mut self) -> Self::BiasesFlatMut<'_>;

    /// Returns a flat iterator over all weights.
    fn weights_flat(&self) -> Self::WeightsFlat<'_>;
    /// Returns a flat mutable iterator over all weights.
    fn weights_flat_mut(&mut self) -> Self::WeightsFlatMut<'_>;

    /// Returns a flat iterator over all parameters.
    fn params(&self) -> Self::ParamsIter<'_>;
    /// Returns a flat mutable iterator over all parameters.
    fn params_mut(&mut self) -> Self::ParamsIterMut<'_>;

    /// Returns an iterator over layer sizes (including input layer).
    fn layer_sizes(&self) -> Self::LayerSizes<'_>;

    /// Copies all biases into a nested `Vec`.
    fn collect_biases(&self) -> Vec<Vec<f32>> {
        self.bias_layers()
            .map(|layer| layer.to_vec())
            .collect()
    }

    /// Copies all weights into a nested `Vec`, preserving the layer/neuron layout.
    fn collect_weights(&self) -> Vec<Vec<Vec<f32>>> {
        self.weight_layers()
            .map(|layer| {
                layer
                    .map(|neuron| neuron.to_vec())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Copies all biases into a flat `Vec`.
    fn collect_biases_flat(&self) -> Vec<f32> {
        self.biases_flat().copied().collect()
    }
    /// Copies all weights into a flat `Vec`.
    fn collect_weights_flat(&self) -> Vec<f32> {
        self.weights_flat().copied().collect()
    }

    /// Copies every parameter into a flat `Vec`.
    fn collect_params_flat(&self) -> Vec<f32> {
        self.params().copied().collect()
    }

    /// Copies all parameters from `src` into `self`.
    ///
    /// # Panics
    ///
    /// Panics if the number of parameters differs.
    fn copy_from(&mut self, src: &impl Params) {
        let mut src_iter = src.params();

        for dst in self.params_mut() {
            *dst = *src_iter
                .next()
                .expect("source has fewer parameters than destination");
        }

        assert!(
            src_iter.next().is_none(),
            "source has more parameters than destination"
        );
    }

    /// Creates a new `Self` and copies all parameters from `src`.
    fn from_params(src: &impl Params) -> Self;
}

pub use heap_params::ParamsHeap;

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    use crate::params_stack;

    type StackParams = params_stack!([2, 2]);

    #[test]
    fn check_params_print_output() {
        // let mut params = ParamsHeap::new(vec![2, 2]);
        let mut params = StackParams::new();

        for (idx, val) in params.params_mut().enumerate() {
            *val = idx as f32;
        }

        let biases = params.collect_biases();
        assert_eq!(biases, vec![vec![0.0_f32, 1.0_f32]]);

        let weights = params.collect_weights();
        assert_eq!(weights, vec![vec![vec![2.0_f32, 3.0_f32], vec![4.0_f32, 5.0_f32]]]);

        let topology: Vec<usize> = params.layer_sizes().copied().collect();
        assert_eq!(topology, vec![2usize, 2usize]);
    }
}