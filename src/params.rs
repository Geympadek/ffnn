pub mod heap_params;
pub mod stack_params;

pub trait Params {
    type BiasesIter<'a>: Iterator<Item=&'a [f32]> where Self: 'a;
    type BiasesIterMut<'a>: Iterator<Item=&'a mut [f32]> where Self: 'a;

    type WeightsRowIter<'a>: Iterator<Item = &'a [f32]> where Self: 'a;
    type WeightsIter<'a>: Iterator<Item = Self::WeightsRowIter<'a>> where Self: 'a;

    type WeightsRowIterMut<'a>: Iterator<Item = &'a mut [f32]> where Self: 'a;
    type WeightsIterMut<'a>: Iterator<Item = Self::WeightsRowIterMut<'a>> where Self: 'a;

    type BiasesBuff<'a>: Iterator<Item=&'a f32> where Self: 'a;
    type BiasesBuffMut<'a>: Iterator<Item=&'a mut f32> where Self: 'a;

    type WeightsBuff<'a>: Iterator<Item=&'a f32> where Self: 'a;
    type WeightsBuffMut<'a>: Iterator<Item=&'a mut f32> where Self: 'a;

    type RawParamIter<'a>: Iterator<Item=&'a f32> where Self: 'a;
    type RawParamIterMut<'a>: Iterator<Item=&'a mut f32> where Self: 'a;
    
    fn biases_iter(&self) -> Self::BiasesIter<'_>;
    fn biases_iter_mut(&mut self) -> Self::BiasesIterMut<'_>;

    fn weights_iter(&self) -> Self::WeightsIter<'_>;
    fn weights_iter_mut(&mut self) -> Self::WeightsIterMut<'_>;

    fn biases_buff(&self) -> Self::BiasesBuff<'_>;
    fn biases_buff_mut(&mut self) -> Self::BiasesBuffMut<'_>;

    fn weights_buff(&self) -> Self::WeightsBuff<'_>;
    fn weights_buff_mut(&mut self) -> Self::WeightsBuffMut<'_>;

    fn iter(&self) -> Self::RawParamIter<'_>;
    fn iter_mut(&mut self) -> Self::RawParamIterMut<'_>;

    ///Constructs a new vector of vectors, copies all biases into it.
    fn construct_biases(&self) -> Vec<Vec<f32>> {
        let mut result = Vec::new();

        for layer in self.biases_iter() {
            let mut bias_layer = Vec::new();
            for bias in layer {
                bias_layer.push(*bias);
            }
            result.push(bias_layer);
        }
        result
    }

    ///Constructs a new 3D vector, copies all weights into it, while preserving the layout.
    fn construct_weights(&self) -> Vec<Vec<Vec<f32>>> {
        let mut result = Vec::new();

        for layer in self.weights_iter() {
            let mut weight_layer = Vec::new();
            for neuron in layer {
                let mut weights_neuron = Vec::new();
                for weight in neuron {
                    weights_neuron.push(*weight);
                }
                weight_layer.push(weights_neuron);
            }
            result.push(weight_layer);
        }
        result
    }

    fn construct_biases_raw(&self) -> Vec<f32> {
        self.biases_buff().copied().collect()
    }

    fn construct_weights_raw(&self) -> Vec<f32> {
        self.weights_buff().copied().collect()
    }

    fn construct_raw(&self) -> Vec<f32> {
        self.iter().copied().collect()
    }
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

        for (idx, val) in params.iter_mut().enumerate() {
            *val = idx as f32;
        }

        let biases = params.construct_biases();
        assert_eq!(biases, vec![vec![0.0_f32, 1.0_f32]]);

        let weights = params.construct_weights();
        assert_eq!(weights, vec![vec![vec![2.0_f32, 3.0_f32], vec![4.0_f32, 5.0_f32]]]);
    }
}