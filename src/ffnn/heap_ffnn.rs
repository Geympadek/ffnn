use crate::{ffnn::{FFNN, activation::{self, ActivationVal}}, params::{Params, ParamsHeap}};

/// Heap implementation that stores all the parameters in a contigous data structure
/// 
/// # Examples
/// ```
/// use crate::ffnn::ffnn::{self, activation, ParamsHeap, FFNNHeap, FFNN};
/// 
/// let ffnn = FFNNHeap::new(ParamsHeap::new(vec![2, 2, 1]), activation::ActivationVal::ReLU);
/// let output = ffnn.forward_alloc(&[0_f32, 1_f32]);
/// println!("{:?}", output);
/// ```
#[derive(Clone)]
pub struct FFNNHeap {
    ///Parameters of the neural network stored on the heap
    params: ParamsHeap,
    ///Activation function
    activation: ActivationVal,
}

impl FFNNHeap {
    ///Takes ownership of `params` and sets an activation function
    pub fn new(params: ParamsHeap, activation: ActivationVal) -> Self {
        Self {
            params,
            activation
        }
    }
    ///Feeds forward through neural network without allocating any heap memory.
    /// It uses `buffer` for storing layer outputs temporarely.
    /// * `input` - input slice, expected to be the same size as the first layer of FFNN
    /// * `output` - output slice, expected to be the same size as the last layer of FFNN
    /// * `buffer` - buffer slice used for temporary storage of layer output data
    /// # Panics
    /// Function panics if neither `activation` set to anything.
    /// Or if the lens of slices are smaller than expected.
    pub fn forward_with_buff(&self, input: &[f32], output: &mut [f32], buffer: &mut [f32]) {
        match self.activation {
            ActivationVal::Unset => panic!("No activation function was set"),
            ActivationVal::Linear => self.forward_with_act::<activation::Linear>(input, output, buffer),
            ActivationVal::ReLU => self.forward_with_act::<activation::ReLU>(input, output, buffer),
        }
    }
    fn forward_with_act<Act: activation::ActivationType>(&self, input: &[f32], output: &mut [f32], buffer: &mut [f32]) {
        let buffer_len = *self.params.topology().max().expect("Number of layers is zero");
        assert!(buffer_len * 2<= buffer.len(), "Buffer size is too small");

        let (in_buff, out_buff) = buffer.split_at_mut(buffer_len/2 + 1);

        in_buff.copy_from_slice(input);

        for (bias_layer, weight_layer) in self.params.biases_iter().zip(self.params.weights_iter()) {
            for ((&bias, weights), neuron_output) in bias_layer.iter().zip(weight_layer).zip(out_buff.iter_mut()) {
                let mut raw = bias;

                for (&weight, &value) in weights.iter().zip(in_buff.iter()) {
                    raw += weight * value;
                }

                *neuron_output = Act::activate(raw);
            }
            in_buff.copy_from_slice(out_buff);
        }
        output.copy_from_slice(&out_buff[..output.len()]);
    }
}

impl FFNN for FFNNHeap {
    fn params(&self) -> &impl Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut impl Params {
        &mut self.params
    }
    
    fn forward_alloc(&self, input: &[f32]) -> Vec<f32> {
        let buff_len = *self.params.topology().max().expect("The ffnn doesn't have any layers") * 2;
        let output_len = *self.params.topology().last().expect("The ffnn doesn't have any layers");
        let mut buffer = vec![0_f32;buff_len];
        unsafe {
            let output = std::slice::from_raw_parts_mut(buffer.as_mut_ptr(), output_len);
            self.forward_with_buff(input, output, &mut buffer);
        }

        buffer.truncate(output_len);
        buffer
    }
}

#[derive(Clone)]
pub struct HeapBuilder {
    ffnn: Option<FFNNHeap>,
    activation: ActivationVal
}

impl Default for HeapBuilder  {
    fn default() -> Self {
        Self {
            ffnn: None,
            activation: ActivationVal::Unset
        }
    }
}

impl HeapBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn params_owned(mut self, params: ParamsHeap) -> Self {
        Self {
            ffnn: Some(match &mut self.ffnn {
                None => FFNNHeap::new(params, ActivationVal::Unset),
                Some(val) => FFNNHeap::new(params, val.activation),
            }),
            activation: ActivationVal::Unset
        }
    }

    pub fn params(self, params: &impl Params) -> Self {
        let params = ParamsHeap::create_from(params);
        Self {
            ffnn: Some(FFNNHeap::new(params, self.get_activation())),
            activation: ActivationVal::Unset
        }
    }

    pub fn topology_owned(self, topology: Vec<usize>) -> Self {
        let params = ParamsHeap::new(topology);
        Self {
            ffnn: Some(FFNNHeap::new(params, self.get_activation())),
            activation: ActivationVal::Unset
        }
    }
    
    pub fn topology(self, topology: &[usize]) -> Self {
        let params = ParamsHeap::new(topology.to_vec());
        Self {
            ffnn: Some(FFNNHeap::new(params, self.get_activation())),
            activation: ActivationVal::Unset
        }
    }

    pub fn activation(mut self, act: ActivationVal) -> Self {
        self.activation = act;
        if let Some(val) = &mut self.ffnn {
            val.activation = act;
        }
        self
    }

    fn get_activation(&self) -> ActivationVal {
        match &self.ffnn {
            None => self.activation,
            Some(val) => val.activation
        }
    }

    pub fn build(self) -> Option<FFNNHeap> {
        self.ffnn
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::*;

    use ffnn::heap_ffnn::*;

    #[test]
    fn check_forward() {
        let mut foo = HeapBuilder::new()
            .topology(&[2, 2, 1])
            .build().expect("Unable to build ffnn");

        for (weight, val) in foo.params_mut().weights_buff_mut().zip([1, 1, 0, 0, 1, 0].iter().copied()) {
            *weight = val as f32;
        }

        for (bias, val) in foo.params_mut().biases_buff_mut().zip([-1, 0, 0].iter().copied()) {
            *bias = val as f32;
        }

        let output = foo.forward_alloc(&[0_f32, 0_f32]);
        assert_eq!(output, vec![0_f32]);

        let output = foo.forward_alloc(&[0_f32, 1_f32]);
        assert_eq!(output, vec![0_f32]);

        let output = foo.forward_alloc(&[1_f32, 0_f32]);
        assert_eq!(output, vec![0_f32]);

        let output = foo.forward_alloc(&[1_f32, 1_f32]);
        assert_eq!(output, vec![1_f32]);
    }
}