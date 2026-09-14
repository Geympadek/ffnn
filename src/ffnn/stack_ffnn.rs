use crate::{ffnn::{FFNN}, params::{self, Params}};
use params::stack_params::{self, Layer, ParamsStack};
use crate::ffnn::activation;
use std::marker::PhantomData;

///Stack implementation of Feed forwarding neural network. 
/// Doesn't allocate any heap memory unless prompted. 
/// Embeddes Layer structure inside of its type.
/// It is recommended to be used with macro `stack_ffnn!()`. 
/// Don't use this type directly, unless you know what you are doing.
#[derive(Clone, Copy)]
pub struct FFNNStack<Layers: Layer + Default, Activation: activation::ActivationType = activation::Unset> {
    params: ParamsStack<Layers>,
    activation: activation::ActivationVal,
    phantom: PhantomData<Activation>
}

impl<Layers: Layer + Default, Activation: activation::ActivationType> FFNNStack<Layers, Activation> {
    ///Creates a new instance of FFNN, while copying parameters from given refenence and setting activation function.
    pub fn new(params: &impl Params, activation: activation::ActivationVal) -> Self {
        Self {
            params: ParamsStack::create_from(params),
            activation: activation,
            phantom: PhantomData
        }
    }

    ///Creates a new instance of FFNN with activation function set to `activation`
    pub fn with_activation(activation: activation::ActivationVal) -> Self {
        let mut result = Self::default();
        result.activation = activation;
        result
    }

    ///Creates a new instance of FFNN while copying parameters from `params`
    pub fn from_params(params: &impl Params) -> Self {
        let mut result= Self::default();
        result.params = stack_params::ParamsStack::create_from(params);
        result
    }

    ///Feeds forward through FFNN and sends its outputs to `outputs` slice. Doesn't allocate any heap memory.
    /// # Panics
    /// May panic if `inputs` len and `outputs` len doesn't match FFNN layer structure.
    pub fn forward(&self, inputs: &[f32], outputs: &mut [f32]) {
        let layers = &self.params.layers;
        match self.activation {
            activation::ActivationVal::ReLU => layers.forward::<activation::ReLU>(inputs, outputs),
            activation::ActivationVal::Linear => layers.forward::<activation::Linear>(inputs, outputs),
            activation::ActivationVal::Unset => layers.forward::<Activation>(inputs, outputs),
        }
    }
}

impl<L: Layer + Default, A: activation::ActivationType> Default for FFNNStack<L, A> {
    fn default() -> Self {
        Self {
            params: Default::default(),
            activation: Default::default(),
            phantom: Default::default()
        }
    }
}

impl<L: Layer + Default, A: activation::ActivationType> FFNN<A> for FFNNStack<L, A> {
    fn forward_alloc(&self, input: &[f32]) -> Vec<f32> {
        let last_layer = self.params.topology().last().expect("Size of the FFNN is 0 layers.");
        let mut outputs = vec![0_f32;*last_layer];
        
        self.forward(input, outputs.as_mut_slice());
        outputs
    }

    fn params(&self) -> &impl Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut impl Params {
        &mut self.params
    }
}

/// Shorthand for getting FFNNStack<> type.
/// # Examples
/// ```
/// ```
#[macro_export]
macro_rules! stack_ffnn {
    ([$($dims:literal), +$(,)?], $activation:ty) => {
        crate::ffnn::stack_ffnn::FFNNStack<crate::create_stack_layers!([$($dims),+]), $activation>
    };
}

#[cfg(test)]
mod tests {
    use crate::{ffnn::FFNN, params::{Params}, params_stack};
    
    type StackFFNN = stack_ffnn!([2, 2, 1], crate::ffnn::activation::ReLU);
    #[test]
    fn check_forward() {
        let mut foo = StackFFNN::default();

        for (weight, val) in foo.params.weights_buff_mut().zip([1, 1, 0, 0, 1, 0].iter().copied()) {
            *weight = val as f32;
        }

        for (bias, val) in foo.params.biases_buff_mut().zip([-1, 0, 0].iter().copied()) {
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

    #[test]
    fn copy_test() {
        type Params = params_stack!([2, 3, 2]);
        let a: Params = Default::default();
        let mut b = a.clone();
        for val in b.iter_mut() {
            *val = 0.5f32;
        }
        let raw = a.construct_raw();
        assert_eq!(raw.iter().sum::<f32>(), 0_f32);
        let raw = b.construct_raw();
        assert_eq!(raw, vec![0.5f32; raw.len()]);
    }
}