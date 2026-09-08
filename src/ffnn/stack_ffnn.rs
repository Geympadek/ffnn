use crate::{ffnn::{FFNN}, params::{self, Params}};
use params::stack_params::{self, Layer, ParamsStack};
use crate::ffnn::activation;
use std::marker::PhantomData;


pub struct FFNNStack<Layers: Layer, Activation: activation::ActivationType = activation::Unset> {
    params: ParamsStack<Layers>,
    activation: activation::ActivationVal,
    phantom: PhantomData<Activation>
}

impl<Layers: Layer, Activation: activation::ActivationType> FFNNStack<Layers, Activation> {
    pub fn new() -> Self {
        Self {
            params: ParamsStack::new(),
            activation: activation::ActivationVal::Unset,
            phantom: PhantomData
        }
    }

    pub fn forward(&self, inputs: &[f32], outputs: &mut [f32]) {
        let layers = &self.params.layers;
        match self.activation {
            activation::ActivationVal::ReLU => layers.forward::<activation::ReLU>(inputs, outputs),
            activation::ActivationVal::Linear => layers.forward::<activation::Linear>(inputs, outputs),
            activation::ActivationVal::Unset => layers.forward::<Activation>(inputs, outputs),
        }
    }
}

impl<L: Layer, A: activation::ActivationType> FFNN<A> for FFNNStack<L, A> {
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