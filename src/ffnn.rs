pub use crate::params::{Params, ParamsHeap};

pub mod activation;
pub mod heap_ffnn;
pub mod stack_ffnn;

pub use heap_ffnn::{FFNNHeap};

///A common trait for FFNN implementations.
pub trait FFNN {
    ///Feed forwards inputs through neural network, returning outputs in a new vector
    /// ## Parameters
    /// * `input` - Input slice, expected to be the same length as input layer
    /// # Panics
    /// Panics if `input` length does not equal input layer's length, or if activation function is not set
    fn forward_alloc(&self, input: &[f32]) -> Vec<f32>;

    ///Returns a reference to inner parameters structure
    fn params(&self) -> &impl Params;
    ///Returns a mutable reference to inner parameters structure
    fn params_mut(&mut self) -> &mut impl Params;
}