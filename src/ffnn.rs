use crate::params::Params;

pub mod activation;
pub mod heap_ffnn;
pub mod stack_ffnn;

pub trait FFNN<Act: activation::ActivationType> {
    fn forward_alloc(&self, input: &[i32]) -> Vec<f32>;

    fn params(&self) -> &impl Params;
    fn params_mut(&mut self) -> &mut impl Params;
}