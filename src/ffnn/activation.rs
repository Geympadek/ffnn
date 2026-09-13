#[inline]
fn relu(val: f32) -> f32 {
    val.max(0_f32)
}

#[inline]
fn linear(val: f32) -> f32 {
    val
}

pub trait ActivationType {
    fn activate(val: f32) -> f32;

    fn to_val() -> ActivationVal;
}

pub struct ReLU {}
impl ActivationType for ReLU {
    fn activate(val: f32) -> f32 {
        relu(val)
    }

    fn to_val() -> ActivationVal {
        ActivationVal::ReLU
    }
}

pub struct Linear {}
impl ActivationType for Linear {
    fn activate(val: f32) -> f32 {
        linear(val)
    }

    fn to_val() -> ActivationVal {
        ActivationVal::Linear
    }
}

///To be used when activation type is used dynamically
pub struct Unset {}
impl ActivationType for Unset {
    fn activate(_: f32) -> f32 {
        panic!("`UseVal` doesn't have an implementation for `activate` function. To fix this issue pass activation function to the FFNN constructor.")
    }

    fn to_val() -> ActivationVal {
        ActivationVal::Unset
    }
}

#[derive(Clone, Copy, Default)]
pub enum ActivationVal {
    #[default]
    Unset,
    ReLU,
    Linear,
}

impl ActivationVal {
    pub fn activate(&self, val: f32) -> f32 {
        match *self {
            ActivationVal::ReLU => relu(val),
            ActivationVal::Linear => linear(val),
            ActivationVal::Unset => panic!("No Activation function was specified for FFNN.")
        }
    }
}