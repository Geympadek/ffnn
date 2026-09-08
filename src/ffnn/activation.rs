#[inline]
fn relu(val: f32) -> f32 {
    val.max(0_f32)
}

#[inline]
fn linear(val: f32) -> f32 {
    val
}

pub trait ActivationType {
    fn activate(val: f32) -> Option<f32>;
}

pub struct ReLU {}
impl ActivationType for ReLU {
    fn activate(val: f32) -> Option<f32> {
        Some(relu(val))
    }
}

pub struct Linear {}
impl ActivationType for Linear {
    fn activate(val: f32) -> Option<f32> {
        Some(linear(val))
    }
}

///To be used when activation type is used dynamically
pub struct UseVal {}
impl ActivationType for UseVal {
    fn activate(_: f32) -> Option<f32> {
        None
    }
}

pub enum ActivationVal {
    ReLU,
    Linear
}

impl ActivationVal {
    pub fn activate(&self, val: f32) -> f32 {
        match *self {
            ActivationVal::ReLU => relu(val),
            ActivationVal::Linear => linear(val)
        }
    }
}