mod conv_layer;
mod input_layer;
mod loss_layer;
mod nonlinear_layer;
mod pool_layer;

pub use conv_layer::*;
pub use input_layer::*;
pub use loss_layer::*;
pub use nonlinear_layer::*;
pub use pool_layer::*;

use crate::{vol::Vol, Float};

#[typetag::serde(tag = "type")]
pub trait NetLayer: Send {
    fn name(&self) -> &str;
    fn forward(&mut self, in_act: &Vol, out_act: &mut Vol, is_training: bool);
    fn backward(&mut self, in_act: &mut Vol, out_act: &Vol);
    fn params_and_grads(&mut self) -> Vec<LayerDetails<'_>>;

    fn weights(&self) -> Vec<&Vol> {
        Vec::new()
    }

    fn out_sx(&self) -> usize;
    fn out_sy(&self) -> usize;
    fn out_depth(&self) -> usize;
}

#[typetag::serde(tag = "type")]
pub trait FinalLayer: Send {
    fn name(&self) -> &str;
    fn forward(&mut self, in_act: &Vol, out_act: &mut Vol, is_training: bool);
    fn backward(&mut self, y: usize, in_act: &mut Vol, out_act: &Vol) -> Float;
    fn params_and_grads(&mut self) -> Vec<LayerDetails<'_>>;

    fn weights(&self) -> Vec<&Vol> {
        Vec::new()
    }

    fn out_sx(&self) -> usize;
    fn out_sy(&self) -> usize;
    fn out_depth(&self) -> usize;
}

#[derive(Debug)]
pub struct LayerDetails<'a> {
    pub params: &'a mut [Float],
    pub grads: &'a mut [Float],
    pub l1_decay_mul: Float,
    pub l2_decay_mul: Float,
}
