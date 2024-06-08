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

use crate::vol::Vol;

pub trait NetLayer {
    fn forward(&mut self, in_act: &Vol, out_act: &mut Vol, is_training: bool);
    fn backward(&mut self, in_act: &mut Vol, out_act: &Vol);
    fn params_and_grads(&mut self) -> Vec<LayerDetails<'_>>;

    fn out_sx(&self) -> usize;
    fn out_sy(&self) -> usize;
    fn out_depth(&self) -> usize;
}

pub trait FinalLayer {
    fn forward(&mut self, in_act: &Vol, out_act: &mut Vol, is_training: bool);
    fn backward(&mut self, y: usize, in_act: &mut Vol, out_act: &Vol) -> f32;
    fn params_and_grads(&mut self) -> Vec<LayerDetails<'_>>;

    fn out_sx(&self) -> usize;
    fn out_sy(&self) -> usize;
    fn out_depth(&self) -> usize;
}

#[derive(Debug)]
pub struct LayerDetails<'a> {
    pub params: &'a mut [f32],
    pub grads: &'a mut [f32],
    pub l1_decay_mul: f32,
    pub l2_decay_mul: f32,
}
