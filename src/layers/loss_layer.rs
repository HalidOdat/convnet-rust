// Layers that implement a loss. Currently these are the layers that
// can initiate a backward() pass. In future we probably want a more
// flexible system that can accomodate multiple losses to do multi-task
// learning, and stuff like that. But for now, one of the layers in this
// file must be the final layer in a Net.

use crate::{vol::Vol, Float};

use super::{FinalLayer, NetLayer};

/// This is a classifier, with N discrete classes from 0 to N-1
/// it gets a stream of N incoming numbers and computes the softmax
/// function (exponentiate and normalize to sum to 1 as probabilities should)
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SofmaxLayer {
    num_inputs: usize,
    out_depth: usize,
    out_sx: usize,
    out_sy: usize,

    es: Vec<Float>,
}

impl SofmaxLayer {
    pub fn new(in_sx: usize, in_sy: usize, in_depth: usize) -> Self {
        let num_inputs = in_sx * in_sy * in_depth;
        Self {
            num_inputs,
            out_depth: num_inputs,
            out_sx: 1,
            out_sy: 1,

            es: vec![0.0; num_inputs],
        }
    }
}

#[typetag::serde]
impl FinalLayer for SofmaxLayer {
    fn forward(
        &mut self,
        in_act: &crate::vol::Vol,
        out_act: &mut crate::vol::Vol,
        _is_training: bool,
    ) {
        let v = in_act;

        debug_assert_eq!(out_act.sx(), 1);
        debug_assert_eq!(out_act.sy(), 1);
        debug_assert_eq!(out_act.depth(), self.out_depth);
        // *out_act = Vol::zeros(1, 1, self.out_depth);

        // compute max activation
        // var amax = V.w[0];
        let mut amax = v.w[0];
        // for(var i=1;i<this.out_depth;i++) {
        for i in 1..self.out_depth {
            if v.w[i] > amax {
                amax = v.w[i];
            }
        }

        // compute exponentials (carefully to not blow up)
        // var es = global.zeros(this.out_depth);
        // TODO: Could remove!
        self.es.fill(0.0);

        // var esum = 0.0;
        let mut esum = 0.0;

        // for(var i=0;i<this.out_depth;i++) {
        for i in 0..self.out_depth {
            // var e = Math.exp(as[i] - amax);
            let e = (v.w[i] - amax).exp();
            // esum += e;
            esum += e;
            // es[i] = e;
            self.es[i] = e;
        }

        // normalize and output to sum to one
        // for(var i=0;i<this.out_depth;i++) {
        for i in 0..self.out_depth {
            // es[i] /= esum;
            self.es[i] /= esum;
            // A.w[i] = es[i];
            out_act.w[i] = self.es[i];
        }

        // this.es = es; // save these for backprop
        // this.out_act = A;
    }

    fn backward(&mut self, y: usize, in_act: &mut Vol, out_act: &Vol) -> Float {
        // compute and accumulate gradient wrt weights and bias of this layer
        // var x = this.in_act;
        let x = in_act;

        // x.dw = global.zeros(x.w.length); // zero out the gradient of input Vol
        x.dw.fill(0.0);

        // for(var i=0;i<this.out_depth;i++) {
        for i in 0..self.out_depth {
            // var indicator = i === y ? 1.0 : 0.0;
            let indicator = Float::from(i == y);

            // var mul = -(indicator - this.es[i]);
            let mul = -(indicator - self.es[i]);

            // x.dw[i] = mul;
            x.dw[i] = mul;
        }

        // loss is the class negative log likelihood
        // return -Math.log(this.es[y]);
        -(self.es[y].ln())
    }

    fn out_sx(&self) -> usize {
        self.out_sx
    }
    fn out_sy(&self) -> usize {
        self.out_sy
    }
    fn out_depth(&self) -> usize {
        self.out_depth
    }

    fn params_and_grads(&mut self) -> Vec<super::LayerDetails<'_>> {
        Vec::new()
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct RegressionLayer {
    num_inputs: usize,
    out_depth: usize,
    out_sx: usize,
    out_sy: usize,
}

impl RegressionLayer {
    pub fn new(in_sx: usize, in_sy: usize, in_depth: usize) -> Self {
        let num_inputs = in_sx * in_sy * in_depth;
        Self {
            num_inputs,
            out_depth: num_inputs,
            out_sx: 1,
            out_sy: 1,
        }
    }
}

#[typetag::serde]
impl NetLayer for RegressionLayer {
    fn forward(&mut self, in_act: &Vol, out_act: &mut Vol, _is_training: bool) {
        *out_act = in_act.clone();
    }

    fn backward(&mut self, in_act: &mut Vol, _out_act: &Vol) {
        // // compute and accumulate gradient wrt weights and bias of this layer
        // var x = this.in_act;
        let x = in_act;

        // x.dw = global.zeros(x.w.length); // zero out the gradient of input Vol
        x.dw.fill(0.0);

        // var loss = 0.0;
        let mut loss = 0.0;

        // if(y instanceof Array || y instanceof Float64Array) {
        //     for(var i=0;i<this.out_depth;i++) {
        //     var dy = x.w[i] - y[i];
        //     x.dw[i] = dy;
        //     loss += 0.5*dy*dy;
        //     }
        // } else if(typeof y === 'number') {
        //     // lets hope that only one number is being regressed
        //     var dy = x.w[0] - y;
        //     x.dw[0] = dy;
        //     loss += 0.5*dy*dy;
        // } else {
        //     // assume it is a struct with entries .dim and .val
        //     // and we pass gradient only along dimension dim to be equal to val
        //     var i = y.dim;
        //     var yi = y.val;
        //     var dy = x.w[i] - yi;
        //     x.dw[i] = dy;
        //     loss += 0.5*dy*dy;
        // }
        // return loss;

        todo!()
    }

    fn out_depth(&self) -> usize {
        self.out_depth
    }
    fn out_sx(&self) -> usize {
        self.out_sx
    }
    fn out_sy(&self) -> usize {
        self.out_sy
    }

    fn params_and_grads(&mut self) -> Vec<super::LayerDetails<'_>> {
        todo!()
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SVMLayer {
    num_inputs: usize,
    out_depth: usize,
    out_sx: usize,
    out_sy: usize,
}

impl SVMLayer {
    pub fn new(in_sx: usize, in_sy: usize, in_depth: usize) -> Self {
        let num_inputs = in_sx * in_sy * in_depth;
        Self {
            num_inputs,
            out_depth: num_inputs,
            out_sx: 1,
            out_sy: 1,
        }
    }
}

#[typetag::serde]
impl FinalLayer for SVMLayer {
    fn forward(&mut self, in_act: &Vol, out_act: &mut Vol, _is_training: bool) {
        *out_act = in_act.clone();
    }

    fn backward(&mut self, y: usize, in_act: &mut Vol, _out_act: &Vol) -> Float {
        // compute and accumulate gradient wrt weights and bias of this layer
        // var x = this.in_act;
        let x = in_act;

        // x.dw = global.zeros(x.w.length); // zero out the gradient of input Vol
        x.dw.fill(0.0);

        // // we're using structured loss here, which means that the score
        // // of the ground truth should be higher than the score of any other
        // // class, by a margin
        // var yscore = x.w[y]; // score of ground truth
        let yscore: Float = x.w[y];

        // var margin = 1.0;
        let margin = 1.0;

        // var loss = 0.0;
        let mut loss = 0.0;

        // for(var i=0;i<this.out_depth;i++) {
        for i in 0..self.out_depth {
            // if(y === i) { continue; }
            if
            /* y */
            1 == i {
                continue;
            }
            // var ydiff = -yscore + x.w[i] + margin;
            let ydiff = -yscore + x.w[i] + margin;
            if ydiff > 0.0 {
                // violating dimension, apply loss
                x.dw[i] += 1.0;
                x.dw[y] -= 1.0;
                loss += ydiff;
            }
        }

        loss
    }

    fn out_sx(&self) -> usize {
        self.out_sx
    }
    fn out_sy(&self) -> usize {
        self.out_sy
    }
    fn out_depth(&self) -> usize {
        self.out_depth
    }

    fn params_and_grads(&mut self) -> Vec<super::LayerDetails<'_>> {
        Vec::new()
    }
}
