//! This file contains all layers that do dot products with input,
//! but usually in a different connectivity pattern and weight sharing
//! schemes:
//! - FullyConn is fully connected dot products
//! - ConvLayer does convolutions (so weight sharing spatially)
//! putting them together in one file because they are very similar

use crate::{vol::Vol, Float};

use super::NetLayer;

pub struct ConvLayer {
    pub(crate) out_depth: usize,
    sx: usize,
    sy: usize,

    in_depth: usize,
    in_sx: usize,
    in_sy: usize,

    stride: usize,
    padding: usize,
    l1_decay_mul: Float,
    l2_decay_mul: Float,

    pub(crate) out_sx: usize,
    pub(crate) out_sy: usize,

    bias: Float,

    filters: Vec<Vol>,
    biases: Vol,
}

impl ConvLayer {
    pub fn builder(
        filters: usize,
        sx: usize,
        in_depth: usize,
        in_sx: usize,
        in_sy: usize,
    ) -> ConvLayerBuilder {
        ConvLayerBuilder::new(filters, sx, in_depth, in_sx, in_sy)
    }
}

pub struct ConvLayerBuilder {
    out_depth: usize,
    sx: usize,

    in_depth: usize,
    in_sx: usize,
    in_sy: usize,

    sy: usize,
    stride: usize,
    padding: usize,
    l1_decay_mul: Float,
    l2_decay_mul: Float,

    bias: Float,
}

impl ConvLayerBuilder {
    pub fn new(filters: usize, sx: usize, in_depth: usize, in_sx: usize, in_sy: usize) -> Self {
        Self {
            // required
            out_depth: filters,
            sx,
            in_depth,
            in_sx,
            in_sy,

            // optional
            sy: sx,
            stride: 1,
            padding: 0,
            l1_decay_mul: 0.0,
            l2_decay_mul: 1.0,
            bias: 0.0,
        }
    }

    pub fn sy(mut self, value: usize) -> Self {
        self.sy = value;
        self
    }

    /// stride at which we apply filters to input volume
    ///
    /// default: 1
    pub fn stride(mut self, value: usize) -> Self {
        self.stride = value;
        self
    }

    /// padding to add around borders of input volume
    ///
    /// default: 0
    pub fn padding(mut self, value: usize) -> Self {
        self.padding = value;
        self
    }

    /// default: 0.0
    pub fn l1_decay_mul(mut self, value: Float) -> Self {
        self.l1_decay_mul = value;
        self
    }

    /// default: 1.0
    pub fn l2_decay_mul(mut self, value: Float) -> Self {
        self.l2_decay_mul = value;
        self
    }

    /// default: 0.0
    pub fn bias(mut self, value: Float) -> Self {
        self.bias = value;
        self
    }

    pub fn build(self) -> ConvLayer {
        let filters = (0..self.out_depth)
            .map(|_| Vol::new(self.sx, self.sy, self.in_depth))
            .collect();
        ConvLayer {
            out_depth: self.out_depth,
            sx: self.sx,
            sy: self.sy,
            in_depth: self.in_depth,
            in_sx: self.in_sx,
            in_sy: self.in_sy,
            stride: self.stride,
            padding: self.padding,
            l1_decay_mul: self.l1_decay_mul,
            l2_decay_mul: self.l2_decay_mul,
            bias: self.bias,
            // computed
            // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
            // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
            // final application.
            out_sx: ((self.in_sx + self.padding * 2 - self.sx) / self.stride + 1),
            out_sy: ((self.in_sy + self.padding * 2 - self.sy) / self.stride + 1),
            filters,
            biases: Vol::with_constant(1, 1, self.out_depth, self.bias),
        }
    }
}

impl NetLayer for ConvLayer {
    fn forward(&mut self, in_act: &Vol, out_act: &mut Vol, _is_training: bool) {
        let v_sx = in_act.sx() as isize;
        let v_sy = in_act.sy() as isize;
        let xy_stride = self.stride as isize;

        for d in 0..self.out_depth {
            let f = &mut self.filters[d];
            let mut x;
            let mut y = -(self.padding as isize);

            // for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
            for ay in 0..self.out_sy {
                x = -(self.padding as isize);

                // for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride
                for ax in 0..self.out_sx {
                    // convolve centered at this particular location
                    let mut a = 0.0;

                    // for(var fy=0;fy<f.sy;fy++) {
                    for fy in 0..f.sy() as isize {
                        let oy = y + fy; // coordinates in the original input array coordinates

                        for fx in 0..f.sx() as isize {
                            let ox = x + fx;

                            // if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                            if oy >= 0 && oy < v_sy && ox >= 0 && ox < v_sx {
                                for fd in 0..f.depth() as isize {
                                    let f_index =
                                        ((f.sx() as isize * fy) + fx) * f.depth() as isize + fd;
                                    let v_index = ((v_sx * oy) + ox) * in_act.depth() as isize + fd;

                                    debug_assert!(f_index >= 0, "filter index must be >= 0");
                                    debug_assert!(v_index >= 0, "in vol index must be >= 0");

                                    a += f.w[f_index as usize] * in_act.w[v_index as usize];
                                }
                            }
                        }
                    }

                    a += self.biases.w[d];
                    out_act.set(ax, ay, d, a);

                    x += xy_stride;
                }

                y += xy_stride;
            }
        }
    }

    fn backward(&mut self, in_act: &mut Vol, out_act: &Vol) {
        let v = in_act;

        debug_assert_eq!(
            v.w.len(),
            v.dw.len(),
            "weight arrays should have the same length"
        );

        v.dw.fill(0.0); // zero out gradient wrt bottom data, we're about to fill it

        // var V_sx = V.sx |0;
        // var V_sy = V.sy |0;
        // var xy_stride = this.stride |0;
        let v_sx = v.sx() as isize;
        let v_sy = v.sy() as isize;
        let xy_stride = self.stride as isize;

        // for(var d=0;d<this.out_depth;d++) {
        for d in 0..self.out_depth {
            // var f = this.filters[d];
            let f = &mut self.filters[d];
            // var x = -this.pad |0;
            let mut x;
            // var y = -this.pad |0;
            let mut y = -(self.padding as isize);

            // for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
            for ay in 0..self.out_sy {
                // x = -this.pad |0;
                x = -(self.padding as isize);

                // for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride
                for ax in 0..self.out_sx {
                    // convolve centered at this particular location
                    // var chain_grad = this.out_act.get_grad(ax,ay,d); // gradient from above, from chain rule
                    let chain_grad = out_act.get_gradiant(ax, ay, d); // gradient from above, from chain rule

                    // for(var fy=0;fy<f.sy;fy++) {
                    for fy in 0..f.sy() as isize {
                        // var oy = y+fy; // coordinates in the original input array coordinates
                        let oy = y + fy;

                        // for(var fx=0;fx<f.sx;fx++) {
                        for fx in 0..f.sx() as isize {
                            // var ox = x+fx;
                            let ox = x + fx;

                            // if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                            if oy >= 0 && oy < v_sy && ox >= 0 && ox < v_sx {
                                // for(var fd=0;fd<f.depth;fd++) {
                                for fd in 0..f.depth() as isize {
                                    // // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    // var ix1 = ((V_sx * oy)+ox)*V.depth+fd;
                                    let ix1 = ((v_sx * oy) + ox) * v.depth() as isize + fd;
                                    // var ix2 = ((f.sx * fy)+fx)*f.depth+fd;
                                    let ix2 =
                                        ((f.sx() as isize * fy) + fx) * f.depth() as isize + fd;

                                    f.dw[ix2 as usize] += v.w[ix1 as usize] * chain_grad;
                                    v.dw[ix1 as usize] += f.w[ix2 as usize] * chain_grad;
                                    // }
                                }
                                // }
                            }

                            // }
                        }
                        // }
                    }

                    // this.biases.dw[d] += chain_grad;
                    self.biases.dw[d] += chain_grad;

                    // }
                    x += xy_stride;
                }

                // }
                y += xy_stride;
            }
            // }
        }
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
        let mut result = Vec::new();
        for filter in &mut self.filters {
            result.push(super::LayerDetails {
                params: &mut filter.w,
                grads: &mut filter.dw,
                l1_decay_mul: self.l1_decay_mul,
                l2_decay_mul: self.l2_decay_mul,
            });
        }

        result.push(super::LayerDetails {
            params: &mut self.biases.w,
            grads: &mut self.biases.dw,
            l1_decay_mul: 0.0,
            l2_decay_mul: 0.0,
        });
        result
    }
}

pub struct FullyConnLayer {
    // required
    out_depth: usize,

    // optional
    l1_decay_mul: Float,
    l2_decay_mul: Float,
    bias: Float,

    // computed
    num_inputs: usize,
    out_sx: usize,
    out_sy: usize,

    filters: Vec<Vol>,
    biases: Vol,
}

impl FullyConnLayer {
    pub fn builder(
        num_neurons: usize,
        in_sx: usize,
        in_sy: usize,
        in_depth: usize,
    ) -> FullyConnLayerBuilder {
        FullyConnLayerBuilder::new(num_neurons, in_sx, in_sy, in_depth)
    }
}

impl NetLayer for FullyConnLayer {
    fn forward(&mut self, in_act: &Vol, out_act: &mut Vol, _is_training: bool) {
        // this.in_act = V;
        // var A = new Vol(1, 1, this.out_depth, 0.0);
        debug_assert_eq!(out_act.sx(), 1);
        debug_assert_eq!(out_act.sy(), 1);
        debug_assert_eq!(out_act.depth(), self.out_depth);
        // var Vw = V.w;
        // for(var i=0;i<this.out_depth;i++) {
        for i in 0..self.out_depth {
            // var a = 0.0;
            let mut a = 0.0;

            // var wi = this.filters[i].w;
            let wi = &self.filters[i].w;

            // for(var d=0;d<this.num_inputs;d++) {
            for d in 0..self.num_inputs {
                a += in_act.w[d] * wi[d]; // for efficiency use Vols directly for now
            }
            a += self.biases.w[i];
            out_act.w[i] = a;
        }
        // this.out_act = A;
        // return this.out_act;
    }

    fn backward(&mut self, in_act: &mut Vol, out_act: &Vol) {
        // var V = this.in_act;
        // V.dw = global.zeros(V.w.length); // zero out the gradient in input Vol
        in_act.dw.fill(0.0);

        // // compute gradient wrt weights and data
        // for(var i=0;i<this.out_depth;i++) {
        for i in 0..self.out_depth {
            // var tfi = this.filters[i];
            let tfi = &mut self.filters[i];

            // var chain_grad = this.out_act.dw[i];
            let chain_grad = out_act.dw[i];

            // for(var d=0;d<this.num_inputs;d++) {
            for d in 0..self.num_inputs {
                // V.dw[d] += tfi.w[d]*chain_grad; // grad wrt input data
                in_act.dw[d] += tfi.w[d] * chain_grad;
                // tfi.dw[d] += V.w[d]*chain_grad; // grad wrt params
                tfi.dw[d] += in_act.w[d] * chain_grad;
            }
            // this.biases.dw[i] += chain_grad;
            self.biases.dw[i] += chain_grad;
        }
    }

    fn out_sx(&self) -> usize {
        self.out_sx
    }

    fn out_depth(&self) -> usize {
        self.out_depth
    }

    fn out_sy(&self) -> usize {
        self.out_sy
    }

    fn params_and_grads(&mut self) -> Vec<super::LayerDetails<'_>> {
        let mut result = Vec::new();
        for filter in &mut self.filters {
            result.push(super::LayerDetails {
                params: &mut filter.w,
                grads: &mut filter.dw,
                l1_decay_mul: self.l1_decay_mul,
                l2_decay_mul: self.l2_decay_mul,
            });
        }

        result.push(super::LayerDetails {
            params: &mut self.biases.w,
            grads: &mut self.biases.dw,
            l1_decay_mul: 0.0,
            l2_decay_mul: 0.0,
        });
        result
    }
}

pub struct FullyConnLayerBuilder {
    // required
    out_depth: usize,

    // optional
    l1_decay_mul: Float,
    l2_decay_mul: Float,
    bias: Float,

    in_sx: usize,
    in_sy: usize,
    in_depth: usize,
}

impl FullyConnLayerBuilder {
    fn new(num_neurons: usize, in_sx: usize, in_sy: usize, in_depth: usize) -> Self {
        Self {
            // required
            out_depth: num_neurons,

            l1_decay_mul: 0.0,
            l2_decay_mul: 1.0,
            bias: 0.0,

            in_sx,
            in_sy,
            in_depth,
        }
    }

    /// default: 0.0
    pub fn l1_decay_mul(mut self, value: Float) -> Self {
        self.l1_decay_mul = value;
        self
    }
    /// default: 1.0
    pub fn l2_decay_mul(mut self, value: Float) -> Self {
        self.l2_decay_mul = value;
        self
    }
    /// default: 0.0
    pub fn bias(mut self, value: Float) -> Self {
        self.bias = value;
        self
    }

    pub fn build(self) -> FullyConnLayer {
        let num_inputs = self.in_sx * self.in_sy * self.in_depth;

        let filters = (0..self.out_depth)
            .map(|_| Vol::new(1, 1, num_inputs))
            .collect();

        let biases = Vol::with_constant(1, 1, self.out_depth, self.bias);
        FullyConnLayer {
            out_depth: self.out_depth,

            l1_decay_mul: self.l1_decay_mul,
            l2_decay_mul: self.l2_decay_mul,
            bias: self.bias,

            // computed
            num_inputs,
            out_sx: 1,
            out_sy: 1,
            filters,
            biases,
        }
    }
}
