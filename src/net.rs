use crate::{
    layers::{
        ConvLayer, FinalLayer, FullyConnLayer, InputLayer, LayerDetails, NetLayer, PoolLayer,
        ReluLayer, SofmaxLayer, TanhLayer,
    },
    vol::Vol,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Tanh,
}

#[derive(Debug, Clone, Copy)]
pub enum Layer {
    Input {
        width: usize,
        height: usize,
        depth: usize,
    },
    Conv {
        sx: usize,
        filters: usize,
        stride: usize,
        padding: usize,
        activation: Activation,
    },
    Pool {
        sx: usize,
        stride: usize,
    },
    Dense {
        neurons: usize,
        activation: Activation,
    },
}

pub enum EndLayer {
    Softmax { classes: usize },
}

// Net manages a set of layers
// For now constraints: Simple linear order of layers, first layer input last layer a cost layer
pub struct Net {
    layers: Vec<Box<dyn NetLayer>>,
    final_layer: Box<dyn FinalLayer>,
    acts: Vec<Vol>,
}

impl Net {
    pub fn new(def_layers: &[Layer], def_final_layer: EndLayer) -> Self {
        let mut layers: Vec<Box<dyn NetLayer>> = Vec::new();

        #[derive(Clone, Copy)]
        struct Dim {
            in_sx: usize,
            in_sy: usize,
            in_depth: usize,
        }

        let mut dim_set = false;
        let mut dim = Dim {
            in_depth: usize::MAX,
            in_sx: usize::MAX,
            in_sy: usize::MAX,
        };

        let mut acts = Vec::new();
        for def in def_layers {
            match *def {
                Layer::Input {
                    width,
                    height,
                    depth,
                } => {
                    if dim_set {
                        panic!("input layer must be the first");
                    }
                    let layer: Box<dyn NetLayer> =
                        Box::new(InputLayer::with_dimensions(depth, width, height));
                    layers.push(layer);

                    dim_set = true;
                    dim = Dim {
                        in_sx: width,
                        in_sy: height,
                        in_depth: depth,
                    };

                    acts.push(Vol::zeros(dim.in_sx, dim.in_sy, dim.in_depth));
                }
                Layer::Conv {
                    sx,
                    filters,
                    stride,
                    padding,
                    activation,
                } => {
                    if !dim_set {
                        panic!("dim must be specified");
                    };
                    let layer = ConvLayer::builder(filters, sx, dim.in_depth, dim.in_sx, dim.in_sy)
                        .padding(padding)
                        .stride(stride)
                        // relus like a bit of positive bias to get gradients early
                        // otherwise it's technically possible that a relu unit will never turn on (by chance)
                        // and will never get any gradient and never contribute any computation. Dead relu.
                        .bias(if activation == Activation::Relu {
                            0.1
                        } else {
                            0.0
                        })
                        .build();

                    dim = Dim {
                        in_sx: layer.out_sx(),
                        in_sy: layer.out_sy(),
                        in_depth: layer.out_depth(),
                    };

                    layers.push(Box::new(layer));
                    acts.push(Vol::zeros(dim.in_sx, dim.in_sy, dim.in_depth));

                    let layer: Box<dyn NetLayer> = match activation {
                        Activation::Relu => {
                            Box::new(ReluLayer::new(dim.in_sx, dim.in_sy, dim.in_depth))
                        }
                        Activation::Tanh => {
                            Box::new(TanhLayer::new(dim.in_sx, dim.in_sy, dim.in_depth))
                        }
                    };

                    dim = Dim {
                        in_sx: layer.out_sx(),
                        in_sy: layer.out_sy(),
                        in_depth: layer.out_depth(),
                    };

                    layers.push(layer);
                    acts.push(Vol::zeros(dim.in_sx, dim.in_sy, dim.in_depth));
                }
                Layer::Pool { sx, stride } => {
                    if !dim_set {
                        panic!("dim must be specified");
                    };
                    let layer: Box<dyn NetLayer> = Box::new(
                        PoolLayer::builder(sx, dim.in_depth, dim.in_sx, dim.in_sy)
                            .stride(stride)
                            .build(),
                    );

                    dim = Dim {
                        in_sx: layer.out_sx(),
                        in_sy: layer.out_sy(),
                        in_depth: layer.out_depth(),
                    };

                    layers.push(layer);
                    acts.push(Vol::zeros(dim.in_sx, dim.in_sy, dim.in_depth));
                }
                Layer::Dense {
                    neurons,
                    activation,
                } => {
                    if !dim_set {
                        panic!("dim must be specified");
                    };
                    let layer =
                        FullyConnLayer::builder(neurons, dim.in_sx, dim.in_sy, dim.in_depth)
                            // relus like a bit of positive bias to get gradients early
                            // otherwise it's technically possible that a relu unit will never turn on (by chance)
                            // and will never get any gradient and never contribute any computation. Dead relu.
                            .bias(if activation == Activation::Relu {
                                0.1
                            } else {
                                0.0
                            })
                            .build();

                    dim = Dim {
                        in_sx: layer.out_sx(),
                        in_sy: layer.out_sy(),
                        in_depth: layer.out_depth(),
                    };

                    layers.push(Box::new(layer));
                    acts.push(Vol::zeros(dim.in_sx, dim.in_sy, dim.in_depth));

                    let layer: Box<dyn NetLayer> = match activation {
                        Activation::Relu => {
                            Box::new(ReluLayer::new(dim.in_sx, dim.in_sy, dim.in_depth))
                        }
                        Activation::Tanh => {
                            Box::new(TanhLayer::new(dim.in_sx, dim.in_sy, dim.in_depth))
                        }
                    };

                    dim = Dim {
                        in_sx: layer.out_sx(),
                        in_sy: layer.out_sy(),
                        in_depth: layer.out_depth(),
                    };

                    layers.push(layer);
                    acts.push(Vol::zeros(dim.in_sx, dim.in_sy, dim.in_depth));
                }
            }
        }

        let final_layer: Box<dyn FinalLayer> = match def_final_layer {
            EndLayer::Softmax { classes } => {
                if !dim_set {
                    panic!("dim must be specified for final layer");
                };

                // add an fc layer here, there is no reason the user should
                // have to worry about this and we almost always want to
                let layer = FullyConnLayer::builder(classes, dim.in_sx, dim.in_sy, dim.in_depth)
                    // relus like a bit of positive bias to get gradients early
                    // otherwise it's technically possible that a relu unit will never turn on (by chance)
                    // and will never get any gradient and never contribute any computation. Dead relu.
                    .bias(0.1)
                    .build();

                dim = Dim {
                    in_sx: layer.out_sx(),
                    in_sy: layer.out_sy(),
                    in_depth: layer.out_depth(),
                };

                layers.push(Box::new(layer));
                acts.push(Vol::zeros(dim.in_sx, dim.in_sy, dim.in_depth));

                let layer = Box::new(SofmaxLayer::new(dim.in_sx, dim.in_sy, dim.in_depth));

                dim = Dim {
                    in_sx: layer.out_sx(),
                    in_sy: layer.out_sy(),
                    in_depth: layer.out_depth(),
                };
                acts.push(Vol::zeros(dim.in_sx, dim.in_sy, dim.in_depth));

                layer
            }
        };
        Self {
            layers,
            final_layer,
            acts,
        }
    }

    fn adjecent(acts: &mut [Vol], at: usize) -> (&mut Vol, &mut Vol) {
        let (a, b) = acts.split_at_mut(at);
        (
            a.last_mut().expect("should be at least one element"),
            &mut b[0],
        )
    }

    // forward prop the network.
    // The trainer class passes is_training = true, but when this function is
    // called from outside (not from the trainer), it defaults to prediction mode
    pub fn forward(&mut self, vol: &Vol, is_training: bool) -> Vol {
        debug_assert_eq!(self.layers.len() + 1, self.acts.len());

        let n = self.acts.len();

        self.layers[0].forward(vol, &mut self.acts[0], is_training);
        for i in 1..self.layers.len() {
            let (a, b) = Self::adjecent(&mut self.acts, i);
            self.layers[i].forward(a, b, is_training);
        }

        let (a, b) = Self::adjecent(&mut self.acts, n - 1);
        self.final_layer.forward(a, b, is_training);

        b.clone()
    }

    // backprop: compute gradients wrt all parameters
    pub fn backward(&mut self, y: usize, vol: &mut Vol) -> f32 {
        let n = self.acts.len();

        let (a, b) = Self::adjecent(&mut self.acts, n - 1);
        let loss = self.final_layer.backward(y, a, b);

        for i in (1..self.layers.len()).rev() {
            let (a, b) = Self::adjecent(&mut self.acts, i);
            self.layers[i].backward(a, b);
        }

        self.layers[0].backward(vol, &self.acts[0]);

        loss
    }

    // this is a convenience function for returning the argmax
    // prediction, assuming the last layer of the net is a softmax
    pub fn get_prediction(&self) -> usize {
        // var S = this.layers[this.layers.length-1];
        let s = &self.final_layer;
        // assert(S.layer_type === 'softmax', 'getPrediction function assumes softmax as last layer of the net!');

        // var p = S.out_act.w;

        // var maxv = p[0];
        // var maxi = 0;
        // for(var i=1;i<p.length;i++) {
        //     if(p[i] > maxv) { maxv = p[i]; maxi = i;}
        // }
        // return maxi; // return index of the class with highest class probability
        todo!()
    }

    pub fn get_cost_loss(&mut self, vol: &Vol, y: usize) -> f32 {
        self.forward(vol, false);

        let n = self.acts.len();
        let (in_act, out_act) = Self::adjecent(&mut self.acts, n - 1);

        self.final_layer.backward(y, in_act, out_act)
    }

    pub fn params_and_grads(&mut self) -> Vec<LayerDetails> {
        let mut result = Vec::new();
        for layer in &mut self.layers {
            let responses = layer.params_and_grads();
            for response in responses {
                result.push(response);
            }
        }
        let responses = self.final_layer.params_and_grads();
        for response in responses {
            result.push(response);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use rand::random;

    use crate::{vol::Vol, Trainer};

    use super::{Activation, EndLayer, Layer, Net};

    fn setup() -> Net {
        Net::new(
            &[
                Layer::Input {
                    width: 1,
                    height: 1,
                    depth: 2,
                },
                Layer::Dense {
                    neurons: 5,
                    activation: Activation::Tanh,
                },
                Layer::Dense {
                    neurons: 5,
                    activation: Activation::Tanh,
                },
            ],
            EndLayer::Softmax { classes: 3 },
        )
    }

    #[test]
    fn layer_count() {
        let net = setup();

        // tanh are their own layers. Softmax gets its own fully connected layer.
        // this should all get desugared just fine.
        assert_eq!(net.layers.len(), 6);
    }

    #[test]
    fn forward_probabilities() {
        let mut net = setup();
        let probability_volume = net.forward(&Vol::from([0.2, -0.3].as_ref()), false);

        assert_eq!(probability_volume.w.len(), 3); // 3 classes output
        for w in &probability_volume.w {
            assert!(*w > 0.0);
            assert!(*w < 1.0);
        }
        let w = &probability_volume.w;
        assert_eq!(w[0] + w[1] + w[2], 1.0);
    }

    #[test]
    fn increse_prob_for_ground_truth_class_when_trained() {
        let mut net = setup();
        let mut trainer = Trainer::builder(&mut net)
            .learning_rate(0.0001)
            .momentum(0.0)
            .batch_size(1)
            .l2_decay(0.0)
            .build();

        // lets test 100 random point and label settings
        // note that this should work since l2 and l1 regularization are off
        // an issue is that if step size is too high, this could technically fail...
        // for(var k=0;k<100;k++) {
        for _k in 0..100 {
            // var x = new convnetjs.Vol([Math.random() * 2 - 1, Math.random() * 2 - 1]);
            let mut x =
                Vol::from([random::<f32>() * 2.0 - 1.0, random::<f32>() * 2.0 - 1.0].as_ref());
            let pv = trainer.net().forward(&x, false);

            // var gti = Math.floor(Math.random() * 3);
            let gti = (random::<f32>() * 3.0).floor() as usize;

            // trainer.train(x, gti);
            trainer.train(&mut x, gti);

            // var pv2 = net.forward(x);
            let pv2 = trainer.net().forward(&x, false);

            // expect(pv2.w[gti]).toBeGreaterThan(pv.w[gti]);
            assert!(pv2.w[gti] > pv.w[gti]);
        }
    }

    #[test]
    fn compute_correct_gradient_at_data() {
        // here we only test the gradient at data, but if this is
        // right then that's comforting, because it is a function
        // of all gradients above, for all layers.

        let mut net = setup();

        // var x = new convnetjs.Vol([Math.random() * 2 - 1, Math.random() * 2 - 1]);
        let mut x = Vol::from([random::<f32>() * 2.0 - 1.0, random::<f32>() * 2.0 - 1.0].as_ref());
        // var gti = Math.floor(Math.random() * 3); // ground truth index
        let gti = (random::<f32>() * 3.0).floor() as usize;

        let mut trainer = Trainer::builder(&mut net)
            .learning_rate(0.0001)
            .momentum(0.0)
            .batch_size(1)
            .l2_decay(0.0)
            .build();

        // trainer.train(x, gti); // computes gradients at all layers, and at x

        trainer.train(&mut x, gti);

        // var delta = 0.000001;
        let delta = 0.0001;

        // for(var i=0;i<x.w.length;i++) {
        for i in 0..x.w.len() {
            // var grad_analytic = x.dw[i];
            let grad_analytic = x.dw[i];

            // var xold = x.w[i];
            let xold = x.w[i];

            // x.w[i] += delta;
            x.w[i] += delta;

            // var c0 = net.getCostLoss(x, gti);
            let c0 = trainer.net().get_cost_loss(&x, gti);

            // x.w[i] -= 2*delta;
            x.w[i] -= 2.0 * delta;

            // var c1 = net.getCostLoss(x, gti);
            let c1 = trainer.net().get_cost_loss(&x, gti);

            // x.w[i] = xold; // reset
            x.w[i] = xold;

            // var grad_numeric = (c0 - c1)/(2 * delta);
            let grad_numeric = (c0 - c1) / (2.0 * delta);

            // var rel_error = Math.abs(grad_analytic - grad_numeric)/Math.abs(grad_analytic + grad_numeric);
            let rel_error =
                (grad_analytic - grad_numeric).abs() / (grad_analytic + grad_numeric).abs();

            println!(
                "{i}: numeric: {grad_numeric}, analytic: {grad_analytic}, rel error: {rel_error}"
            );

            //   expect(rel_error).toBeLessThan(1e-2);
            assert!(rel_error < 1e-2);
        }
    }
}
