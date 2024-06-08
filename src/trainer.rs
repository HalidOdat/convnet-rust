use crate::{net::Net, utils::zeros, vol::Vol, Float};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Method {
    Sgd,
    Adadlta { ro: Float, eps: Float },
    Adam { eps: Float, beta1: Float, beta2: Float },
}

pub struct Trainer<'net> {
    net: &'net mut Net,

    learning_rate: Float,
    l1_decay: Float,
    l2_decay: Float,
    batch_size: usize,
    method: Method,
    momentum: Float,

    /// iteration counter
    k: usize,

    // last iteration gradients (used for momentum calculations)
    gsum: Vec<Vec<Float>>,

    // used in adam or adadelta
    xsum: Vec<Vec<Float>>,

    regression: bool,
}

impl<'net> Trainer<'net> {
    pub fn builder(net: &mut Net) -> TrainerBuilder<'_> {
        TrainerBuilder::new(net)
    }

    pub fn net(&mut self) -> &mut Net {
        &mut *self.net
    }

    pub fn train(&mut self, x: &mut Vol, y: usize) {
        self.net.forward(x, true);
        let cost_loss = self.net.backward(y, x);

        let mut l1_decay_loss = 0.0;
        let mut l2_decay_loss = 0.0;

        assert!(!self.regression, "Regression not supported for now");

        self.k += 1;
        if self.k % self.batch_size == 0 {
            // var pglist = this.net.getParamsAndGrads();
            let mut pg_list = self.net.params_and_grads();

            // // initialize lists for accumulators. Will only be done once on first iteration
            // if(this.gsum.length === 0 && (this.method !== 'sgd' || this.momentum > 0.0)) {
            if self.gsum.is_empty() && (self.method == Method::Sgd || self.momentum > 0.0) {
                // only vanilla sgd doesnt need either lists
                // momentum needs gsum
                // adagrad needs gsum
                // adam and adadelta needs gsum and xsum
                // for(var i=0;i<pglist.length;i++) {
                for pg in &pg_list {
                    // this.gsum.push(global.zeros(pglist[i].params.length));
                    self.gsum.push(zeros(pg.params.len()));

                    // if(this.method === 'adam' || this.method === 'adadelta') {
                    if matches!(self.method, Method::Adam { .. } | Method::Adadlta { .. }) {
                        // this.xsum.push(global.zeros(pglist[i].params.length));
                        self.xsum.push(zeros(pg.params.len()));
                    } else {
                        // this.xsum.push([]); // conserve memory
                        self.xsum.push(Vec::new());
                    }
                }
            }

            // // perform an update for all sets of weights
            // for(var i=0;i<pglist.length;i++) {
            // var pg = pglist[i]; // param, gradient, other options in future (custom learning rate etc)
            for (i, pg) in pg_list.iter_mut().enumerate() {
                // var p = pg.params;
                // let p = pg.params;
                // var g = pg.grads;
                // let g = pg.grads;

                // // learning rate for some parameters.
                // var l2_decay_mul = typeof pg.l2_decay_mul !== 'undefined' ? pg.l2_decay_mul : 1.0;
                let l2_decay_mul = pg.l2_decay_mul;
                // var l1_decay_mul = typeof pg.l1_decay_mul !== 'undefined' ? pg.l1_decay_mul : 1.0;
                let l1_decay_mul = pg.l1_decay_mul;

                // var l2_decay = this.l2_decay * l2_decay_mul;
                let l2_decay = self.l2_decay * l2_decay_mul;
                // var l1_decay = this.l1_decay * l1_decay_mul;
                let l1_decay = self.l1_decay * l1_decay_mul;

                // var plen = p.length;
                let plen = pg.params.len();

                // for(var j=0;j<plen;j++) {
                for j in 0..plen {
                    // l2_decay_loss += l2_decay*p[j]*p[j]/2; // accumulate weight decay loss
                    l2_decay_loss += l2_decay * pg.params[j] * pg.params[j] / 2.0;

                    // l1_decay_loss += l1_decay*Math.abs(p[j]);
                    l1_decay_loss += l1_decay * pg.params[j].abs();

                    // var l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
                    let l1grad = l1_decay * if pg.params[j] > 0.0 { 1.0 } else { -1.0 };
                    // var l2grad = l2_decay * (p[j]);
                    let l2grad = l2_decay * pg.params[j];

                    // var gij = (l2grad + l1grad + g[j]) / this.batch_size; // raw batch gradient
                    let gij = (l2grad + l1grad + pg.grads[j]) / self.batch_size as Float;

                    // var gsumi = this.gsum[i];
                    // var xsumi = this.xsum[i];
                    let gsumi = &mut self.gsum[i];
                    let xsumi = &mut self.xsum[i];

                    match self.method {
                        Method::Adam { eps, beta1, beta2 } => {
                            // adam update
                            gsumi[j] = gsumi[j] * beta1 + (1.0 - beta1) * gij; // update biased first moment estimate
                            xsumi[j] = xsumi[j] * beta2 + (1.0 - beta2) * gij * gij; // update biased second moment estimate
                            let bias_corr1 = gsumi[j] * (1.0 - beta1.powi(self.k as i32)); // correct bias first moment estimate
                            let bias_corr2 = xsumi[j] * (1.0 - beta2.powi(self.k as i32)); // correct bias second moment estimate
                            let dx = -self.learning_rate * bias_corr1 / (bias_corr2.sqrt() + eps);
                            pg.params[j] += dx;
                        }
                        Method::Adadlta { ro, eps } => {
                            gsumi[j] = ro * gsumi[j] + (1.0 - ro) * gij * gij;
                            let dx = -((xsumi[j] + eps) / (gsumi[j] + eps)).sqrt() * gij;
                            xsumi[j] = ro * xsumi[j] + (1.0 - ro) * dx * dx; // yes, xsum lags behind gsum by 1.
                            pg.params[j] += dx;
                        }
                        Method::Sgd => {
                            // vanilla sgd
                            pg.params[j] += -self.learning_rate * gij;
                        }
                    }
                    pg.grads[j] = 0.0;
                }
            }
        }
    }
}

pub struct TrainerBuilder<'net> {
    net: &'net mut Net,

    learning_rate: Float,
    l1_decay: Float,
    l2_decay: Float,
    batch_size: usize,
    method: Method,
    momentum: Float,
}

impl<'net> TrainerBuilder<'net> {
    fn new(net: &'net mut Net) -> Self {
        Self {
            net,

            learning_rate: 0.01,
            l1_decay: 0.0,
            l2_decay: 0.0,
            batch_size: 1,
            method: Method::Sgd,
            momentum: 0.9,
        }
    }

    pub fn learning_rate(mut self, value: Float) -> Self {
        self.learning_rate = value;
        self
    }

    pub fn l1_decay(mut self, value: Float) -> Self {
        self.l1_decay = value;
        self
    }

    pub fn l2_decay(mut self, value: Float) -> Self {
        self.l2_decay = value;
        self
    }

    pub fn batch_size(mut self, value: usize) -> Self {
        self.batch_size = value;
        self
    }

    pub fn method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    pub fn momentum(mut self, value: Float) -> Self {
        self.momentum = value;
        self
    }

    pub fn build(self) -> Trainer<'net> {
        Trainer::<'net> {
            net: self.net,
            learning_rate: self.learning_rate,
            l1_decay: self.l1_decay,
            l2_decay: self.l2_decay,
            batch_size: self.batch_size,
            method: self.method,
            momentum: self.momentum,
            k: 0,
            gsum: vec![],
            xsum: vec![],
            // TODO: check this
            regression: false,
        }
    }
}
