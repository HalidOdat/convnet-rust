use crate::{net::Net, utils::zeros, vol::Vol, Float, Sample};
use rand::seq::SliceRandom;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Method {
    Sgd,
    Adadlta {
        ro: Float,
        eps: Float,
    },
    Adam {
        eps: Float,
        beta1: Float,
        beta2: Float,
    },
}

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct TrainStats {
    pub epoch: u32,
    pub epoch_count: u32,
    pub batch: u32,
    pub batch_count: u32,
    pub samples_per_batch: u32,

    pub l1_decay_loss: Float,
    pub l2_decay_loss: Float,
    pub cost_loss: Float,
}

impl TrainStats {
    pub fn loss(&self) -> Float {
        self.cost_loss + self.l1_decay_loss + self.l2_decay_loss
    }
}

pub struct Trainer {
    learning_rate: Float,
    l1_decay: Float,
    l2_decay: Float,
    batch_size: u32,
    method: Method,
    momentum: Float,

    // last iteration gradients (used for momentum calculations)
    gsum: Vec<Vec<Float>>,

    // used in adam or adadelta
    xsum: Vec<Vec<Float>>,

    regression: bool,

    /// iteration counter
    k: u32,

    epoch_count: u32,
    epoch_counter: u32,

    pub samples: Vec<Sample>,

    samples_indices: Vec<u32>,
    validation_indices: Vec<u32>,

    pub classification_loss: Window,
    pub l2_weight_decay_loss: Window,
    pub training_accuracy: Window,
    pub validation_accuracy: Window,
}

impl Trainer {
    pub fn builder() -> TrainerBuilder {
        TrainerBuilder::new()
    }

    pub fn train_sample(&mut self, net: &mut Net, x: &mut Vol, y: usize) -> TrainStats {
        net.forward(x, true);
        let cost_loss = net.backward(y, x);

        let mut l1_decay_loss = 0.0;
        let mut l2_decay_loss = 0.0;

        assert!(!self.regression, "Regression not supported for now");

        self.k += 1;
        if self.k % self.batch_size == 0 {
            // var pglist = this.net.getParamsAndGrads();
            let mut pg_list = net.params_and_grads();

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
                // let p = pg.params;
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

        TrainStats {
            epoch: self.epoch_count,
            epoch_count: self.epoch_count,
            batch: self.batch_size,
            batch_count: self.batch_size,
            samples_per_batch: self.samples_indices.len() as u32 / self.batch_size,

            l1_decay_loss,
            l2_decay_loss,
            cost_loss,
        }
    }

    fn randomize_samples(&mut self) {
        self.samples_indices.shuffle(&mut rand::thread_rng());
    }

    pub fn train(&mut self, net: &mut Net, stats: &mut TrainStats) -> bool {
        if self.k >= self.samples_indices.len() as u32 {
            self.k = 0;
            self.epoch_counter += 1;
        }
        if self.epoch_counter >= self.epoch_count {
            *stats = TrainStats {
                epoch: self.epoch_count,
                epoch_count: self.epoch_count,
                batch: self.batch_size,
                batch_count: self.batch_size,
                samples_per_batch: self.samples_indices.len() as u32 / self.batch_size,
                l1_decay_loss: stats.l1_decay_loss,
                l2_decay_loss: stats.l2_decay_loss,
                cost_loss: stats.cost_loss,
            };
            self.epoch_counter = 0;
            self.k = 0;
            self.randomize_samples();
            return false;
        }
        let sample_index = self.samples_indices[self.k as usize];
        let sample = &self.samples[sample_index as usize];

        let sample_data = &mut sample.data.clone();
        let sample_label = sample.label;
        let train_stats = self.train_sample(net, sample_data, sample_label as usize);

        *stats = TrainStats {
            epoch: self.epoch_count,
            epoch_count: self.epoch_count,
            batch: self.k / self.samples_indices.len() as u32,
            batch_count: self.batch_size,
            samples_per_batch: self.samples_indices.len() as u32 / self.batch_size,
            l1_decay_loss: stats.l1_decay_loss,
            l2_decay_loss: stats.l2_decay_loss,
            cost_loss: stats.cost_loss,
        };

        let lossx = train_stats.cost_loss;
        let lossw = train_stats.l2_decay_loss;

        // keep track of stats such as the average training error and loss
        let yhat = net.get_prediction() as u32;
        let train_acc = if yhat == sample_label { 1.0 } else { 0.0 };

        self.classification_loss.add(lossx);
        self.l2_weight_decay_loss.add(lossw);
        self.training_accuracy.add(train_acc);

        if let Some(test_sample_index) = self
            .validation_indices
            .choose(&mut rand::thread_rng())
            .copied()
        {
            let test_sample = &self.samples[test_sample_index as usize];

            net.forward(&test_sample.data, true);
            let yhat_test = net.get_prediction();
            let test_train_acc = if yhat_test == test_sample.label as usize {
                1.0
            } else {
                0.0
            };
            self.validation_accuracy.add(test_train_acc);
        }

        true
    }

    pub fn train_without_validation(&mut self, net: &mut Net, stats: &mut TrainStats) -> bool {
        if self.k >= self.samples_indices.len() as u32 {
            self.k = 0;
            self.epoch_counter += 1;
        }
        if self.epoch_counter >= self.epoch_count {
            *stats = TrainStats {
                epoch: self.epoch_count,
                epoch_count: self.epoch_count,
                batch: self.batch_size,
                batch_count: self.batch_size,
                samples_per_batch: self.samples_indices.len() as u32 / self.batch_size,
                l1_decay_loss: stats.l1_decay_loss,
                l2_decay_loss: stats.l2_decay_loss,
                cost_loss: stats.cost_loss,
            };
            self.epoch_counter = 0;
            self.k = 0;
            self.randomize_samples();
            return false;
        }
        let Some(sample) = self.samples.choose(&mut rand::thread_rng()) else {
            return false;
        };

        let sample_data = &mut sample.data.clone();
        let sample_label = sample.label;
        let train_stats = self.train_sample(net, sample_data, sample_label as usize);

        *stats = TrainStats {
            epoch: self.epoch_count,
            epoch_count: self.epoch_count,
            batch: self.k / self.samples_indices.len() as u32,
            batch_count: self.batch_size,
            samples_per_batch: self.samples_indices.len() as u32 / self.batch_size,
            l1_decay_loss: stats.l1_decay_loss,
            l2_decay_loss: stats.l2_decay_loss,
            cost_loss: stats.cost_loss,
        };

        let lossx = train_stats.cost_loss;
        let lossw = train_stats.l2_decay_loss;

        // keep track of stats such as the average training error and loss
        let yhat = net.get_prediction() as u32;
        let train_acc = if yhat == sample_label { 1.0 } else { 0.0 };

        self.classification_loss.add(lossx);
        self.l2_weight_decay_loss.add(lossw);
        self.training_accuracy.add(train_acc);

        true
    }
}

pub struct TrainerBuilder {
    learning_rate: Float,
    l1_decay: Float,
    l2_decay: Float,
    batch_size: u32,
    method: Method,
    momentum: Float,

    samples: Vec<Sample>,
    epoch_count: u32,

    validation: f32,
}

impl TrainerBuilder {
    fn new() -> Self {
        Self {
            learning_rate: 0.01,
            l1_decay: 0.0,
            l2_decay: 0.0,
            batch_size: 1,
            method: Method::Sgd,
            momentum: 0.9,
            epoch_count: 1,
            samples: Vec::default(),
            validation: 0.2,
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

    pub fn batch_size(mut self, value: u32) -> Self {
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

    pub fn epoch(mut self, value: u32) -> Self {
        self.epoch_count = value;
        self
    }

    pub fn validation(mut self, value: f32) -> Self {
        self.validation = value;
        self
    }

    pub fn samples(mut self, samples: Vec<Sample>) -> Self {
        self.samples = samples;
        self
    }

    pub fn build(self) -> Trainer {
        let mut indices: Vec<u32> = (0..self.samples.len() as u32).collect();
        indices.shuffle(&mut rand::thread_rng());

        let n = indices.len();
        let middle = (n as f32 * self.validation).floor() as usize;

        Trainer {
            learning_rate: self.learning_rate,
            l1_decay: self.l1_decay,
            l2_decay: self.l2_decay,
            batch_size: self.batch_size,
            method: self.method,
            momentum: self.momentum,
            gsum: vec![],
            xsum: vec![],
            // TODO: check this
            regression: false,
            k: 0,

            epoch_count: self.epoch_count,
            epoch_counter: 0,
            samples: self.samples,
            samples_indices: indices.split_off(middle),
            validation_indices: indices,
            classification_loss: Window::default(),
            l2_weight_decay_loss: Window::default(),
            training_accuracy: Window::default(),
            validation_accuracy: Window::default(),
        }
    }
    pub fn build_without_validation(self) -> Trainer {
        Trainer {
            learning_rate: self.learning_rate,
            l1_decay: self.l1_decay,
            l2_decay: self.l2_decay,
            batch_size: self.batch_size,
            method: self.method,
            momentum: self.momentum,
            gsum: vec![],
            xsum: vec![],
            // TODO: check this
            regression: false,
            k: 0,

            epoch_count: self.epoch_count,
            epoch_counter: 0,
            samples: self.samples,
            samples_indices: Vec::new(),
            validation_indices: Vec::new(),
            classification_loss: Window::default(),
            l2_weight_decay_loss: Window::default(),
            training_accuracy: Window::default(),
            validation_accuracy: Window::default(),
        }
    }
}

pub struct Window {
    values: Vec<Float>,
    len: usize,
    min_len: usize,
    sum: Float,
}

impl Default for Window {
    fn default() -> Self {
        Self::new(100)
    }
}

impl Window {
    pub fn new(len: usize) -> Self {
        Self::with_min_len(len, 20)
    }

    pub fn with_min_len(len: usize, min_len: usize) -> Self {
        assert!(len >= min_len);
        Self {
            values: Vec::new(),
            len,
            min_len,
            sum: 0.0,
        }
    }

    pub fn add(&mut self, x: Float) {
        self.values.push(x);
        self.sum += x;
        if self.values.len() > self.len {
            let xold = self.values.remove(0);
            self.sum -= xold;
        }
    }

    pub fn average(&self) -> Float {
        if self.values.len() < self.min_len {
            return -1.0;
        }

        self.sum / self.values.len() as Float
    }

    pub fn reset(&mut self) {
        self.values.clear();
        self.sum = 0.0;
    }
}
