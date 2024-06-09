use super::NetLayer;

// Implements ReLU nonlinearity elementwise
// x -> max(0, x)
// the output is in [0, inf)
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ReluLayer {
    pub(crate) out_sx: usize,
    pub(crate) out_sy: usize,
    pub(crate) out_depth: usize,
}

impl ReluLayer {
    pub fn new(in_sx: usize, in_sy: usize, in_depth: usize) -> Self {
        Self {
            out_sx: in_sx,
            out_sy: in_sy,
            out_depth: in_depth,
        }
    }
}

#[typetag::serde]
impl NetLayer for ReluLayer {
    fn forward(
        &mut self,
        in_act: &crate::vol::Vol,
        out_act: &mut crate::vol::Vol,
        _is_training: bool,
    ) {
        assert_eq!(
            in_act.w.len(),
            out_act.w.len(),
            "in and out should have the same length"
        );

        // let n = in_act.w.len();
        for (in_w, out_w) in in_act.w.iter().cloned().zip(out_act.w.iter_mut()) {
            if in_w < 0.0 {
                // threshold at 0
                *out_w = 0.0;
            } else {
                *out_w = in_w;
            }
        }
    }

    fn backward(&mut self, in_act: &mut crate::vol::Vol, out_act: &crate::vol::Vol) {
        assert_eq!(
            in_act.w.len(),
            out_act.w.len(),
            "in and out should have the same length"
        );

        // var V = this.in_act; // we need to set dw of this
        // var V2 = this.out_act;
        // var N = V.w.length;
        // V.dw = global.zeros(N); // zero out gradient wrt data

        for i in 0..in_act.w.len() {
            if out_act.w[i] <= 0.0 {
                // threshold
                in_act.dw[i] = 0.0;
            } else {
                in_act.dw[i] = out_act.dw[i];
            }
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
        Vec::new()
    }
}

// Implements Sigmoid nnonlinearity elementwise
// x -> 1/(1+e^(-x))
// so the output is between 0 and 1.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SigmoidLayer {
    out_sx: usize,
    out_sy: usize,
    out_depth: usize,
}

impl SigmoidLayer {
    pub fn new(in_sx: usize, in_sy: usize, in_depth: usize) -> Self {
        Self {
            out_sx: in_sx,
            out_sy: in_sy,
            out_depth: in_depth,
        }
    }
}

#[typetag::serde]
impl NetLayer for SigmoidLayer {
    fn forward(
        &mut self,
        in_act: &crate::vol::Vol,
        out_act: &mut crate::vol::Vol,
        _is_training: bool,
    ) {
        assert_eq!(
            in_act.w.len(),
            out_act.w.len(),
            "in and out should have the same length"
        );

        out_act.dw.fill(0.0);

        for (in_w, out_w) in in_act.w.iter().cloned().zip(out_act.w.iter_mut()) {
            *out_w = 1.0 / (1.0 + (-in_w).exp());
        }
    }

    fn backward(&mut self, in_act: &mut crate::vol::Vol, out_act: &crate::vol::Vol) {
        assert_eq!(in_act.w.len(), in_act.dw.len());
        assert_eq!(
            in_act.w.len(),
            out_act.w.len(),
            "in and out should have the same length"
        );

        // var V = this.in_act; // we need to set dw of this
        // var V2 = this.out_act;
        // var N = V.w.length;
        // V.dw = global.zeros(N); // zero out gradient wrt data

        for i in 0..in_act.w.len() {
            let v2wi = out_act.w[i];
            in_act.dw[i] = v2wi * (1.0 - v2wi) * out_act.dw[i];
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
        Vec::new()
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct TanhLayer {
    out_sx: usize,
    out_sy: usize,
    out_depth: usize,
}

impl TanhLayer {
    pub fn new(in_sx: usize, in_sy: usize, in_depth: usize) -> Self {
        Self {
            out_sx: in_sx,
            out_sy: in_sy,
            out_depth: in_depth,
        }
    }
}

#[typetag::serde]
impl NetLayer for TanhLayer {
    fn forward(
        &mut self,
        in_act: &crate::vol::Vol,
        out_act: &mut crate::vol::Vol,
        _is_training: bool,
    ) {
        assert_eq!(
            in_act.w.len(),
            out_act.w.len(),
            "in and out should have the same length"
        );

        for (in_w, out_w) in in_act.w.iter().cloned().zip(out_act.w.iter_mut()) {
            *out_w = in_w.tanh();
        }
    }

    fn backward(&mut self, in_act: &mut crate::vol::Vol, out_act: &crate::vol::Vol) {
        assert_eq!(in_act.w.len(), in_act.dw.len());
        assert_eq!(
            in_act.w.len(),
            out_act.w.len(),
            "in and out should have the same length"
        );

        // var V = this.in_act; // we need to set dw of this
        // var V2 = this.out_act;
        // var N = V.w.length;
        // V.dw = global.zeros(N); // zero out gradient wrt data

        for i in 0..in_act.w.len() {
            let v2wi = out_act.w[i];
            in_act.dw[i] = (1.0 - v2wi * v2wi) * out_act.dw[i];
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
        Vec::new()
    }
}
