use super::NetLayer;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct InputLayer {
    // required
    out_depth: usize,

    // optional
    out_sx: usize,
    out_sy: usize,
}

impl InputLayer {
    pub fn new(depth: usize) -> Self {
        Self {
            out_depth: depth,
            out_sx: 1,
            out_sy: 1,
        }
    }

    pub fn with_dimensions(depth: usize, width: usize, height: usize) -> Self {
        Self {
            out_depth: depth,
            out_sx: width,
            out_sy: height,
        }
    }
}

#[typetag::serde]
impl NetLayer for InputLayer {
    fn forward(
        &mut self,
        in_act: &crate::vol::Vol,
        out_act: &mut crate::vol::Vol,
        _is_training: bool,
    ) {
        debug_assert_eq!(in_act.w.len(), in_act.dw.len());
        debug_assert_eq!(out_act.w.len(), out_act.dw.len());
        debug_assert_eq!(in_act.w.len(), out_act.w.len());
        out_act.w.copy_from_slice(&in_act.w);
        out_act.dw.copy_from_slice(&in_act.dw);
    }
    fn backward(&mut self, in_act: &mut crate::vol::Vol, out_act: &crate::vol::Vol) {
        debug_assert_eq!(out_act.w.len(), out_act.dw.len());
        debug_assert_eq!(in_act.w.len(), in_act.dw.len());
        debug_assert_eq!(out_act.w.len(), in_act.w.len());
        in_act.w.copy_from_slice(&out_act.w);
        in_act.dw.copy_from_slice(&out_act.dw);
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
