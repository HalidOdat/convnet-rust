use crate::vol::Vol;

use super::NetLayer;

pub struct PoolLayer {
    // required
    sx: usize,
    in_depth: usize,
    in_sx: usize,
    in_sy: usize,

    // optional
    sy: usize,
    stride: usize,
    padding: usize,

    // computed
    out_depth: usize,
    out_sx: usize,
    out_sy: usize,

    // store switches for x,y coordinates for where the max comes from, for each output neuron
    switch_x: Vec<usize>,
    switch_y: Vec<usize>,
}

impl PoolLayer {
    pub fn builder(sx: usize, in_depth: usize, in_sx: usize, in_sy: usize) -> PoolLayerBuilder {
        PoolLayerBuilder::new(sx, in_depth, in_sx, in_sy)
    }
}

pub struct PoolLayerBuilder {
    // required
    sx: usize,
    in_depth: usize,
    in_sx: usize,
    in_sy: usize,

    // optional
    sy: usize,
    stride: usize,
    padding: usize,
}

impl PoolLayerBuilder {
    fn new(
        // required
        sx: usize,
        in_depth: usize,
        in_sx: usize,
        in_sy: usize,
    ) -> Self {
        let sy = sx;
        let stride = 2;
        let padding = 0;
        Self {
            sx,
            in_depth,
            in_sx,
            in_sy,

            sy,
            stride,
            padding,
        }
    }

    pub fn sy(mut self, value: usize) -> Self {
        self.sy = value;
        self
    }

    pub fn stride(mut self, value: usize) -> Self {
        self.stride = value;
        self
    }

    pub fn padding(mut self, value: usize) -> Self {
        self.padding = value;
        self
    }

    pub fn build(self) -> PoolLayer {
        let out_depth = self.in_depth;
        let out_sx = (self.in_sx + self.padding * 2 - self.sx) / self.stride + 1;
        let out_sy = (self.in_sy + self.padding * 2 - self.sy) / self.stride + 1;

        PoolLayer {
            sx: self.sx,
            in_depth: self.in_depth,
            in_sx: self.in_sx,
            in_sy: self.in_sy,
            sy: self.sy,
            stride: self.stride,
            padding: self.padding,

            out_depth,
            out_sx,
            out_sy,

            // store switches for x,y coordinates for where the max comes from, for each output neuron
            switch_x: vec![0; out_sx * out_sy * out_depth],
            switch_y: vec![0; out_sx * out_sy * out_depth],
        }
    }
}

impl NetLayer for PoolLayer {
    fn forward(
        &mut self,
        in_act: &crate::vol::Vol,
        out_act: &mut crate::vol::Vol,
        _is_training: bool,
    ) {
        let v = in_act;

        // TOOD: Reuse out_act
        *out_act = Vol::zeros(self.out_sx, self.out_sy, self.out_depth);

        // var n=0; // a counter for switches
        let mut n = 0;
        // for(var d=0;d<this.out_depth;d++) {
        for d in 0..self.out_depth {
            // var x = -this.pad;
            let mut x = -(self.padding as isize);
            // var y = -this.pad;
            let mut y = -(self.padding as isize);

            // for(var ax=0; ax<this.out_sx; x+=this.stride,ax++) {
            for ax in 0..self.out_sx {
                // y = -this.pad;
                y = -(self.padding as isize);

                // for(var ay=0; ay<this.out_sy; y+=this.stride,ay++) {
                for ay in 0..self.out_sy {
                    // // convolve centered at this particular location
                    // var a = -99999; // hopefully small enough ;\
                    let mut a = -99999.0;

                    // var winx=-1,winy=-1;
                    let mut winx = -1;
                    let mut winy = -1;

                    // for(var fx=0;fx<this.sx;fx++) {
                    for fx in 0..self.sx as isize {
                        // for(var fy=0;fy<this.sy;fy++) {
                        for fy in 0..self.sy as isize {
                            // var oy = y+fy;
                            let oy = y + fy;
                            // var ox = x+fx;
                            let ox = x + fx;

                            // if(oy>=0 && oy<V.sy && ox>=0 && ox<V.sx) {
                            if oy >= 0 && oy < v.sy() as isize && ox >= 0 && ox < v.sx() as isize {
                                // var v = V.get(ox, oy, d);
                                let v = v.get(ox as usize, oy as usize, d);
                                // perform max pooling and store pointers to where
                                // the max came from. This will speed up backprop
                                // and can help make nice visualizations in future
                                if v > a {
                                    a = v;
                                    winx = ox;
                                    winy = oy;
                                }
                            }
                        }
                    }

                    debug_assert!(winx >= 0);
                    debug_assert!(winy >= 0);

                    // this.switchx[n] = winx;
                    self.switch_x[n] = winx as usize;
                    // this.switchy[n] = winy;
                    self.switch_y[n] = winy as usize;
                    // n++;
                    n += 1;
                    // A.set(ax, ay, d, a);
                    out_act.set(ax, ay, d, a);

                    y += self.stride as isize;
                }

                x += self.stride as isize;
            }
        }
    }

    fn backward(&mut self, in_act: &mut Vol, out_act: &Vol) {
        // // pooling layers have no parameters, so simply compute
        // // gradient wrt data here
        // var V = this.in_act;
        let v = in_act;

        debug_assert_eq!(v.w.len(), v.dw.len());

        // V.dw = global.zeros(V.w.length); // zero out gradient wrt data
        v.dw.fill(0.0);

        // var A = this.out_act; // computed in forward pass

        // var n = 0;
        let mut n = 0;
        // for(var d=0;d<this.out_depth;d++) {
        for d in 0..self.out_depth {
            // var x = -this.pad;
            let mut x = -(self.padding as isize);
            // var y = -this.pad;
            let mut y = -(self.padding as isize);

            // for(var ax=0; ax<this.out_sx; x+=this.stride,ax++) {
            for ax in 0..self.out_sx {
                // y = -this.pad;
                y = -(self.padding as isize);

                // for(var ay=0; ay<this.out_sy; y+=this.stride,ay++) {
                for ay in 0..self.out_sy {
                    // var chain_grad = this.out_act.get_grad(ax,ay,d);
                    let chain_grad = out_act.get_gradiant(ax, ay, d);

                    // V.add_grad(this.switchx[n], this.switchy[n], d, chain_grad);
                    v.add_gradiant(self.switch_x[n], self.switch_y[n], d, chain_grad);

                    // n++;
                    n += 1;

                    y += self.stride as isize;
                }

                x += self.stride as isize;
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
