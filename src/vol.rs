use image::DynamicImage;

use crate::utils::randn;

// Vol is the basic building block of all data in a net.
// it is essentially just a 3D volume of numbers, with a
// width (sx), height (sy), and depth (depth).
// it is used to hold data for all filters, all volumes,
// all weights, and also stores all gradients w.r.t.
// the data. c is optionally a value to initialize the volume
// with. If c is missing, fills the Vol with random numbers.
#[derive(Debug, Clone)]
pub struct Vol {
    sx: usize,
    sy: usize,
    depth: usize,

    pub w: Vec<f32>,
    pub dw: Vec<f32>,
}

impl From<&[f32]> for Vol {
    fn from(value: &[f32]) -> Self {
        let sx = 1;
        let sy = 1;
        let depth = value.len();

        let n = sx * sy * depth;
        Self {
            sx,
            sy,
            depth,
            w: value.to_vec(),
            dw: vec![0.0; n],
        }
    }
}

impl Vol {
    pub fn new(sx: usize, sy: usize, depth: usize) -> Self {
        let n = sx * sy * depth;

        // weight normalization is done to equalize the output
        // variance of every neuron, otherwise neurons with a lot
        // of incoming connections have outputs of larger variance
        let scale = (1.0 / (n as f32)).sqrt();
        let mut w = Vec::with_capacity(n);
        for _ in 0..n {
            let value = randn(0.0, scale);
            w.push(value);
        }

        let dw = vec![0.0; n];
        Self {
            sx,
            sy,
            depth,
            w,
            dw,
        }
    }

    pub fn with_constant(sx: usize, sy: usize, depth: usize, constant: f32) -> Self {
        let n = sx * sy * depth;

        let w = vec![constant; n];
        let dw = vec![constant; n];
        Self {
            sx,
            sy,
            depth,
            w,
            dw,
        }
    }

    pub fn zeros(sx: usize, sy: usize, depth: usize) -> Self {
        Self::with_constant(sx, sy, depth, 0.0)
    }

    fn get_index(&self, x: usize, y: usize, d: usize) -> usize {
        ((self.sx * y) + x) * self.depth + d
    }

    pub fn get(&self, x: usize, y: usize, d: usize) -> f32 {
        let index = self.get_index(x, y, d);
        self.w[index]
    }

    pub fn set(&mut self, x: usize, y: usize, d: usize, value: f32) {
        let index = self.get_index(x, y, d);
        self.w[index] = value
    }

    pub fn add(&mut self, x: usize, y: usize, d: usize, value: f32) {
        let index = self.get_index(x, y, d);
        self.w[index] += value
    }

    pub fn get_gradiant(&self, x: usize, y: usize, d: usize) -> f32 {
        let index = self.get_index(x, y, d);
        self.dw[index]
    }

    pub fn set_gradiant(&mut self, x: usize, y: usize, d: usize, value: f32) {
        let index = self.get_index(x, y, d);
        self.dw[index] = value
    }

    pub fn add_gradiant(&mut self, x: usize, y: usize, d: usize, value: f32) {
        let index = self.get_index(x, y, d);
        self.dw[index] += value
    }

    pub fn clone_and_zero(&self) -> Self {
        Self::with_constant(self.sx, self.sy, self.depth, 0.0)
    }

    // addFrom
    // addFromScaled
    // setConst

    // pub fn augment() {}
    pub fn from_rgba_image(img: &DynamicImage) -> Self {
        let width = img.width() as usize;
        let height = img.width() as usize;

        let bytes = img.as_bytes();

        assert_eq!(
            4 * width * height,
            bytes.len(),
            "image should have 4 components rgba"
        );

        let mut vol = Self::new(width, height, 4);
        for (w, pixel) in vol.w.iter_mut().zip(bytes.iter().copied()) {
            // normalize image pixels to [-0.5, 0.5]
            *w = (pixel as f32) / 255.0 - 0.5;
        }

        vol
    }

    pub fn sx(&self) -> usize {
        self.sx
    }
    pub fn sy(&self) -> usize {
        self.sy
    }
    pub fn depth(&self) -> usize {
        self.depth
    }
}
