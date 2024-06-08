use rand_distr::{Distribution, Normal, Uniform};

use crate::Float;

pub fn gauss_random(mean: Float, std: Float) -> Float {
    let normal = Normal::new(mean, std).expect("std must be finite");

    normal.sample(&mut rand::thread_rng())
}

pub fn randf(a: Float, b: Float) -> Float {
    let uniform = Uniform::new(a, b);
    uniform.sample(&mut rand::thread_rng())
}

pub fn randi(a: i64, b: i64) -> i64 {
    let uniform = Uniform::new(a, b);
    uniform.sample(&mut rand::thread_rng())
}

pub fn randn(mu: Float, std: Float) -> Float {
    mu + gauss_random(0.0, 1.0) * std
}

pub fn zeros(n: usize) -> Vec<Float> {
    vec![0.0; n]
}

pub struct MinMax {
    min_value: Float,
    min_index: usize,
    max_value: Float,
    max_index: usize,
    diff_value: Float,
}

// return max and min of a given non-empty array.
pub fn maxmin(values: &[Float]) -> Option<MinMax> {
    if values.is_empty() {
        return None;
    }

    let mut maxv = values[0];
    let mut minv = values[0];
    let mut maxi = 0;
    let mut mini = 0;
    for (i, value) in values.iter().copied().enumerate() {
        if value > maxv {
            maxv = value;
            maxi = i;
        }
        if value < minv {
            minv = value;
            mini = i;
        }
    }
    Some(MinMax {
        min_value: minv,
        min_index: mini,
        max_value: maxv,
        max_index: maxi,
        diff_value: maxv - minv,
    })
}
