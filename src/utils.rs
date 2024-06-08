use rand_distr::{Distribution, Normal, Uniform};

pub fn gauss_random(mean: f32, std: f32) -> f32 {
    let normal = Normal::new(mean, std).expect("std must be finite");

    normal.sample(&mut rand::thread_rng())
}

pub fn randf(a: f32, b: f32) -> f32 {
    let uniform = Uniform::new(a, b);
    uniform.sample(&mut rand::thread_rng())
}

pub fn randi(a: i64, b: i64) -> i64 {
    let uniform = Uniform::new(a, b);
    uniform.sample(&mut rand::thread_rng())
}

pub fn randn(mu: f32, std: f32) -> f32 {
    mu + gauss_random(0.0, 1.0) * std
}

pub fn zeros(n: usize) -> Vec<f32> {
    vec![0.0; n]
}

pub struct MinMax {
    min_value: f32,
    min_index: usize,
    max_value: f32,
    max_index: usize,
    diff_value: f32,
}

// return max and min of a given non-empty array.
pub fn maxmin(values: &[f32]) -> Option<MinMax> {
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
