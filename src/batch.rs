use crate::vol::Vol;

use rand::seq::SliceRandom;

pub struct Sample {
    pub data: Vol,
    pub label: usize,
}

pub struct Epoch {
    smaples: Vec<Sample>,
}

pub struct DataSet {
    epoch: Epoch,
}

impl DataSet {
    pub fn new() {}
}
