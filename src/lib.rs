mod batch;
pub mod layers;
pub mod net;
mod trainer;
pub mod utils;
pub mod vol;

pub use batch::*;
pub use trainer::*;

pub type Float = f32;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
