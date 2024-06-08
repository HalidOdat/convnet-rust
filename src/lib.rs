pub mod layers;
pub mod net;
mod trainer;
pub mod utils;
pub mod vol;

pub use trainer::*;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
