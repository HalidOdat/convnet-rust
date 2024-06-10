mod layers;
mod net;
mod trainer;
mod utils;
mod vol;

pub use layers::*;
pub use net::*;
pub use trainer::*;
pub use utils::*;
pub use vol::*;

pub type Float = f32;

#[cfg(feature = "uniffi")]
uniffi::setup_scaffolding!();
