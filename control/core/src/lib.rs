//! Decisions for ExVRAM Control. The desktop shell is a separate crate.

pub mod estimate;
pub mod gpu;
pub mod launch;
pub mod recipe;
pub mod recommend;
pub mod settings;
pub mod slots;
pub mod storage;

pub use estimate::{estimate, launch_gate, KvFormat, MemoryEstimate, CONTEXT_PRESETS};
pub use gpu::{parse_nvidia_smi_line, GpuSample};
pub use launch::{server_arguments, thread_count, LaunchRequest};
pub use recipe::parse_recipe_yaml;
pub use recommend::recommend;
pub use settings::{apply_profile, Settings};
