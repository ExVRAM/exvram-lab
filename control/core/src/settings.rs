//! Local user settings. This file is not a benchmark result.

use serde::{Deserialize, Serialize};

use crate::estimate::{KvFormat, MEASURED_CONTEXT};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    pub schema: u32,
    pub model_path: Option<String>,
    pub server_path: Option<String>,
    pub models_dir: Option<String>,
    pub context_tokens: u32,
    pub kv: String,
    pub mode: String,
    pub profile: String,
    pub protect_pc: bool,
    #[serde(default = "default_batch")]
    pub batch: u32,
    #[serde(default = "default_ubatch")]
    pub ubatch: u32,
    #[serde(default = "default_flash")]
    pub flash_attn: bool,
    #[serde(default)]
    pub threads_override: Option<u32>,
}

fn default_batch() -> u32 {
    32
}
fn default_ubatch() -> u32 {
    1
}
fn default_flash() -> bool {
    true
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            schema: 1,
            model_path: None,
            server_path: None,
            models_dir: None,
            context_tokens: MEASURED_CONTEXT,
            kv: KvFormat::Q4.as_arg().into(),
            mode: "balanced".into(),
            profile: "chat".into(),
            protect_pc: true,
            batch: default_batch(),
            ubatch: default_ubatch(),
            flash_attn: true,
            threads_override: None,
        }
    }
}

impl Settings {
    pub fn kv_format(&self) -> Result<KvFormat, String> {
        KvFormat::parse(&self.kv).ok_or_else(|| format!("unsupported KV format {}", self.kv))
    }

    pub fn normalized(mut self) -> Result<Self, String> {
        if self.schema == 0 {
            self.schema = 1;
        }
        if self.schema != 1 {
            return Err(format!("unsupported settings schema {}", self.schema));
        }
        if self.context_tokens < 256 {
            return Err("context must be at least 256 tokens".into());
        }
        self.kv_format()?;
        if !matches!(self.mode.as_str(), "balanced" | "performance" | "quality") {
            return Err(format!("unsupported mode {}", self.mode));
        }
        if !matches!(
            self.profile.as_str(),
            "chat" | "documents" | "coding" | "research" | "custom"
        ) {
            return Err(format!("unsupported profile {}", self.profile));
        }
        if !(1..=2048).contains(&self.batch) || !(1..=512).contains(&self.ubatch) {
            return Err("batch/ubatch вне поддерживаемого диапазона".into());
        }
        if self.ubatch > self.batch {
            return Err("ubatch не может быть больше batch".into());
        }
        if self.threads_override.is_some_and(|threads| !(1..=32).contains(&threads)) {
            return Err("threads вне диапазона 1..32".into());
        }
        Ok(self)
    }
}

pub fn apply_profile(profile: &str, total_vram_mib: u32, protect_pc: bool) -> Result<u32, String> {
    let preferred = match profile {
        "chat" | "coding" => 8192,
        "documents" => 16384,
        "research" => MEASURED_CONTEXT,
        "custom" => return Err("custom keeps the current context".into()),
        _ => return Err(format!("unsupported profile {profile}")),
    };
    let kv = KvFormat::Q4;
    if crate::estimate::preset_allowed(preferred, total_vram_mib, kv, protect_pc) {
        Ok(preferred)
    } else if crate::estimate::preset_allowed(8192, total_vram_mib, kv, protect_pc) {
        Ok(8192)
    } else if crate::estimate::preset_allowed(4096, total_vram_mib, kv, protect_pc) {
        Ok(4096)
    } else {
        Err("даже 4K не помещается в эту оценку VRAM".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_reject_unknown_kv() {
        let settings = Settings::default();
        let text = serde_json::to_string(&settings).unwrap();
        let loaded: Settings = serde_json::from_str(&text).unwrap();
        assert_eq!(loaded.normalized().unwrap().protect_pc, true);
        let mut bad = settings;
        bad.kv = "q2_x".into();
        assert!(bad.normalized().is_err());
    }

    #[test]
    fn documents_profile_falls_back_when_16k_does_not_fit() {
        let context = apply_profile("documents", 8151, true).unwrap();
        assert!(context <= 8448);
        assert_ne!(context, 16384);
    }

    #[test]
    fn research_profile_keeps_the_measured_window_on_8gb() {
        assert_eq!(apply_profile("research", 8151, true).unwrap(), 8448);
    }
}
