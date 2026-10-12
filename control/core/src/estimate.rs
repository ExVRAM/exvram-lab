//! Memory estimate anchored on the published occupied-8k IQ2_XXS recipe.
//!
//! The q4 / 8448 total is the measured peak. Every other point scales KV and
//! is marked modeled. Speed is never invented here.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvFormat {
    Q4,
    Q8,
    F16,
}

impl KvFormat {
    pub fn as_arg(self) -> &'static str {
        match self {
            Self::Q4 => "q4_0",
            Self::Q8 => "q8_0",
            Self::F16 => "f16",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "q4_0" => Some(Self::Q4),
            "q8_0" => Some(Self::Q8),
            "f16" => Some(Self::F16),
            _ => None,
        }
    }
}

pub const MEASURED_CONTEXT: u32 = 8448;
pub const WEIGHTS_MIB: f64 = 6521.13;
pub const Q4_KV_MIB_AT_MEASURED: f64 = 148.5;
pub const Q8_KV_MIB_AT_MEASURED: f64 = 280.5;
pub const MEASURED_PEAK_MIB: f64 = 7767.0;
pub const MEASURED_DECODE_TOK_S: f64 = 30.1014;

const OTHER_MIB: f64 = MEASURED_PEAK_MIB - WEIGHTS_MIB - Q4_KV_MIB_AT_MEASURED;
/// P5 configured-8192 row. Not the occupied-8k series, so totals stay modeled.
const F16_KV_MIB_AT_CONFIGURED_8192: f64 = 512.0;
const Q8_PEAK_MIB: f64 = 7788.0;

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryEstimate {
    pub weights_mib: f64,
    pub kv_mib: f64,
    pub other_mib: f64,
    pub total_mib: f64,
    pub basis: &'static str,
}

pub fn estimate(context_tokens: u32, kv: KvFormat) -> MemoryEstimate {
    let scale = f64::from(context_tokens) / f64::from(MEASURED_CONTEXT);
    let (kv_mib, kv_basis) = match kv {
        KvFormat::Q4 => (Q4_KV_MIB_AT_MEASURED * scale, "measured"),
        KvFormat::Q8 => (Q8_KV_MIB_AT_MEASURED * scale, "measured"),
        KvFormat::F16 => (
            F16_KV_MIB_AT_CONFIGURED_8192 * f64::from(context_tokens) / 8192.0,
            "modeled",
        ),
    };
    if context_tokens == MEASURED_CONTEXT && kv == KvFormat::Q4 {
        return MemoryEstimate {
            weights_mib: WEIGHTS_MIB,
            kv_mib: Q4_KV_MIB_AT_MEASURED,
            other_mib: OTHER_MIB,
            total_mib: MEASURED_PEAK_MIB,
            basis: "measured",
        };
    }
    if context_tokens == MEASURED_CONTEXT && kv == KvFormat::Q8 {
        return MemoryEstimate {
            weights_mib: WEIGHTS_MIB,
            kv_mib: Q8_KV_MIB_AT_MEASURED,
            other_mib: Q8_PEAK_MIB - WEIGHTS_MIB - Q8_KV_MIB_AT_MEASURED,
            total_mib: Q8_PEAK_MIB,
            basis: "measured",
        };
    }
    MemoryEstimate {
        weights_mib: WEIGHTS_MIB,
        kv_mib,
        other_mib: OTHER_MIB,
        total_mib: WEIGHTS_MIB + kv_mib + OTHER_MIB,
        basis: if kv_basis == "measured" { "modeled" } else { "modeled" },
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchGate {
    pub allowed: bool,
    pub reason: String,
}

pub fn launch_gate(
    total_vram_mib: u32,
    free_vram_mib: u32,
    estimate_mib: f64,
    protect_pc: bool,
) -> LaunchGate {
    // The published peak already sits close to an 8 GB card. The reserve is
    // the leftover after that peak, not a second copy of the framebuffer.
    let leftover = if protect_pc { 256.0 } else { 128.0 };
    if estimate_mib > f64::from(total_vram_mib) - leftover {
        return LaunchGate {
            allowed: false,
            reason: format!(
                "Оценка {estimate_mib:.0} МиБ не оставляет запас на карте {total_vram_mib} МиБ. Модель не будет загружена."
            ),
        };
    }
    let used = total_vram_mib.saturating_sub(free_vram_mib);
    let used_limit = if protect_pc { 2048 } else { 4096 };
    if used > used_limit {
        return LaunchGate {
            allowed: false,
            reason: format!(
                "Другие программы уже держат {used} МиБ VRAM. ExVRAM их не закрывает."
            ),
        };
    }
    LaunchGate {
        allowed: true,
        reason: "Запас памяти достаточен для этой оценки.".into(),
    }
}

pub const CONTEXT_PRESETS: &[u32] = &[4096, 8192, MEASURED_CONTEXT, 16384, 32768, 65536];

pub fn preset_allowed(context_tokens: u32, total_vram_mib: u32, kv: KvFormat, protect_pc: bool) -> bool {
    let estimate = estimate(context_tokens, kv);
    launch_gate(total_vram_mib, total_vram_mib, estimate.total_mib, protect_pc).allowed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_anchor_is_the_published_peak() {
        let estimate = estimate(8448, KvFormat::Q4);
        assert_eq!(estimate.basis, "measured");
        assert!((estimate.total_mib - 7767.0).abs() < 0.1);
        assert!((estimate.kv_mib - 148.5).abs() < 0.1);
    }

    #[test]
    fn larger_context_is_modeled_and_costs_more_kv() {
        let small = estimate(8448, KvFormat::Q4);
        let large = estimate(16896, KvFormat::Q4);
        assert_eq!(large.basis, "modeled");
        assert!(large.kv_mib > small.kv_mib * 1.9);
    }

    #[test]
    fn f16_is_not_labeled_measured() {
        assert_eq!(estimate(8448, KvFormat::F16).basis, "modeled");
    }

    #[test]
    fn eight_gigabyte_card_refuses_a_context_that_does_not_fit() {
        let huge = estimate(65536, KvFormat::F16);
        let gate = launch_gate(8151, 8151, huge.total_mib, true);
        assert!(!gate.allowed);
    }

    #[test]
    fn protect_mode_refuses_a_second_heavy_gpu_user() {
        let estimate = estimate(8448, KvFormat::Q4);
        let busy = launch_gate(8151, 8151 - 3000, estimate.total_mib, true);
        assert!(!busy.allowed);
        assert!(busy.reason.contains("не закрывает"));
        let desktop = launch_gate(8151, 8151 - 1200, estimate.total_mib, true);
        assert!(desktop.allowed);
    }
}
