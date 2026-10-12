//! Host RAM and Windows commit headroom. No process is created or killed here.

const PROTECT_RAM_BYTES: u64 = 1536 * 1024 * 1024;
const PROTECT_COMMIT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostSample {
    pub ram_available_bytes: u64,
    pub commit_free_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostDecision {
    pub allowed: bool,
    pub reason: String,
    pub kills_processes: bool,
    pub changes_system_settings: bool,
}

pub fn host_pressure(sample: HostSample, protect_pc: bool) -> HostDecision {
    if !protect_pc {
        return HostDecision {
            allowed: true,
            reason: "Protect My PC выключен. Проверка RAM и commit пропущена.".into(),
            kills_processes: false,
            changes_system_settings: false,
        };
    }
    if sample.ram_available_bytes < PROTECT_RAM_BYTES {
        return refused(
            "Свободной RAM меньше 1.5 ГиБ. Запуск отменён, чтобы не увести систему в pagefile.",
        );
    }
    if sample
        .commit_free_bytes
        .is_some_and(|free| free < PROTECT_COMMIT_BYTES)
    {
        return refused(
            "Свободный commit меньше 2 ГиБ. Запуск отменён, pagefile не меняется и чужие процессы не закрываются.",
        );
    }
    HostDecision {
        allowed: true,
        reason: "Запас RAM и commit достаточен.".into(),
        kills_processes: false,
        changes_system_settings: false,
    }
}

fn refused(reason: &str) -> HostDecision {
    HostDecision {
        allowed: false,
        reason: reason.into(),
        kills_processes: false,
        changes_system_settings: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protect_mode_keeps_the_one_and_a_half_gib_ram_floor() {
        let decision = host_pressure(
            HostSample {
                ram_available_bytes: PROTECT_RAM_BYTES - 1,
                commit_free_bytes: Some(8 * 1024 * 1024 * 1024),
            },
            true,
        );
        assert!(!decision.allowed);
        assert!(decision.reason.contains("1.5"));
        assert!(!decision.kills_processes);
        assert!(!decision.changes_system_settings);
    }

    #[test]
    fn low_commit_headroom_defers_without_touching_other_processes() {
        let decision = host_pressure(
            HostSample {
                ram_available_bytes: 8 * 1024 * 1024 * 1024,
                commit_free_bytes: Some(PROTECT_COMMIT_BYTES - 1),
            },
            true,
        );
        assert!(!decision.allowed);
        assert!(decision.reason.contains("commit"));
        assert!(!decision.kills_processes);
    }

    #[test]
    fn missing_commit_reading_does_not_invent_pressure() {
        let decision = host_pressure(
            HostSample {
                ram_available_bytes: 8 * 1024 * 1024 * 1024,
                commit_free_bytes: None,
            },
            true,
        );
        assert!(decision.allowed);
    }

    #[test]
    fn protection_off_does_not_add_a_new_refusal() {
        let decision = host_pressure(
            HostSample {
                ram_available_bytes: 0,
                commit_free_bytes: Some(0),
            },
            false,
        );
        assert!(decision.allowed);
        assert!(!decision.kills_processes);
    }

    #[test]
    fn one_hundred_thousand_decisions_stay_under_a_millisecond_each() {
        let sample = HostSample {
            ram_available_bytes: 5 * 1024 * 1024 * 1024,
            commit_free_bytes: Some(12 * 1024 * 1024 * 1024),
        };
        let started = std::time::Instant::now();
        for _ in 0..100_000 {
            assert!(host_pressure(sample, true).allowed);
        }
        let elapsed = started.elapsed();
        assert!(
            elapsed.as_millis() < 500,
            "100000 decisions took {elapsed:?}"
        );
    }

    #[test]
    fn a_comfortable_host_is_allowed() {
        let decision = host_pressure(
            HostSample {
                ram_available_bytes: 5 * 1024 * 1024 * 1024,
                commit_free_bytes: Some(12 * 1024 * 1024 * 1024),
            },
            true,
        );
        assert!(decision.allowed);
    }
}
