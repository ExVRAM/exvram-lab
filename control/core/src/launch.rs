//! llama-server argument list. No process is spawned here.

use crate::estimate::KvFormat;

pub const DEFAULT_THREADS_PROTECTED: u32 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchRequest {
    pub server_exe: String,
    pub model_path: String,
    pub port: u16,
    pub context_tokens: u32,
    pub kv: KvFormat,
    pub protect_pc: bool,
    pub logical_cpus: u32,
    pub batch: u32,
    pub ubatch: u32,
    pub flash_attn: bool,
    pub threads_override: Option<u32>,
}

pub fn thread_count(protect_pc: bool, logical_cpus: u32, threads_override: Option<u32>) -> u32 {
    if protect_pc {
        return DEFAULT_THREADS_PROTECTED;
    }
    let fallback = logical_cpus.clamp(2, 8).saturating_sub(2).max(2);
    threads_override.unwrap_or(fallback).clamp(1, 32)
}

pub fn server_arguments(request: &LaunchRequest) -> Vec<String> {
    let threads = thread_count(request.protect_pc, request.logical_cpus, request.threads_override).to_string();
    let kv = request.kv.as_arg().to_string();
    let flash = if request.flash_attn { "on" } else { "off" };
    vec![
        request.server_exe.clone(),
        "--model".into(),
        request.model_path.clone(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        request.port.to_string(),
        "--no-webui".into(),
        "--ctx-size".into(),
        request.context_tokens.to_string(),
        "--gpu-layers".into(),
        "999".into(),
        "--device".into(),
        "CUDA0".into(),
        "--split-mode".into(),
        "none".into(),
        "--fit".into(),
        "off".into(),
        "--load-mode".into(),
        "dio".into(),
        "--prio".into(),
        "-1".into(),
        "--threads".into(),
        threads.clone(),
        "--threads-batch".into(),
        threads,
        "--batch-size".into(),
        request.batch.to_string(),
        "--ubatch-size".into(),
        request.ubatch.to_string(),
        "--flash-attn".into(),
        flash.into(),
        "--cache-type-k".into(),
        kv.clone(),
        "--cache-type-v".into(),
        kv,
        "--parallel".into(),
        "1".into(),
        "--reasoning".into(),
        "off".into(),
        "--ctx-checkpoints".into(),
        "0".into(),
        "--cache-ram".into(),
        "0".into(),
        "--verbosity".into(),
        "3".into(),
    ]
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendPhase {
    Stopped,
    Starting,
    Running,
    Failed(String),
}

pub fn reconcile_owned_process(pid_alive: bool) -> BackendPhase {
    if pid_alive {
        BackendPhase::Running
    } else {
        BackendPhase::Stopped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(protect: bool) -> LaunchRequest {
        LaunchRequest {
            server_exe: "llama-server".into(),
            model_path: "model.gguf".into(),
            port: 17860,
            context_tokens: 8448,
            kv: KvFormat::Q4,
            protect_pc: protect,
            logical_cpus: 16,
            batch: 32,
            ubatch: 1,
            flash_attn: true,
            threads_override: None,
        }
    }

    #[test]
    fn protect_mode_caps_threads_and_binds_loopback() {
        let args = server_arguments(&request(true));
        assert!(args.contains(&"127.0.0.1".into()));
        assert!(args.windows(2).any(|pair| pair == ["--threads", "4"]));
        assert!(args.windows(2).any(|pair| pair == ["--cache-type-k", "q4_0"]));
        assert!(args.windows(2).any(|pair| pair == ["--ctx-size", "8448"]));
        assert!(!args.iter().any(|arg| arg.contains("cmd.exe") || arg == "/c"));
    }

    #[test]
    fn kv_change_is_a_real_flag() {
        let mut request = request(true);
        request.kv = KvFormat::Q8;
        let args = server_arguments(&request);
        assert!(args.windows(2).any(|pair| pair == ["--cache-type-k", "q8_0"]));
        assert!(args.windows(2).any(|pair| pair == ["--cache-type-v", "q8_0"]));
    }

    #[test]
    fn a_dead_owned_pid_is_stopped_not_running() {
        assert_eq!(reconcile_owned_process(false), BackendPhase::Stopped);
        assert_eq!(reconcile_owned_process(true), BackendPhase::Running);
    }

    #[test]
    fn unprotected_threads_still_leave_cores_free() {
        assert!(thread_count(false, 16, None) < 16);
        assert_eq!(thread_count(true, 32, Some(16)), 4);
    }
}
