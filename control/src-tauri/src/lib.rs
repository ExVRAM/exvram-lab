//! Desktop shell. Decisions live in exvram-control-core. This file only
//! talks to nvidia-smi, the local llama-server process, and the window.

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use exvram_control_core::estimate::{estimate, launch_gate, preset_allowed, KvFormat, CONTEXT_PRESETS};
use exvram_control_core::gpu::parse_nvidia_smi_line;
use exvram_control_core::launch::{server_arguments, LaunchRequest};
use exvram_control_core::recipe::{identity_matches, parse_recipe_yaml, PublishedRecipe};
use exvram_control_core::recommend::recommend;
use exvram_control_core::settings::{apply_profile, Settings};
use exvram_control_core::slots::{parse_slots, parse_speed_metrics};
use exvram_control_core::storage::{may_adopt_process, model_fits_physical_ram, storage_allowed};
use serde::Serialize;
use sysinfo::System;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State, WindowEvent};
use tauri_plugin_notification::NotificationExt;

const MISSING: &str = "Нет данных";

struct Inner {
    settings: Settings,
    system: System,
    cpu_ready: bool,
    child: Option<Child>,
    port: Option<u16>,
    generation: u64,
    phase: String,
    notice: String,
    launched_context: Option<u32>,
    launched_kv: Option<KvFormat>,
}

struct AppState {
    inner: Mutex<Inner>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PresetRow {
    tokens: u32,
    label: String,
    allowed: bool,
    estimate: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelRow {
    name: String,
    path: String,
    verified: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Panel {
    model_name: String,
    parameters: String,
    quant: String,
    backend: String,
    load_state: String,
    gpu_name: String,
    vram_total: String,
    vram_used: String,
    vram_free: String,
    temperature: String,
    gpu_load: String,
    decode: String,
    prefill: String,
    ttft: String,
    cpu: String,
    ram: String,
    configured_context: String,
    occupied_context: String,
    free_context: String,
    kv: String,
    kv_memory: String,
    protect: bool,
    mode: String,
    profile: String,
    phase: String,
    notice: String,
    recommendation: String,
    memory_change: String,
    speed_change: String,
    quality_risk: String,
    restart_required: bool,
    presets: Vec<PresetRow>,
    models: Vec<ModelRow>,
    batch: u32,
    ubatch: u32,
    flash_attn: bool,
    threads: String,
    server_label: String,
}

fn missing() -> String {
    MISSING.to_string()
}

fn app_dir() -> PathBuf {
    let base = std::env::var("LOCALAPPDATA")
        .or_else(|_| std::env::var("APPDATA"))
        .unwrap_or_else(|_| std::env::temp_dir().display().to_string());
    let dir = PathBuf::from(base).join("ExVRAM").join("control");
    let _ = fs::create_dir_all(&dir);
    dir
}

fn settings_path() -> PathBuf {
    app_dir().join("settings.json")
}

fn pid_path() -> PathBuf {
    app_dir().join("owned-process.json")
}

fn log_path() -> PathBuf {
    app_dir().join("llama-server.log")
}

fn load_settings() -> Settings {
    let Ok(text) = fs::read_to_string(settings_path()) else {
        return Settings::default();
    };
    serde_json::from_str::<Settings>(&text)
        .ok()
        .and_then(|settings| settings.normalized().ok())
        .unwrap_or_default()
}

fn save_settings(settings: &Settings) -> Result<(), String> {
    let settings = settings.clone().normalized()?;
    let text = serde_json::to_string_pretty(&settings).map_err(|err| err.to_string())?;
    fs::write(settings_path(), text).map_err(|err| err.to_string())
}

fn find_repo(start: &Path) -> Option<PathBuf> {
    let mut cursor = Some(start);
    while let Some(dir) = cursor {
        if dir
            .join("recipes")
            .join("rtx5060_8gb_qwen38_27b_8k.yaml")
            .is_file()
        {
            return Some(dir.to_path_buf());
        }
        cursor = dir.parent();
    }
    None
}

fn repo_root() -> Option<PathBuf> {
    if let Ok(dir) = std::env::current_dir() {
        if let Some(found) = find_repo(&dir) {
            return Some(found);
        }
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
        .and_then(|dir| find_repo(&dir))
}

fn published_recipe() -> Option<PublishedRecipe> {
    let path = repo_root()?.join("recipes").join("rtx5060_8gb_qwen38_27b_8k.yaml");
    let text = fs::read_to_string(path).ok()?;
    parse_recipe_yaml(&text).ok()
}

fn drive_kind(path: &Path) -> Option<u32> {
    #[cfg(windows)]
    {
        let text = path.to_string_lossy();
        let mut chars = text.chars();
        let letter = chars.next()?;
        if chars.next() != Some(':') {
            return None;
        }
        let root: Vec<u16> = format!("{letter}:\\")
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        extern "system" {
            fn GetDriveTypeW(root: *const u16) -> u32;
        }
        return Some(unsafe { GetDriveTypeW(root.as_ptr()) });
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        None
    }
}

fn check_path(path: &Path) -> Result<(), String> {
    storage_allowed(&path.to_string_lossy(), drive_kind(path))
}

fn recipe_applies(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    let Ok(bytes) = fs::metadata(path).map(|meta| meta.len()) else {
        return false;
    };
    identity_matches(name, bytes, published_recipe().as_ref())
}

fn discover_server(root: &Path) -> Option<PathBuf> {
    let cache = root.join(".cache").join("llama");
    let mut hits = Vec::new();
    let entries = fs::read_dir(cache).ok()?;
    for entry in entries.flatten() {
        let candidate = entry.path().join("llama-server.exe");
        if candidate.is_file() {
            hits.push(candidate);
        }
    }
    hits.sort();
    hits.pop()
}

fn logical_cpus(system: &System) -> u32 {
    system.cpus().len().max(1) as u32
}

fn sample_gpu() -> Result<exvram_control_core::GpuSample, String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().find(|line| !line.trim().is_empty()).unwrap_or("");
    parse_nvidia_smi_line(line)
}

fn http_get(port: u16, path: &str) -> Result<String, String> {
    let address = format!("127.0.0.1:{port}")
        .parse()
        .map_err(|err: std::net::AddrParseError| err.to_string())?;
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(350))
        .map_err(|err| err.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_millis(350)))
        .ok();
    stream
        .set_write_timeout(Some(Duration::from_millis(350)))
        .ok();
    write!(
        stream,
        "GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
    )
    .map_err(|err| err.to_string())?;
    let mut buffer = String::new();
    stream
        .read_to_string(&mut buffer)
        .map_err(|err| err.to_string())?;
    Ok(buffer
        .split_once("\r\n\r\n")
        .map(|(_, body)| body.to_string())
        .unwrap_or_default())
}

fn free_port() -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|err| err.to_string())?;
    listener
        .local_addr()
        .map(|addr| addr.port())
        .map_err(|err| err.to_string())
}

fn process_image(pid: u32) -> Option<String> {
    #[cfg(windows)]
    {
        extern "system" {
            fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut core::ffi::c_void;
            fn QueryFullProcessImageNameW(
                process: *mut core::ffi::c_void,
                flags: u32,
                buffer: *mut u16,
                size: *mut u32,
            ) -> i32;
            fn CloseHandle(handle: *mut core::ffi::c_void) -> i32;
        }
        const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle.is_null() {
                return None;
            }
            let mut buffer = [0u16; 1024];
            let mut length = buffer.len() as u32;
            let ok = QueryFullProcessImageNameW(handle, 0, buffer.as_mut_ptr(), &mut length);
            CloseHandle(handle);
            if ok == 0 {
                return None;
            }
            return Some(String::from_utf16_lossy(&buffer[..length as usize]));
        }
    }
    #[cfg(not(windows))]
    {
        let _ = pid;
        None
    }
}

fn remember_pid(pid: u32, port: u16) {
    let _ = fs::write(
        pid_path(),
        format!(r#"{{"pid":{pid},"port":{port}}}"#),
    );
}

fn forget_pid() {
    let _ = fs::remove_file(pid_path());
}

fn command_matches_owned_port(system: &mut System, pid: u32, port: u16) -> bool {
    system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    let Some(process) = system.process(sysinfo::Pid::from_u32(pid)) else {
        return false;
    };
    let args: Vec<String> = process
        .cmd()
        .iter()
        .map(|part| part.to_string_lossy().into_owned())
        .collect();
    let port_text = port.to_string();
    let has_port = args.windows(2).any(|pair| pair[0] == "--port" && pair[1] == port_text);
    let loopback = args.windows(2).any(|pair| pair[0] == "--host" && pair[1] == "127.0.0.1");
    has_port && loopback
}

fn adopt_orphaned(inner: &mut Inner) {
    if inner.child.is_some() {
        return;
    }
    let Ok(text) = fs::read_to_string(pid_path()) else {
        return;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        forget_pid();
        return;
    };
    let Some(pid) = value.get("pid").and_then(|item| item.as_u64()) else {
        forget_pid();
        return;
    };
    let Some(image) = process_image(pid as u32) else {
        forget_pid();
        return;
    };
    if !may_adopt_process(&image) {
        forget_pid();
        return;
    }
    let Some(port) = value.get("port").and_then(|item| item.as_u64()).map(|port| port as u16) else {
        forget_pid();
        return;
    };
    if !command_matches_owned_port(&mut inner.system, pid as u32, port) {
        forget_pid();
        return;
    }
    inner.port = Some(port);
    inner.phase = "running".into();
    inner.notice = "Найден свой llama-server после прошлого закрытия панели. Чужие процессы не трогаются.".into();
    inner.generation += 1;
}

fn stop_owned(inner: &mut Inner) {
    if let Some(child) = inner.child.as_mut() {
        let pid = child.id();
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            let _ = Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .creation_flags(0x08000000)
                .status();
        }
        #[cfg(not(windows))]
        {
            let _ = child.kill();
        }
        let _ = child.wait();
    } else if let Ok(text) = fs::read_to_string(pid_path()) {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(pid) = value.get("pid").and_then(|item| item.as_u64()) {
                let port = value.get("port").and_then(|item| item.as_u64()).unwrap_or(0) as u16;
                let owned = process_image(pid as u32).is_some_and(|image| may_adopt_process(&image))
                    && command_matches_owned_port(&mut inner.system, pid as u32, port);
                if owned {
                    #[cfg(windows)]
                    {
                        use std::os::windows::process::CommandExt;
                        let _ = Command::new("taskkill")
                            .args(["/PID", &pid.to_string(), "/T", "/F"])
                            .creation_flags(0x08000000)
                            .status();
                    }
                }
            }
        }
    }
    inner.child = None;
    inner.port = None;
    inner.phase = "stopped".into();
    inner.launched_context = None;
    inner.launched_kv = None;
    forget_pid();
}

fn notify(app: &AppHandle, body: &str) {
    let _ = app
        .notification()
        .builder()
        .title("ExVRAM Control")
        .body(body)
        .show();
}

fn mib_text(value: f64, basis: &str) -> String {
    format!("{value:.0} МиБ ({basis})")
}

fn build_panel(inner: &mut Inner) -> Panel {
    if !inner.cpu_ready {
        inner.system.refresh_cpu_usage();
        std::thread::sleep(Duration::from_millis(200));
        inner.cpu_ready = true;
    }
    inner.system.refresh_cpu_usage();
    inner.system.refresh_memory();
    let cpu = inner.system.global_cpu_usage();
    let cpu_text = if inner.cpu_ready && cpu > 0.0 {
        format!("{cpu:.0}%")
    } else {
        missing()
    };
    let ram_total = inner.system.total_memory();
    let ram_used = inner.system.used_memory();
    let ram_text = if ram_total == 0 {
        missing()
    } else {
        format!(
            "{:.1} / {:.1} ГиБ",
            ram_used as f64 / 1024.0 / 1024.0 / 1024.0,
            ram_total as f64 / 1024.0 / 1024.0 / 1024.0
        )
    };

    let gpu = sample_gpu().ok();
    let (gpu_name, vram_total, vram_used, vram_free, temperature, gpu_load) = match &gpu {
        Some(sample) => (
            sample.name.clone(),
            format!("{} МиБ", sample.total_mib),
            format!("{} МиБ", sample.used_mib),
            format!("{} МиБ", sample.free_mib),
            sample
                .temperature_c
                .map(|value| format!("{value} °C"))
                .unwrap_or_else(missing),
            sample
                .utilization_percent
                .map(|value| format!("{value}%"))
                .unwrap_or_else(missing),
        ),
        None => (
            missing(),
            missing(),
            missing(),
            missing(),
            missing(),
            missing(),
        ),
    };

    if let Some(child) = inner.child.as_mut() {
        if let Ok(Some(status)) = child.try_wait() {
            inner.phase = "failed".into();
            inner.notice = format!(
                "llama-server завершился сам (код {status}). Панель его не перезапускает."
            );
            inner.child = None;
            inner.port = None;
            forget_pid();
        }
    } else if inner.phase == "running" {
        let alive = fs::read_to_string(pid_path()).ok().and_then(|text| {
            let value: serde_json::Value = serde_json::from_str(&text).ok()?;
            let pid = value.get("pid")?.as_u64()?;
            process_image(pid as u32).filter(|image| may_adopt_process(image))
        });
        if alive.is_none() {
            inner.phase = "stopped".into();
            inner.notice = "Свой llama-server уже не запущен.".into();
            forget_pid();
        }
    }

    let mut decode = missing();
    let mut prefill = missing();
    let ttft = missing();
    let mut occupied: Option<u32> = None;
    let mut configured_live: Option<u32> = None;
    if inner.phase == "running" {
        if let Some(port) = inner.port {
            if let Ok(body) = http_get(port, "/slots") {
                if let Ok(view) = parse_slots(&body) {
                    occupied = view.occupied_tokens;
                    configured_live = view.configured_context;
                }
            }
            if let Ok(body) = http_get(port, "/metrics") {
                let (prompt, predicted) = parse_speed_metrics(&body);
                if let Some(value) = predicted {
                    decode = format!("{value:.2} tok/s");
                }
                if let Some(value) = prompt {
                    prefill = format!("{value:.2} tok/s");
                }
            }
        }
    }

    let settings = inner.settings.clone();
    let model_path = settings.model_path.clone().unwrap_or_default();
    let model = PathBuf::from(&model_path);
    let applies = !model_path.is_empty() && recipe_applies(&model);
    let model_name = model
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(MISSING)
        .to_string();
    let (parameters, quant) = if applies {
        ("27B".to_string(), "UD-IQ2_XXS".to_string())
    } else {
        (missing(), missing())
    };
    let kv = settings.kv_format().unwrap_or(KvFormat::Q4);
    let memory = estimate(settings.context_tokens, kv);
    let kv_memory = if applies {
        mib_text(memory.kv_mib, memory.basis)
    } else {
        missing()
    };
    let total_vram = gpu.as_ref().map(|sample| sample.total_mib).unwrap_or(0);
    let free_vram = gpu.as_ref().map(|sample| sample.free_mib).unwrap_or(0);
    let advice = recommend(
        total_vram,
        free_vram,
        settings.protect_pc,
        &settings.mode,
        inner.launched_context,
        inner.launched_kv,
        applies,
    );
    let restart_required = inner.phase == "running"
        && (inner.launched_context.is_some_and(|value| value != settings.context_tokens)
            || inner.launched_kv.is_some_and(|value| value != kv));
    let memory_change = if advice.memory_applies {
        format!(
            "{} → {:.0} МиБ, {}",
            settings.context_tokens, advice.expected_total_mib, advice.memory_basis
        )
    } else {
        missing()
    };
    let configured = configured_live.unwrap_or(settings.context_tokens);
    let free_context = occupied
        .map(|used| configured.saturating_sub(used).to_string())
        .unwrap_or_else(missing);
    let presets = CONTEXT_PRESETS
        .iter()
        .copied()
        .map(|tokens| {
            let allowed = applies && total_vram > 0 && preset_allowed(tokens, total_vram, kv, settings.protect_pc);
            let estimate = if applies {
                let item = estimate(tokens, kv);
                mib_text(item.total_mib, item.basis)
            } else {
                missing()
            };
            let label = match tokens {
                4096 => "4K",
                8192 => "8K",
                8448 => "8K*",
                16384 => "16K",
                32768 => "32K",
                65536 => "64K",
                _ => "Custom",
            };
            PresetRow {
                tokens,
                label: label.into(),
                allowed,
                estimate,
            }
        })
        .collect();
    let models = list_models(&settings);
    let load_state = match inner.phase.as_str() {
        "running" => "запущена",
        "starting" => "загрузка",
        "failed" => "ошибка запуска",
        _ => "остановлена",
    };

    Panel {
        model_name: if model_path.is_empty() { missing() } else { model_name },
        parameters,
        quant,
        backend: "llama.cpp".into(),
        load_state: load_state.into(),
        gpu_name,
        vram_total,
        vram_used,
        vram_free,
        temperature,
        gpu_load,
        decode,
        prefill,
        ttft,
        cpu: cpu_text,
        ram: ram_text,
        configured_context: format!("{configured} токенов"),
        occupied_context: occupied
            .map(|value| format!("{value} токенов"))
            .unwrap_or_else(missing),
        free_context: if occupied.is_some() {
            format!("{free_context} токенов")
        } else {
            missing()
        },
        kv: kv.as_arg().into(),
        kv_memory,
        protect: settings.protect_pc,
        mode: settings.mode.clone(),
        profile: settings.profile.clone(),
        phase: inner.phase.clone(),
        notice: inner.notice.clone(),
        recommendation: advice.explanation,
        memory_change,
        speed_change: advice.speed_note,
        quality_risk: advice.quality_risk,
        restart_required: restart_required || advice.restart_required && inner.phase == "running",
        presets,
        models,
        batch: settings.batch,
        ubatch: settings.ubatch,
        flash_attn: settings.flash_attn,
        threads: settings
            .threads_override
            .map(|value| value.to_string())
            .unwrap_or_else(|| "авто".into()),
        server_label: settings
            .server_path
            .as_ref()
            .map(|path| Path::new(path).file_name().and_then(|name| name.to_str()).unwrap_or(path).to_string())
            .unwrap_or_else(missing),
    }
}

fn list_models(settings: &Settings) -> Vec<ModelRow> {
    let dir = settings
        .models_dir
        .as_ref()
        .map(PathBuf::from)
        .or_else(|| repo_root().map(|root| root.join("models")));
    let Some(dir) = dir else {
        return Vec::new();
    };
    if check_path(&dir).is_err() {
        return Vec::new();
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for entry in entries.flatten().take(200) {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
            continue;
        }
        rows.push(ModelRow {
            name: path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model")
                .to_string(),
            path: path.display().to_string(),
            verified: recipe_applies(&path),
        });
    }
    rows.sort_by(|left, right| right.verified.cmp(&left.verified).then(left.name.cmp(&right.name)));
    rows
}

fn mutate<T>(state: &AppState, change: impl FnOnce(&mut Inner) -> Result<T, String>) -> Result<T, String> {
    let mut inner = state.inner.lock().map_err(|err| err.to_string())?;
    change(&mut inner)
}

#[tauri::command]
fn snapshot(state: State<'_, AppState>) -> Result<Panel, String> {
    let mut inner = state.inner.lock().map_err(|err| err.to_string())?;
    Ok(build_panel(&mut inner))
}

#[tauri::command]
fn pick_model(state: State<'_, AppState>) -> Result<Panel, String> {
    let file = rfd::FileDialog::new()
        .set_title("Модель GGUF")
        .add_filter("GGUF", &["gguf"])
        .pick_file();
    let Some(file) = file else {
        return snapshot(state);
    };
    check_path(&file)?;
    mutate(&state, |inner| {
        inner.settings.model_path = Some(file.display().to_string());
        save_settings(&inner.settings)?;
        Ok(build_panel(inner))
    })
}

#[tauri::command]
fn pick_server(state: State<'_, AppState>) -> Result<Panel, String> {
    let file = rfd::FileDialog::new()
        .set_title("llama-server")
        .add_filter("llama-server", &["exe"])
        .pick_file();
    let Some(file) = file else {
        return snapshot(state);
    };
    check_path(&file)?;
    let name = file.file_name().and_then(|value| value.to_str()).unwrap_or("");
    if !name.eq_ignore_ascii_case("llama-server.exe") {
        return Err("Нужен именно llama-server.exe".into());
    }
    mutate(&state, |inner| {
        inner.settings.server_path = Some(file.display().to_string());
        save_settings(&inner.settings)?;
        Ok(build_panel(inner))
    })
}

#[tauri::command]
fn pick_models_dir(state: State<'_, AppState>) -> Result<Panel, String> {
    let dir = rfd::FileDialog::new().set_title("Папка моделей").pick_folder();
    let Some(dir) = dir else {
        return snapshot(state);
    };
    check_path(&dir)?;
    mutate(&state, |inner| {
        inner.settings.models_dir = Some(dir.display().to_string());
        save_settings(&inner.settings)?;
        Ok(build_panel(inner))
    })
}

#[tauri::command]
fn select_listed_model(state: State<'_, AppState>, path: String) -> Result<Panel, String> {
    let file = PathBuf::from(&path);
    check_path(&file)?;
    if file.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
        return Err("Ожидается файл .gguf".into());
    }
    mutate(&state, |inner| {
        inner.settings.model_path = Some(path);
        save_settings(&inner.settings)?;
        Ok(build_panel(inner))
    })
}

#[tauri::command]
fn set_context(state: State<'_, AppState>, tokens: u32) -> Result<Panel, String> {
    if !(256..=131072).contains(&tokens) {
        return Err("Контекст вне диапазона 256..131072".into());
    }
    mutate(&state, |inner| {
        let model = inner.settings.model_path.clone().unwrap_or_default();
        let applies = recipe_applies(Path::new(&model));
        if applies {
            let gpu = sample_gpu().map_err(|_| "Нет данных по VRAM".to_string())?;
            let kv = inner.settings.kv_format()?;
            if !preset_allowed(tokens, gpu.total_mib, kv, inner.settings.protect_pc) {
                let estimate = estimate(tokens, kv);
                return Err(launch_gate(gpu.total_mib, gpu.total_mib, estimate.total_mib, inner.settings.protect_pc).reason);
            }
        }
        inner.settings.context_tokens = tokens;
        inner.settings.profile = "custom".into();
        save_settings(&inner.settings)?;
        if inner.phase == "running" {
            inner.notice = "Новый контекст применится при следующем запуске. Сейчас сервер не перезапускается.".into();
        }
        Ok(build_panel(inner))
    })
}

#[tauri::command]
fn set_kv(state: State<'_, AppState>, kv: String) -> Result<Panel, String> {
    let parsed = KvFormat::parse(&kv).ok_or_else(|| format!("KV {kv} не поддерживается"))?;
    mutate(&state, |inner| {
        let model = inner.settings.model_path.clone().unwrap_or_default();
        if recipe_applies(Path::new(&model)) {
            if let Ok(gpu) = sample_gpu() {
                if !preset_allowed(inner.settings.context_tokens, gpu.total_mib, parsed, inner.settings.protect_pc) {
                    return Err("Этот KV не оставляет запас VRAM на текущем контексте".into());
                }
            }
        }
        inner.settings.kv = parsed.as_arg().into();
        save_settings(&inner.settings)?;
        if inner.phase == "running" {
            inner.notice = "KV применится при следующем запуске.".into();
        } else if parsed != KvFormat::Q4 {
            inner.notice = "q8 и f16 не являются рекомендацией: quality gate = NOT_RUN, скорость q8 на занятых 8192 нестабильна, f16 там не домерен.".into();
        }
        Ok(build_panel(inner))
    })
}

#[tauri::command]
fn set_mode(state: State<'_, AppState>, mode: String) -> Result<Panel, String> {
    if !matches!(mode.as_str(), "balanced" | "performance" | "quality") {
        return Err("Неизвестный режим".into());
    }
    mutate(&state, |inner| {
        inner.settings.mode = mode;
        let model = inner.settings.model_path.clone().unwrap_or_default();
        let applies = recipe_applies(Path::new(&model));
        if applies {
            if let Ok(gpu) = sample_gpu() {
                let advice = recommend(
                    gpu.total_mib,
                    gpu.free_mib,
                    inner.settings.protect_pc,
                    &inner.settings.mode,
                    inner.launched_context,
                    inner.launched_kv,
                    true,
                );
                inner.settings.context_tokens = advice.context_tokens;
                inner.settings.kv = advice.kv.as_arg().into();
                inner.notice = advice.quality_risk.clone();
            }
        }
        save_settings(&inner.settings)?;
        Ok(build_panel(inner))
    })
}

#[tauri::command]
fn set_profile(state: State<'_, AppState>, profile: String) -> Result<Panel, String> {
    mutate(&state, |inner| {
        if profile == "custom" {
            inner.settings.profile = profile;
            save_settings(&inner.settings)?;
            inner.notice = "Custom оставляет текущий контекст.".into();
            return Ok(build_panel(inner));
        }
        let gpu = sample_gpu().map_err(|_| "Нет данных по VRAM".to_string())?;
        let context = apply_profile(&profile, gpu.total_mib, inner.settings.protect_pc)?;
        inner.settings.profile = profile;
        inner.settings.context_tokens = context;
        inner.settings.kv = KvFormat::Q4.as_arg().into();
        save_settings(&inner.settings)?;
        inner.notice = format!("Профиль выставил контекст {context}. Это настройка llama.cpp, не отдельный режим качества.");
        Ok(build_panel(inner))
    })
}

#[tauri::command]
fn set_protect(state: State<'_, AppState>, enabled: bool) -> Result<Panel, String> {
    mutate(&state, |inner| {
        inner.settings.protect_pc = enabled;
        save_settings(&inner.settings)?;
        inner.notice = if enabled {
            "Protect My PC включён: 4 потока, низкий приоритет, отказ при чужой VRAM свыше 2048 МиБ.".into()
        } else {
            "Protect My PC выключен. Потоков больше, запас VRAM меньше. Чужие программы всё равно не закрываются.".into()
        };
        Ok(build_panel(inner))
    })
}

#[tauri::command]
fn apply_advanced(
    state: State<'_, AppState>,
    batch: u32,
    ubatch: u32,
    flash_attn: bool,
    threads: Option<u32>,
    acknowledged: bool,
) -> Result<Panel, String> {
    let mut next = {
        let inner = state.inner.lock().map_err(|err| err.to_string())?;
        let mut copy = inner.settings.clone();
        copy.batch = batch;
        copy.ubatch = ubatch;
        copy.flash_attn = flash_attn;
        copy.threads_override = threads;
        copy.normalized()?
    };
    let mut warnings = Vec::new();
    if next.batch != 32 || next.ubatch != 1 {
        warnings.push("batch/ubatch отличаются от проверенного 32/1".to_string());
    }
    if !next.flash_attn {
        warnings.push("Flash Attention будет выключен".to_string());
    }
    if next.threads_override.is_some() && next.protect_pc {
        warnings.push("Protect My PC всё равно оставит 4 потока".to_string());
    }
    if !warnings.is_empty() && !acknowledged {
        return Err(format!("Проверьте перед применением: {}", warnings.join("; ")));
    }
    mutate(&state, |inner| {
        next.model_path = inner.settings.model_path.clone();
        next.server_path = inner.settings.server_path.clone();
        next.models_dir = inner.settings.models_dir.clone();
        next.context_tokens = inner.settings.context_tokens;
        next.kv = inner.settings.kv.clone();
        next.mode = inner.settings.mode.clone();
        next.profile = inner.settings.profile.clone();
        next.protect_pc = inner.settings.protect_pc;
        inner.settings = next;
        save_settings(&inner.settings)?;
        if inner.phase == "running" {
            inner.notice = "Расширенные параметры применятся при следующем запуске.".into();
        }
        Ok(build_panel(inner))
    })
}

#[tauri::command]
fn reset_recommended(state: State<'_, AppState>) -> Result<Panel, String> {
    mutate(&state, |inner| {
        inner.settings.context_tokens = exvram_control_core::estimate::MEASURED_CONTEXT;
        inner.settings.kv = KvFormat::Q4.as_arg().into();
        inner.settings.mode = "balanced".into();
        inner.settings.batch = 32;
        inner.settings.ubatch = 1;
        inner.settings.flash_attn = true;
        inner.settings.threads_override = None;
        inner.settings.protect_pc = true;
        save_settings(&inner.settings)?;
        inner.notice = "Сброс к проверенному рецепту: q4_0, окно 8448, batch 32. Запуск всё равно проверяет свободную VRAM.".into();
        Ok(build_panel(inner))
    })
}

#[tauri::command]
fn export_settings(state: State<'_, AppState>) -> Result<(), String> {
    let path = rfd::FileDialog::new()
        .set_title("Экспорт настроек")
        .set_file_name("exvram-control.json")
        .save_file();
    let Some(path) = path else {
        return Ok(());
    };
    check_path(&path)?;
    let inner = state.inner.lock().map_err(|err| err.to_string())?;
    let text = serde_json::to_string_pretty(&inner.settings).map_err(|err| err.to_string())?;
    fs::write(path, text).map_err(|err| err.to_string())
}

#[tauri::command]
fn import_settings(state: State<'_, AppState>) -> Result<Panel, String> {
    let path = rfd::FileDialog::new()
        .set_title("Импорт настроек")
        .add_filter("JSON", &["json"])
        .pick_file();
    let Some(path) = path else {
        return snapshot(state);
    };
    check_path(&path)?;
    let text = fs::read_to_string(&path).map_err(|err| err.to_string())?;
    let loaded: Settings = serde_json::from_str(&text).map_err(|err| err.to_string())?;
    let loaded = loaded.normalized()?;
    if let Some(model) = &loaded.model_path {
        check_path(Path::new(model))?;
    }
    if let Some(server) = &loaded.server_path {
        check_path(Path::new(server))?;
    }
    mutate(&state, |inner| {
        inner.settings = loaded;
        save_settings(&inner.settings)?;
        Ok(build_panel(inner))
    })
}

#[tauri::command]
fn start(app: AppHandle, state: State<'_, AppState>) -> Result<Panel, String> {
    let mut inner = state.inner.lock().map_err(|err| err.to_string())?;
    if inner.child.is_some() || inner.phase == "running" || inner.phase == "starting" {
        inner.notice = "Сервер уже запускается или работает. Второй экземпляр не создаётся.".into();
        return Ok(build_panel(&mut inner));
    }
    let settings = inner.settings.clone();
    let model = settings
        .model_path
        .clone()
        .ok_or_else(|| "Сначала выберите модель".to_string())?;
    let model_path = PathBuf::from(&model);
    check_path(&model_path)?;
    if !model_path.is_file() {
        return Err("Файл модели не найден".into());
    }
    let server = settings.server_path.clone().or_else(|| {
        repo_root().and_then(|root| discover_server(&root).map(|path| path.display().to_string()))
    });
    let server = server.ok_or_else(|| "llama-server.exe не найден. Укажите его вручную.".to_string())?;
    let server_path = PathBuf::from(&server);
    check_path(&server_path)?;
    if !may_adopt_process(&server) {
        return Err("Запускается только llama-server".into());
    }
    let applies = recipe_applies(&model_path);
    let file_bytes = fs::metadata(&model_path).map(|meta| meta.len()).unwrap_or(0);
    let physical = inner.system.total_memory();
    if !model_fits_physical_ram(file_bytes, physical) {
        return Err("Файл модели больше физической RAM. Запуск отменён, pagefile не меняется.".into());
    }
    let gpu = sample_gpu()?;
    if applies {
        let kv = settings.kv_format()?;
        let estimate = estimate(settings.context_tokens, kv);
        let gate = launch_gate(gpu.total_mib, gpu.free_mib, estimate.total_mib, settings.protect_pc);
        if !gate.allowed {
            inner.notice = gate.reason.clone();
            notify(&app, &gate.reason);
            return Ok(build_panel(&mut inner));
        }
    } else if settings.protect_pc {
        let reason = "Для этой модели нет проверенной оценки памяти. Protect My PC не запускает её вслепую.";
        inner.notice = reason.into();
        notify(&app, reason);
        return Ok(build_panel(&mut inner));
    }
    let available = inner.system.available_memory();
    if settings.protect_pc && available < 1536 * 1024 * 1024 {
        let reason = "Свободной RAM меньше 1.5 ГиБ. Запуск отменён, чтобы не увести систему в pagefile.";
        inner.notice = reason.into();
        notify(&app, reason);
        return Ok(build_panel(&mut inner));
    }
    let port = free_port()?;
    let kv = settings.kv_format()?;
    let request = LaunchRequest {
        server_exe: server,
        model_path: model,
        port,
        context_tokens: settings.context_tokens,
        kv,
        protect_pc: settings.protect_pc,
        logical_cpus: logical_cpus(&inner.system),
        batch: settings.batch,
        ubatch: settings.ubatch,
        flash_attn: settings.flash_attn,
        threads_override: settings.threads_override,
    };
    let args = server_arguments(&request);
    if args.iter().any(|arg| arg == "0.0.0.0" || arg.contains("cmd.exe")) {
        return Err("Команда запуска отклонена".into());
    }
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path())
        .map_err(|err| err.to_string())?;
    let err_log = log.try_clone().map_err(|err| err.to_string())?;
    let mut command = Command::new(&args[0]);
    command
        .args(&args[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(err_log));
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000 | 0x00004000);
    }
    let child = command.spawn().map_err(|err| {
        format!("Не удалось запустить llama-server: {err}")
    })?;
    let pid = child.id();
    remember_pid(pid, port);
    inner.child = Some(child);
    inner.port = Some(port);
    inner.phase = "starting".into();
    inner.launched_context = Some(settings.context_tokens);
    inner.launched_kv = Some(kv);
    inner.generation += 1;
    let generation = inner.generation;
    inner.notice = "Загрузка модели. Окно можно свернуть в трей.".into();
    drop(inner);
    let handle = app.clone();
    std::thread::spawn(move || wait_until_ready(handle, generation, port));
    snapshot(state)
}

fn wait_until_ready(app: AppHandle, generation: u64, port: u16) {
    let deadline = std::time::Instant::now() + Duration::from_secs(180);
    loop {
        let state = app.state::<AppState>();
        let mut inner = match state.inner.lock() {
            Ok(inner) => inner,
            Err(_) => return,
        };
        if inner.generation != generation {
            return;
        }
        if let Some(child) = inner.child.as_mut() {
            if let Ok(Some(status)) = child.try_wait() {
                inner.phase = "failed".into();
                inner.child = None;
                inner.port = None;
                inner.notice = format!("llama-server завершился при загрузке (код {status}).");
                forget_pid();
                let notice = inner.notice.clone();
                drop(inner);
                notify(&app, &notice);
                return;
            }
        }
        drop(inner);
        if http_get(port, "/health").is_ok() {
            let mut inner = match state.inner.lock() {
                Ok(inner) => inner,
                Err(_) => return,
            };
            if inner.generation == generation {
                inner.phase = "running".into();
                inner.notice = "Модель отвечает на 127.0.0.1.".into();
            }
            return;
        }
        if std::time::Instant::now() > deadline {
            let mut inner = match state.inner.lock() {
                Ok(inner) => inner,
                Err(_) => return,
            };
            if inner.generation == generation {
                inner.notice = "Сервер не ответил за 180 с. Процесс ещё может грузить модель; панель его не убивает сама.".into();
            }
            return;
        }
        std::thread::sleep(Duration::from_millis(700));
    }
}

#[tauri::command]
fn stop(state: State<'_, AppState>) -> Result<Panel, String> {
    mutate(&state, |inner| {
        inner.generation += 1;
        stop_owned(inner);
        inner.notice = "Свой llama-server остановлен.".into();
        Ok(build_panel(inner))
    })
}

fn show_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
        let _ = window.emit("presence", "open");
    }
}

pub fn run() {
    let mut inner = Inner {
        settings: load_settings(),
        system: System::new(),
        cpu_ready: false,
        child: None,
        port: None,
        generation: 0,
        phase: "stopped".into(),
        notice: String::new(),
        launched_context: None,
        launched_kv: None,
    };
    if inner.settings.server_path.is_none() {
        if let Some(root) = repo_root() {
            if let Some(server) = discover_server(&root) {
                if check_path(&server).is_ok() {
                    inner.settings.server_path = Some(server.display().to_string());
                }
            }
        }
    }
    adopt_orphaned(&mut inner);
    let state = AppState {
        inner: Mutex::new(inner),
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            snapshot,
            pick_model,
            pick_server,
            pick_models_dir,
            select_listed_model,
            set_context,
            set_kv,
            set_mode,
            set_profile,
            set_protect,
            apply_advanced,
            reset_recommended,
            export_settings,
            import_settings,
            start,
            stop
        ])
        .setup(|app| {
            let show = MenuItem::with_id(app, "show", "Открыть", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Выход", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show, &quit])?;
            let icon = app
                .default_window_icon()
                .cloned()
                .expect("window icon");
            TrayIconBuilder::new()
                .menu(&menu)
                .tooltip("ExVRAM Control")
                .icon(icon)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => show_window(app),
                    "quit" => {
                        if let Some(state) = app.try_state::<AppState>() {
                            if let Ok(mut inner) = state.inner.lock() {
                                stop_owned(&mut inner);
                            }
                        }
                        app.exit(0);
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        show_window(tray.app_handle());
                    }
                })
                .build(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
                let _ = window.emit("presence", "tray");
            }
        })
        .build(tauri::generate_context!())
        .expect("ExVRAM Control failed to build")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(state) = app.try_state::<AppState>() {
                    if let Ok(mut inner) = state.inner.lock() {
                        stop_owned(&mut inner);
                    }
                }
            }
        });
}
