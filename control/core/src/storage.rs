//! Path policy. Removable volumes and the operator F: disk are refused.

pub fn storage_allowed(path: &str, drive_kind: Option<u32>) -> Result<(), String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("путь пуст".into());
    }
    let upper = trimmed.to_ascii_uppercase();
    if upper.starts_with("F:") || upper.starts_with("F\\") {
        return Err("Диск F: не используется, даже если система считает его постоянным.".into());
    }
    // Win32 GetDriveType: 2 is removable.
    if drive_kind == Some(2) {
        return Err("Съёмный диск не используется для модели и llama-server.".into());
    }
    Ok(())
}

pub fn model_fits_physical_ram(file_bytes: u64, physical_bytes: u64) -> bool {
    physical_bytes > 0 && file_bytes <= physical_bytes
}

pub fn may_adopt_process(image_path: &str) -> bool {
    let name = image_path.rsplit(['\\', '/']).next().unwrap_or(image_path);
    name.eq_ignore_ascii_case("llama-server.exe") || name.eq_ignore_ascii_case("llama-server")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_drive_is_refused_even_when_windows_calls_it_fixed() {
        let err = storage_allowed(r"F:\models\x.gguf", Some(3)).unwrap_err();
        assert!(err.contains("F:"));
    }

    #[test]
    fn removable_drive_is_refused() {
        assert!(storage_allowed(r"D:\models\x.gguf", Some(2)).is_err());
        assert!(storage_allowed(r"E:\models\x.gguf", Some(3)).is_ok());
    }

    #[test]
    fn a_gguf_larger_than_ram_is_refused() {
        assert!(!model_fits_physical_ram(20_000_000_000, 16_000_000_000));
        assert!(model_fits_physical_ram(7_266_070_528, 17_000_000_000));
    }

    #[test]
    fn only_llama_server_can_be_adopted_after_a_crash() {
        assert!(may_adopt_process(r"E:\ExVRAM\.cache\llama\llama-server.exe"));
        assert!(!may_adopt_process(r"C:\Windows\System32\notepad.exe"));
    }
}
