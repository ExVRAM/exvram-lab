//! nvidia-smi text parsing. Live process calls stay outside this module.

#[derive(Debug, Clone, PartialEq)]
pub struct GpuSample {
    pub name: String,
    pub total_mib: u32,
    pub used_mib: u32,
    pub free_mib: u32,
    pub temperature_c: Option<u32>,
    pub utilization_percent: Option<u32>,
}

pub fn parse_nvidia_smi_line(line: &str) -> Result<GpuSample, String> {
    let fields: Vec<String> = line.split(',').map(|item| item.trim().to_string()).collect();
    if fields.len() < 4 {
        return Err(format!("expected at least 4 nvidia-smi fields, got {}", fields.len()));
    }
    let total_mib = parse_mib(&fields[1])?;
    let used_mib = parse_mib(&fields[2])?;
    let free_mib = parse_mib(&fields[3])?;
    Ok(GpuSample {
        name: fields[0].clone(),
        total_mib,
        used_mib,
        free_mib,
        temperature_c: fields.get(4).and_then(|value| parse_optional_u32(value)),
        utilization_percent: fields.get(5).and_then(|value| parse_optional_u32(value)),
    })
}

fn parse_mib(value: &str) -> Result<u32, String> {
    if value.is_empty() || value.eq_ignore_ascii_case("[N/A]") || value == "N/A" {
        return Err("VRAM reading is N/A".into());
    }
    value
        .parse::<f64>()
        .map(|number| number.round() as u32)
        .map_err(|_| format!("not a VRAM number: {value}"))
}

fn parse_optional_u32(value: &str) -> Option<u32> {
    if value.is_empty() || value.eq_ignore_ascii_case("N/A") || value == "[N/A]" {
        return None;
    }
    value.parse::<f64>().ok().map(|number| number.round() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_live_style_row() {
        let sample = parse_nvidia_smi_line(
            "NVIDIA GeForce RTX 5060, 8151, 1264, 6887, 42, 3",
        )
        .unwrap();
        assert_eq!(sample.name, "NVIDIA GeForce RTX 5060");
        assert_eq!(sample.total_mib, 8151);
        assert_eq!(sample.used_mib, 1264);
        assert_eq!(sample.free_mib, 6887);
        assert_eq!(sample.temperature_c, Some(42));
        assert_eq!(sample.utilization_percent, Some(3));
    }

    #[test]
    fn missing_temperature_is_not_a_fake_zero() {
        let sample = parse_nvidia_smi_line("GPU, 8000, 100, 7900, N/A, [N/A]").unwrap();
        assert_eq!(sample.temperature_c, None);
        assert_eq!(sample.utilization_percent, None);
    }

    #[test]
    fn rejects_a_short_row() {
        assert!(parse_nvidia_smi_line("only-name").is_err());
    }
}
