//! Read llama-server JSON without inventing missing fields.

use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct SlotView {
    pub configured_context: Option<u32>,
    pub occupied_tokens: Option<u32>,
}

pub fn parse_slots(body: &str) -> Result<SlotView, String> {
    let value: Value = serde_json::from_str(body).map_err(|err| err.to_string())?;
    let slot = value
        .as_array()
        .and_then(|items| items.first())
        .or(Some(&value))
        .ok_or("empty slots")?;
    let configured = slot.get("n_ctx").and_then(Value::as_u64).map(|n| n as u32);
    let occupied = ["n_past", "num_ctx_tokens", "occupied_tokens"]
        .iter()
        .find_map(|key| slot.get(*key).and_then(Value::as_u64))
        .map(|n| n as u32);
    Ok(SlotView {
        configured_context: configured,
        occupied_tokens: occupied,
    })
}

pub fn parse_speed_metrics(body: &str) -> (Option<f64>, Option<f64>) {
    let mut prompt = None;
    let mut predicted = None;
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let name = parts.next().unwrap_or("");
        let value = parts.next().and_then(|token| token.parse::<f64>().ok());
        if name.starts_with("llamacpp:prompt_tokens_seconds")
            || name.starts_with("llamacpp_prompt_tokens_seconds")
        {
            prompt = value;
        }
        if name.starts_with("llamacpp:predicted_tokens_seconds")
            || name.starts_with("llamacpp_predicted_tokens_seconds")
            || name.starts_with("llamacpp:tokens_predicted_seconds")
        {
            predicted = value;
        }
    }
    (prompt, predicted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_keep_configured_and_occupied_apart() {
        let view = parse_slots(r#"[{"id":0,"n_ctx":8448,"n_past":120}]"#).unwrap();
        assert_eq!(view.configured_context, Some(8448));
        assert_eq!(view.occupied_tokens, Some(120));
    }

    #[test]
    fn missing_occupancy_stays_missing() {
        let view = parse_slots(r#"[{"n_ctx":4096}]"#).unwrap();
        assert_eq!(view.occupied_tokens, None);
    }

    #[test]
    fn metrics_ignore_comments() {
        let body = "# HELP llamacpp:prompt_tokens_seconds x\nllamacpp:prompt_tokens_seconds 33.6\nllamacpp:predicted_tokens_seconds 30.1\n";
        let (prompt, predicted) = parse_speed_metrics(body);
        assert_eq!(prompt, Some(33.6));
        assert_eq!(predicted, Some(30.1));
    }
}
