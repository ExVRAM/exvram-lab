//! Read the published recipe YAML. Unknown fields are ignored.

use serde_yaml::Value;

use crate::estimate::MEASURED_CONTEXT;

#[derive(Debug, Clone, PartialEq)]
pub struct PublishedRecipe {
    pub name: String,
    pub model_file: String,
    pub quant: String,
    pub sha256: String,
    pub configured_context: u32,
    pub kv: String,
    pub median_decode_tok_s: Option<f64>,
    pub quality_gate: String,
}

pub const PUBLISHED_MODEL_FILE: &str = "Qwen3.8-27B-UD-IQ2_XXS.gguf";
pub const PUBLISHED_BYTES: u64 = 7_266_070_528;

pub fn identity_matches(file_name: &str, bytes: u64, recipe: Option<&PublishedRecipe>) -> bool {
    let name_ok = file_name == PUBLISHED_MODEL_FILE
        && recipe
            .map(|item| item.model_file == file_name)
            .unwrap_or(true);
    let hash_ok = recipe
        .map(|item| item.sha256.starts_with("e792d8fb"))
        .unwrap_or(true);
    name_ok && hash_ok && bytes == PUBLISHED_BYTES
}

pub fn parse_recipe_yaml(text: &str) -> Result<PublishedRecipe, String> {
    let value: Value = serde_yaml::from_str(text).map_err(|err| err.to_string())?;
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("unnamed")
        .to_string();
    let model = value.get("model").cloned().unwrap_or(Value::Null);
    let runtime = value.get("runtime").cloned().unwrap_or(Value::Null);
    let measurement = value.get("measurement").cloned().unwrap_or(Value::Null);
    Ok(PublishedRecipe {
        name,
        model_file: model
            .get("file")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        quant: model
            .get("quant")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        sha256: model
            .get("sha256")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        configured_context: runtime
            .get("configured_context_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(MEASURED_CONTEXT as u64) as u32,
        kv: runtime
            .get("cache_type_k")
            .and_then(Value::as_str)
            .unwrap_or("q4_0")
            .to_string(),
        median_decode_tok_s: measurement
            .get("median_decode_tok_s")
            .and_then(Value::as_f64),
        quality_gate: measurement
            .get("quality_gate")
            .and_then(Value::as_str)
            .unwrap_or("NOT_RUN")
            .to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn published_8k_recipe_matches_the_file_in_this_repo() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../recipes/rtx5060_8gb_qwen38_27b_8k.yaml");
        let text = std::fs::read_to_string(&path).expect("recipe file");
        let recipe = parse_recipe_yaml(&text).unwrap();
        assert_eq!(recipe.configured_context, 8448);
        assert_eq!(recipe.kv, "q4_0");
        assert_eq!(recipe.quant, "UD-IQ2_XXS");
        assert_eq!(recipe.quality_gate, "NOT_RUN");
        assert!(recipe.median_decode_tok_s.unwrap() > 30.0);
        assert!(recipe.sha256.starts_with("e792d8fb"));
        assert!(identity_matches(
            &recipe.model_file,
            PUBLISHED_BYTES,
            Some(&recipe)
        ));
        assert!(!identity_matches("other.gguf", PUBLISHED_BYTES, Some(&recipe)));
    }
}
