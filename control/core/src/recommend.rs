//! Applies the published recipe. This is not a new search over unmeasured configs.

use crate::estimate::{estimate, launch_gate, KvFormat, MEASURED_CONTEXT, MEASURED_DECODE_TOK_S};

#[derive(Debug, Clone, PartialEq)]
pub struct Recommendation {
    pub context_tokens: u32,
    pub kv: KvFormat,
    pub memory_basis: &'static str,
    pub memory_applies: bool,
    pub expected_total_mib: f64,
    pub speed_note: String,
    pub speed_is_measured: bool,
    pub quality_risk: String,
    pub explanation: String,
    pub restart_required: bool,
}

pub fn recommend(
    total_vram_mib: u32,
    free_vram_mib: u32,
    protect_pc: bool,
    mode: &str,
    running_context: Option<u32>,
    running_kv: Option<KvFormat>,
    recipe_applies: bool,
) -> Recommendation {
    let kv = KvFormat::Q4;
    let mut context = MEASURED_CONTEXT;
    if !launch_gate(
        total_vram_mib,
        free_vram_mib,
        estimate(context, kv).total_mib,
        protect_pc,
    )
    .allowed
    {
        context = [8192, 4096]
            .into_iter()
            .find(|candidate| {
                launch_gate(
                    total_vram_mib,
                    free_vram_mib,
                    estimate(*candidate, kv).total_mib,
                    protect_pc,
                )
                .allowed
            })
            .unwrap_or(4096);
    }
    let memory = estimate(context, kv);
    let exact = recipe_applies && context == MEASURED_CONTEXT && kv == KvFormat::Q4;
    let speed_note = if !recipe_applies {
        "Нет данных".into()
    } else if exact {
        format!(
            "Измерено для занятого контекста 8192 и q4_0 KV: медиана decode {MEASURED_DECODE_TOK_S:.2} tok/s. Другие окна не измерялись этим числом."
        )
    } else {
        "Нет данных".into()
    };
    let quality_note = match mode {
        "quality" => {
            "Режим качества не меняет KV: влияние q8/f16 на интеллект не измерено, quality gate = NOT_RUN. Остаётся проверенный q4_0."
        }
        "performance" => {
            "Режим скорости оставляет q4_0: это единственный KV, у которого на занятых 8192 измерена и скорость, и полный GPU."
        }
        _ => "Баланс — опубликованный рецепт q4_0. Смена KV не заявлена как улучшение качества.",
    };
    Recommendation {
        context_tokens: context,
        kv,
        memory_basis: if recipe_applies { memory.basis } else { "unavailable" },
        memory_applies: recipe_applies,
        expected_total_mib: memory.total_mib,
        speed_is_measured: exact,
        speed_note,
        quality_risk: quality_note.into(),
        explanation: if recipe_applies {
            "Рекомендуется q4 KV: текущий проверенный рецепт держит веса на GPU и тратит около 148.5 МиБ на KV при окне 8448.".into()
        } else {
            "Проверенный рецепт относится только к Qwen3.8-27B UD-IQ2_XXS. Для выбранного файла скорость и VRAM не подставляются.".into()
        },
        restart_required: running_context.is_some_and(|value| value != context)
            || running_kv.is_some_and(|value| value != kv),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_gigabyte_card_gets_the_measured_recipe() {
        let recommendation = recommend(8151, 7000, true, "balanced", None, None, true);
        assert_eq!(recommendation.context_tokens, 8448);
        assert_eq!(recommendation.kv, KvFormat::Q4);
        assert!(recommendation.speed_is_measured);
        assert!(recommendation.speed_note.contains("30.10"));
    }

    #[test]
    fn quality_mode_does_not_invent_a_better_kv() {
        let recommendation = recommend(8151, 7000, true, "quality", Some(4096), Some(KvFormat::Q8), true);
        assert_eq!(recommendation.kv, KvFormat::Q4);
        assert!(recommendation.restart_required);
        assert!(recommendation.quality_risk.contains("NOT_RUN"));
    }

    #[test]
    fn a_tiny_card_does_not_claim_the_measured_speed() {
        let recommendation = recommend(4096, 4096, true, "performance", None, None, true);
        assert!(!recommendation.speed_is_measured);
        assert_eq!(recommendation.speed_note, "Нет данных");
    }

    #[test]
    fn another_file_does_not_inherit_the_measured_speed() {
        let recommendation = recommend(8151, 7000, true, "balanced", None, None, false);
        assert!(!recommendation.speed_is_measured);
        assert!(recommendation.explanation.contains("UD-IQ2_XXS"));
    }
}
