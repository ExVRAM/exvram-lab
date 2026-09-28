# Uncensored quality gate

Status: **NOT RUN**.

There is no Coletti quantized artifact and therefore no uncensored inference,
capability delta, harmful refusal rate or benign over-refusal rate. The prepared
scaffolding keeps `REFUSAL_RATE` and `OVERREFUSAL_RATE` separate and does not
infer refusal behaviour from model names or a missing response file.

The comparison must be matched:

`Base IQ2_XXS` versus `Coletti matching IQ2_XXS`

with identical tokenizer, prompt classes, decoding, context, KV format and
runtime. The quality suite must run before any speculative decoding experiment.
