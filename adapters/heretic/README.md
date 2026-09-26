# Heretic adapter boundary

Heretic is an external research tool for directional abliteration. ExVRAM calls it only through
the upstream CLI (`heretic` / `heretic-llm`). This repository does not vendor Heretic, Optuna,
or the ablation optimizer.

The code license checked at `p-e-w/heretic` commit `3521f8648a0dccf6e12a92666862632235fac7e6`
is AGPL-3.0. That is not compatible with copying into this Apache-2.0 tree. Run it as a
separate process. `exvram plan-heretic` prints the argv and does not execute it.

`plan-heretic --evaluate-model` is the comparison entry point. It does not create a new
checkpoint. A full abliteration run is optional and stays outside ExVRAM until a ready
minimal-refusal checkpoint fails the measured quality or architecture gate.
