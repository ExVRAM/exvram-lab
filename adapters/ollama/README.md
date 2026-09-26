# Ollama adapter boundary

This adapter uses the official `ollama` CLI as an optional local-runtime baseline. ExVRAM first
checks `ollama list` and runs only models that are already installed; it never downloads a model
as a side effect of an experiment. Model licenses remain separate from the MIT-licensed Ollama
source.
