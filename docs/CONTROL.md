# ExVRAM Control

ExVRAM Control is a local Windows panel for the published llama.cpp recipe.
The lab stays in Python. The panel does not run benchmarks, does not quantize,
and does not implement inference.

## Install and run

Requirements: Rust 1.96 or newer, a Windows 11 machine with `nvidia-smi`, and
an existing `llama-server.exe`. The panel looks for that binary under the
repo's `.cache/llama/` directory and also accepts an explicit file choice.
Python is not required to open the panel.

```text
cd control
cargo run -p exvram-control
```

Closing the window hides it. The tray menu has Открыть and Выход. Выход stops
only the llama-server process this panel started. A server started by the P9
runner is not adopted and is not killed.

Settings are written to `%LOCALAPPDATA%\ExVRAM\control\settings.json`. Export
and import copy that JSON. It is not a benchmark result and is not committed.

## Architecture

`control/core` decides. It parses nvidia-smi text, reads
`recipes/rtx5060_8gb_qwen38_27b_8k.yaml`, estimates memory from the published
occupied-8k numbers, and builds the llama-server argument list.

`control/src-tauri` is the window, tray, process supervisor, and telemetry
poll. The page is static HTML. There is no second inference runtime.

The server is bound to `127.0.0.1` only. The interface cannot pass a shell
command. Paths on `F:` and on removable drives are rejected.

## What the numbers mean

The measured point is q4_0 KV, configured context 8448, occupied prompt 8192:
median decode 30.10 tok/s, KV 148.5 MiB, peak 7767 MiB. That speed is shown
only for a file whose name and byte size match the published IQ2_XXS GGUF.

q8_0 at that occupancy has a measured size (280.5 MiB) and an unstable speed,
so it is not recommended. f16 at that occupancy was not completed. A 512 MiB
f16 figure from the older configured-8192 table is scaled and marked modeled.
Anything that is not the measured point is marked modeled, or shown as
«Нет данных». TTFT during a live session is «Нет данных»: the panel does not
issue a hidden generation just to time it.

Quality gate status remains NOT_RUN. Switching KV is not described as a
quality change. Balanced, Performance, and Quality all stay on the measured
q4 recipe when it fits. Quality does not pick a heavier KV.

## Protect My PC

On by default. The panel uses 4 threads, below-normal priority, and direct
I/O. It refuses to start when another program already holds more than 2048 MiB
of VRAM, when the estimate does not leave 256 MiB on the card, when the GGUF
is larger than physical RAM, when less than 1.5 GiB of RAM is free, or when
Windows commit headroom is under 2 GiB. It does not close other programs and
does not change the pagefile. Turning Protect My PC off skips the RAM and
commit floors; the file-larger-than-RAM check stays.

Context and KV changes apply on the next start. The running server is left
alone until Stop.

## Not in this build

ExLlamaV3, TurboQuant, KVQuant, KIVI, SnapKV, PyramidKV, and LLMLingua are not
wired up. The context-optimization control has one working position: off.
AUTO and EXPERIMENTAL are not selectable.

There is no account, no cloud telemetry, and no paid feature.

## Panel overhead

One debug build, window open, no model loaded: the process plus its WebView2
children used about 170 MiB of working set. Over a 4 second sample the parent
used 0.016 s of CPU. The poll is 2 s while the window is visible and 8 s after
it hides to the tray. This is not a release-build figure and not a tray figure.

