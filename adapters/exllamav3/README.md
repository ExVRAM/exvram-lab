# ExLlamaV3 adapter boundary

This directory is reserved for an adapter to the separately installed ExLlamaV3 runtime. No
ExLlamaV3 source is copied here. Keep version, commit, install method, and runtime telemetry in
experiment results when implementing the adapter.

The thin local probe targets the upstream `exllamav3.modules.linear.Linear` entrypoint. It does
not construct a fake layer: EXL3 layer tests require a real packed checkpoint and qmap.
