# Planned measurement curves

The following files are machine-readable plans, not results:

- [`experiments/residency/offload_curve.json`](../experiments/residency/offload_curve.json)
  measures CPU-resident weight MiB against decode speed and PCIe traffic.
- [`experiments/residency/vram_pressure.json`](../experiments/residency/vram_pressure.json)
  finds a practical Windows VRAM safety reserve.
- [`experiments/kv_cache/weight_kv_tradeoff.json`](../experiments/kv_cache/weight_kv_tradeoff.json)
  compares spending the last memory budget on weights versus KV precision.

All three are `PLANNED`. They must be run one configuration at a time with stable model/runtime
storage, bounded timeouts and a stop condition for system lag. A plan entry never supplies a
throughput value by itself.
