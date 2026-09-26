"""Run one measured text-only-input P2 point through an installed ExLlamaV3 runtime.

This deliberately lives at the integration boundary: ExLlamaV3 remains an optional external
dependency and no runtime source is copied into ExVRAM Lab.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from exvram.storage import StorageSafetyError, assert_storage_safe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--gpu-split", default="7.0")
    parser.add_argument("--cache-quant", default=None)
    parser.add_argument(
        "--search-candidate-id",
        default=None,
        help="exact candidate identity from experiments/search/*.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--allow-removable-storage",
        action="store_true",
        help="allow model/runtime/output paths on a removable volume",
    )
    return parser


def _write(payload: dict[str, Any], output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _record_base(args: argparse.Namespace) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "experiment_id": f"p2-exllamav3-exl3-4bpw-c{args.context}",
        "phase": "P2",
        "benchmark_kind": "full_model_shootout",
        "backend": "exllamav3",
        "model_id": "Qwen3.8-27B",
        "artifact_ref": "thelastspark/Qwen3.8-27B-exl3@4.00bpw",
        "synthetic": False,
        "measurement_status": "planned",
        "status": "PLANNED",
        "context_tokens": args.context,
        "batch_size": 1,
        "text_only_input": True,
        "vision_input": False,
        "mtp_enabled": False,
        "runtime_config": {
            "gpu_split_gib": args.gpu_split,
            "cache_quant": args.cache_quant,
            "decode_tokens": args.decode_tokens,
        },
        "notes": [
            "Real runtime measurement; no RTX 5060 result is inferred from model cards.",
            "Text-only input excludes image tokens but does not relabel the native checkpoint.",
        ],
    }
    if args.search_candidate_id:
        record["search_candidate_id"] = args.search_candidate_id
    return record


def _prompt_ids(tokenizer: Any, context: int):
    import torch

    seed_text = (
        "ExVRAM Lab measures reproducible local language model inference. "
        "Report the next token accurately and continue the plain text benchmark. "
    )
    seed_ids = tokenizer.encode(seed_text, add_bos=False).flatten()
    repeats = (context + int(seed_ids.shape[-1]) - 1) // int(seed_ids.shape[-1])
    return seed_ids.repeat(repeats)[:context].unsqueeze(0).to(dtype=torch.long)


def run(args: argparse.Namespace) -> dict[str, Any]:
    record = _record_base(args)
    try:
        assert_storage_safe(
            [args.model_dir, args.output],
            allow_removable=args.allow_removable_storage,
        )
        import torch
        from exllamav3 import Generator, Job
        from exllamav3.model_init import add_args, init

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected ExLlamaV3 environment")

        cache_tokens = max(256, ((args.context + 255) // 256) * 256)
        chunk_tokens = max(256, min(2048, cache_tokens))
        device = torch.cuda.current_device()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        parser = argparse.ArgumentParser(add_help=False)
        add_args(
            parser,
            cache=True,
            default_cache_size=cache_tokens,
            default_recurrent_cache_size=4.0,
            default_autosplit_max_batch_size=1,
            default_chunk_size=chunk_tokens,
        )
        init_argv = [
            "--model_dir",
            args.model_dir,
            "--gpu_split",
            args.gpu_split,
            "--cache_size",
            str(cache_tokens),
            "--autosplit_max_batch_size",
            "1",
            "--chunk_size",
            str(chunk_tokens),
        ]
        if args.cache_quant:
            init_argv.extend(["--cache_quant", args.cache_quant])
        init_args = parser.parse_args(init_argv)
        load_start = time.perf_counter()
        model, config, cache, tokenizer = init(
            init_args,
            quiet=True,
            progress=False,
            override_dynamic_seq_len=cache_tokens,
        )
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - load_start
        load_peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30
        load_peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)

        ids = _prompt_ids(tokenizer, args.context)
        generator = Generator(
            model,
            cache,
            tokenizer,
            max_batch_size=1,
            max_chunk_size=chunk_tokens,
            recurrent_cache_size=4 * 1024**3,
        )
        job = Job(
            input_ids=ids,
            max_new_tokens=args.decode_tokens,
            seed=args.seed,
            decode_special_tokens=False,
        )
        generator.enqueue(job)
        run_start = time.perf_counter()
        final: dict[str, Any] | None = None
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if result.get("stage") == "error":
                    raise result["error"]
                if result.get("eos"):
                    final = result
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - run_start
        if final is None:
            raise RuntimeError("ExLlamaV3 generator ended without a final result")

        prompt_tokens = int(ids.shape[-1])
        new_tokens = int(final.get("new_tokens", args.decode_tokens))
        prefill_seconds = float(final.get("time_prefill", 0.0))
        decode_seconds = float(final.get("time_generate", 0.0))
        model_bytes = sum(
            path.stat().st_size for path in Path(args.model_dir).glob("model-*.safetensors")
        )
        record.update(
            {
                "measurement_status": "measured",
                "status": "PASS",
                "runtime_version": "exllamav3-1.5.1",
                "architecture": getattr(config, "architecture", None),
                "component": getattr(model, "component", "text"),
                "layer_count": getattr(config, "num_hidden_layers", None),
                "model_capabilities": getattr(model, "caps", {}),
                "metrics": {
                    "load_seconds": load_seconds,
                    "cache_tokens": cache_tokens,
                    "model_file_bytes": model_bytes,
                    "artifact_file_bpw_estimate": model_bytes * 8.0 / 27_000_000_000.0,
                    "wall_seconds": wall_seconds,
                    "prompt_tokens": prompt_tokens,
                    "new_tokens": new_tokens,
                    "ttft_ms": prefill_seconds * 1000.0,
                    "prefill_tokens_per_second": (
                        prompt_tokens / prefill_seconds if prefill_seconds > 0 else None
                    ),
                    "decode_tokens_per_second": (
                        new_tokens / decode_seconds if decode_seconds > 0 else None
                    ),
                    "gpu_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                    "gpu_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
                    "gpu_peak_load_allocated_gib": load_peak_allocated,
                    "gpu_peak_load_reserved_gib": load_peak_reserved,
                    "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                    "cuda_device": torch.cuda.get_device_name(device),
                    "compute_capability": list(torch.cuda.get_device_capability(device)),
                },
                "quality_gate": {
                    "status": "NOT_RUN",
                    "reason": "P2 throughput smoke only; quality suite is a separate gate.",
                },
            }
        )
        return record
    except StorageSafetyError as exc:
        record.update(
            {
                "measurement_status": "blocked",
                "status": "INCONCLUSIVE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "quality_gate": {
                    "status": "NOT_RUN",
                    "reason": "storage safety policy blocked external runtime access",
                },
            }
        )
        return record
    except Exception as exc:  # a failed experiment is still a useful machine-readable result
        record.update(
            {
                "measurement_status": "failed",
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "quality_gate": {
                    "status": "NOT_RUN",
                    "reason": "runtime did not produce a valid full-model result",
                },
            }
        )
        return record


def main() -> int:
    args = _parser().parse_args()
    payload = run(args)
    _write(payload, args.output)
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
