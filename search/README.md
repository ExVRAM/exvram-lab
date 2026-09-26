# Search and optimizer notes

The current search is a bounded planner, not an unattended full-model runner. It generates a
cheap context-128 screen first, then expands only a capped set of deeper contexts and backend
profiles. A speed result is usable only when a result record carries the exact generated
`search_candidate_id`; related backend/context observations are shown as context, not reused as
proof for a different quant/KV/residency combination.

Run from the project root:

```text
exvram plan-search --config experiments/search/rtx5060_27b_search.json --stage adaptive
exvram recommend-config --config experiments/search/rtx5060_27b_search.json
```

Search policies cover weight bits, context, KV precision, residency and runtime profiles. Large
model execution remains an explicit external step through the existing runners; the planner does
not download models or start a brute-force GPU sweep.

For a real measurement, pass the exact `candidate_id` from the plan to the matching runner:

```text
python benchmark/run_llamacpp_p2.py ... --search-candidate-id <candidate_id>
python benchmark/run_exllama_p2.py ... --search-candidate-id <candidate_id>
```

The identifier is provenance metadata, not a runtime translation layer. The command-line options
used for the run must still match the candidate fields. Without the exact identifier, the result
is retained as related context but cannot unlock a measured recommendation for that candidate.
