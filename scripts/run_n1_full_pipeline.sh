#!/usr/bin/env bash
set -euo pipefail

generation_pid="${1:?generation PID is required}"
expected_candidates="${2:-3352}"

while kill -0 "$generation_pid" 2>/dev/null; do
  sleep 30
done

generated_file=/tmp/nq_scenario_answers_v3_all.jsonl
generated_count=$(wc -l < "$generated_file")
if [[ "$generated_count" -ne "$expected_candidates" ]]; then
  echo "generation incomplete: ${generated_count}/${expected_candidates}" >&2
  exit 1
fi

python3 /tmp/build_nq_scenario_expansion_final.py validate \
  --generated "$generated_file" \
  --checkpoint /tmp/nq_scenario_baselines_v3_all.jsonl \
  --output-dir /tmp/nq-expanded-v3-validated

python3 /tmp/run_n1_generalization.py \
  --scenarios /tmp/nq_scenario_baselines_v3_all.jsonl \
  --output /tmp/nq-n1-full-result.json \
  --count "$expected_candidates" \
  --candidate-multiplier 2 \
  --passage-word-count 60 \
  --temperature 0.4
