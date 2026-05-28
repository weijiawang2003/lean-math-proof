#!/bin/bash
# Print a one-line summary per v5 followup variant — actual vs v5-00 baseline.
set -e

ROOT=/Users/weijiawang/dev/dojo_sandbox
BASE_METRICS=$(ls $ROOT/project/evolve/autonomous_runs/v5-auto-*/eval/v5-00-baseline-repro/eval-*/metrics.json | head -1)
FOLLOWUP_SCOREBOARD=$(ls $ROOT/project/evolve/autonomous_runs/v5-followup-*/scoreboard.jsonl 2>/dev/null | head -1)

if [ -z "$FOLLOWUP_SCOREBOARD" ]; then
  echo "no followup scoreboard yet"
  exit 0
fi

python3 - <<EOF
import json
base = json.load(open("$BASE_METRICS"))
base_proved = {t["full_name"] for t in base["per_theorem"] if t.get("finished")}
print(f"baseline proved: {len(base_proved)}/38")
for line in open("$FOLLOWUP_SCOREBOARD"):
    d = json.loads(line)
    # Read this candidate's per_theorem to compute true delta
    metrics_path = d["eval_dir"] + "/metrics.json"
    try:
        m = json.load(open(metrics_path))
        proved_set = {t["full_name"] for t in m["per_theorem"] if t.get("finished")}
    except FileNotFoundError:
        proved_set = set()
    new = sorted(proved_set - base_proved)
    lost = sorted(base_proved - proved_set)
    print(f"  {d['name']:<32} {d['proved']:>3}/38 new={new} lost={lost}")
EOF
