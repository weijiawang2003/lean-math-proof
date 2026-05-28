#!/bin/bash
# Launch wave 3 — adaptive mutations around best followup variants.
set -e

ROOT=/Users/weijiawang/dev/dojo_sandbox
SCOREBOARD=$(ls -t $ROOT/project/evolve/autonomous_runs/v5-followup-*/scoreboard.jsonl 2>/dev/null | head -1)

if [ -z "$SCOREBOARD" ]; then
  echo "no followup scoreboard found"
  exit 1
fi

echo "launching wave 3 seeded from: $SCOREBOARD"

cd $ROOT
python3 -m evolve.autonomous_research_wave3 \
    --seed-scoreboard "$SCOREBOARD" \
    --theorem-set nat_defs_medium \
    --ckpt-dir project/models/gen_v5 \
    --max-hours 1.5 \
    --num-children 3 2>&1 | tee project/evolve/autonomous_runs/_wave3_console.log
