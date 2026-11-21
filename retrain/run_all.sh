#!/usr/bin/env bash
# retrain-v1: full pipeline. Settings: config.yaml (retrain section) and f1tenth_sim/config.yaml.
#
#   bash retrain/run_all.sh                 # all steps: tracks collect train export
#   bash retrain/run_all.sh train export    # only some steps
#
# Steps:
#   tracks  : retrain/make_tracks.py  -> f1tenth_racetracks/<Track>_w<cm>/, retrain_tracks_manifest.json
#   collect : retrain/collect.py      -> <retrain.dataset_dir>/ (refuses to overwrite)
#   train   : imitation/train.py --retrain -> checkpoint/bc_retrain_<version>.pt
#   export  : retrain/export.py --fp16     -> checkpoint/export/end2race_retrain_<version>_jit_fp32.pt / _fp16.pt
# Logs: retrain/logs/<step>_<time>.log
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export MPLBACKEND=Agg

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(tracks collect train export)
mkdir -p retrain/logs
stamp=$(date +%Y%m%d_%H%M)

for step in "${steps[@]}"; do
  log="retrain/logs/${step}_${stamp}.log"
  echo "=== ${step} (log: ${log})"
  case "$step" in
    tracks)  python retrain/make_tracks.py ;;
    collect) python retrain/collect.py ;;
    train)   python imitation/train.py --retrain ;;
    export)  python retrain/export.py --fp16 ;;
    *) echo "unknown step: $step"; exit 1 ;;
  esac 2>&1 | grep --line-buffered -v -e "inflated both" -e "non-interactive" -e "plt.show()" | tee "$log"
done
