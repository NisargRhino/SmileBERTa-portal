#!/usr/bin/env bash
# Keep the corpus build alive overnight.
#
# The build already supervises its own batches and kills any that overrun, so
# the remaining failure mode is the supervisor itself dying - an unhandled
# exception, an OOM kill, a dropped connection it did not catch. This loop
# notices and restarts it. Progress lives in the records on disk, so a restart
# resumes rather than repeats.
set -u
RELEASE="${RELEASE:-corpus/v1}"
PY="${PY:-/private/tmp/claude-501/-Users-nisargshah-Documents-cs-SmileBERTa-portal/d5665c0f-0ea0-405a-b676-e25fb81ddc4d/scratchpad/tenv/bin/python}"
WORKERS="${WORKERS:-6}"
MAX_RESTARTS="${MAX_RESTARTS:-200}"
cd "$(dirname "$0")/.."

for attempt in $(seq 1 "$MAX_RESTARTS"); do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] build attempt $attempt ==="
  PYTHONPATH=. "$PY" -u -m pharmacophore.build --release "$RELEASE" --workers "$WORKERS"
  status=$?
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] exited with status $status ==="
  if [ "$status" -eq 0 ]; then
    echo "build reported completion"
    break
  fi
  sleep 30
done
