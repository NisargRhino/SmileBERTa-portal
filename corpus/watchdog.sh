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
# Four, not six. Six workers alone -- with the analysis loop stopped and the
# machine otherwise idle -- still produced three SIGBUS kills in ten minutes,
# because the swap file cannot grow while the disk sits at ~97% full and a page
# that cannot be backed kills the process. Each kill costs its batch's in-flight
# entries, so the extra workers were not buying the throughput they appeared to.
WORKERS="${WORKERS:-4}"
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
