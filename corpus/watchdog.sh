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
# Six workers once produced three SIGBUS kills in ten minutes with nothing else
# running, because the swap file cannot grow while the disk sits at ~97% full.
# Dropping to four was the stopgap; the actual fix was trimming the receptor to
# the ligand neighbourhood before ProLIF converts it, which cut 55-80% of the
# atoms and with them the allocation that was being killed. Six is back because
# each worker now needs a fraction of what it did. If the SIGBUS kills return,
# drop this to 4 and then 2 -- and look at free disk first, since that is the
# constraint underneath all of it.
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
