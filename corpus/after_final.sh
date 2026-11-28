#!/usr/bin/env bash
# Runs once the forward run finishes: the reverse direction, then figures.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-/private/tmp/claude-501/-Users-nisargshah-Documents-cs-SmileBERTa-portal/d5665c0f-0ea0-405a-b676-e25fb81ddc4d/scratchpad/tenv/bin/python}"
RELEASE="${RELEASE:-corpus/v1}"

while pgrep -f 'final_run\.sh' > /dev/null; do sleep 60; done
echo "=== [$(date '+%H:%M:%S')] forward run finished, starting reverse"

PYTHONPATH=. "$PY" -u -m pharmacophore.reverse --release "$RELEASE"
echo "=== [$(date '+%H:%M:%S')] reverse done"

PYTHONPATH=. "$PY" -u -m pharmacophore.figures --release "$RELEASE"
echo "=== [$(date '+%H:%M:%S')] figures done"

git add "$RELEASE"/results "$RELEASE"/figures "$RELEASE"/splits 2>/dev/null
git diff --cached --quiet 2>/dev/null || git commit -q -m "Reverse direction results and refreshed figures

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
echo "=== [$(date '+%H:%M:%S')] all done"
