#!/usr/bin/env bash
# Pull the latest main and restart the docking service.
set -euo pipefail
APP="${APP:-/opt/smileberta}"
# Default to main, NOT the checked-out branch. Deriving it from HEAD meant that
# once the host had been provisioned from a feature branch it kept re-pulling
# that stale branch forever, and every later redeploy silently did nothing.
BRANCH="${BRANCH:-main}"

git -C "$APP" fetch --depth 1 origin "$BRANCH"
git -C "$APP" reset --hard FETCH_HEAD

cd "$APP/deploy"
docker compose up -d --build
docker image prune -f >/dev/null

echo
docker compose ps
echo
curl -fsS --max-time 10 http://localhost:10000/health 2>/dev/null | sed 's/^/    /' || true
