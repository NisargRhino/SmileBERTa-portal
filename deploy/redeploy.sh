#!/usr/bin/env bash
# Pull the latest main and restart the docking service.
set -euo pipefail
APP="${APP:-/opt/smileberta}"
BRANCH="${BRANCH:-$(git -C "$APP" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"

git -C "$APP" fetch --depth 1 origin "$BRANCH"
git -C "$APP" reset --hard FETCH_HEAD

cd "$APP/deploy"
docker compose up -d --build
docker image prune -f >/dev/null

echo
docker compose ps
echo
curl -fsS --max-time 10 http://localhost:10000/health 2>/dev/null | sed 's/^/    /' || true
