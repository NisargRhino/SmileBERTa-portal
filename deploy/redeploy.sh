#!/usr/bin/env bash
# Pull the latest main and restart the docking service.
set -euo pipefail
APP="${APP:-/opt/smileberta}"

git -C "$APP" fetch --depth 1 origin main
git -C "$APP" reset --hard origin/main

cd "$APP/deploy"
docker compose up -d --build
docker image prune -f >/dev/null

echo
docker compose ps
echo
curl -fsS --max-time 10 http://localhost:10000/health 2>/dev/null | sed 's/^/    /' || true
