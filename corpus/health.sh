#!/usr/bin/env bash
# Append a progress snapshot every few minutes, so a stall is visible after the
# fact rather than inferred. Records entry count, process count and rate.
set -u
RELEASE="${RELEASE:-corpus/v1}"
INTERVAL="${INTERVAL:-300}"
cd "$(dirname "$0")/.."
last=0
while true; do
  n=$(cat "$RELEASE"/records/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  procs=$(pgrep -f 'pharmacophore.build' | wc -l | tr -d ' ')
  delta=$((n - last))
  printf '[%s] records=%s procs=%s delta=%s' "$(date '+%H:%M:%S')" "$n" "$procs" "$delta"
  if [ "$delta" -eq 0 ] && [ "$n" -gt 0 ]; then printf '  <-- NO PROGRESS'; fi
  printf '\n'
  last=$n
  sleep "$INTERVAL"
done
