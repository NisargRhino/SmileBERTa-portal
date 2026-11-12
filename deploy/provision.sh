#!/usr/bin/env bash
# Turn a bare Ubuntu 24.04 server into the SmileBERTa docking host.
# Safe to re-run: every step is idempotent.
#
#   export DOCK_DOMAIN=dock.example.com
#   bash provision.sh
#
set -euo pipefail

REPO="${REPO:-https://github.com/NisargRhino/SmileBERTa-portal.git}"
APP="${APP:-/opt/smileberta}"
: "${DOCK_DOMAIN:?Set DOCK_DOMAIN first, e.g. export DOCK_DOMAIN=dock.example.com}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

if [ "$(id -u)" -ne 0 ]; then
	echo "Run this as root (or with sudo)." >&2
	exit 1
fi

say "Checking that $DOCK_DOMAIN points at this server"
server_ip="$(curl -fsS --max-time 10 https://api.ipify.org || true)"
domain_ip="$(getent hosts "$DOCK_DOMAIN" | awk '{print $1; exit}' || true)"
if [ -z "$domain_ip" ]; then
	echo "WARNING: $DOCK_DOMAIN does not resolve yet. Certificate issuance will fail" >&2
	echo "         until its DNS A record points to this server: ${server_ip:-unknown}" >&2
elif [ -n "$server_ip" ] && [ "$domain_ip" != "$server_ip" ]; then
	echo "WARNING: $DOCK_DOMAIN resolves to $domain_ip but this server is $server_ip." >&2
	echo "         Fix the DNS A record, or the certificate will be refused." >&2
else
	echo "OK: $DOCK_DOMAIN -> $domain_ip"
fi

say "Installing base packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl git ufw unattended-upgrades >/dev/null

say "Installing Docker"
if ! command -v docker >/dev/null 2>&1; then
	curl -fsSL https://get.docker.com | sh
else
	echo "Docker already present: $(docker --version)"
fi

say "Configuring firewall (SSH, HTTP, HTTPS only)"
ufw allow OpenSSH >/dev/null
ufw allow 80/tcp >/dev/null
ufw allow 443/tcp >/dev/null
ufw --force enable >/dev/null
ufw status verbose | sed 's/^/    /'

say "Enabling automatic security updates"
dpkg-reconfigure -f noninteractive unattended-upgrades >/dev/null 2>&1 || true

say "Fetching the application into $APP"
if [ -d "$APP/.git" ]; then
	git -C "$APP" fetch --depth 1 origin main
	git -C "$APP" reset --hard origin/main
else
	git clone --depth 1 "$REPO" "$APP"
fi

say "Writing deploy/.env"
cat >"$APP/deploy/.env" <<EOF
DOCK_DOMAIN=$DOCK_DOMAIN
# Match these to the server's core count: concurrency x cpu should equal vCPUs.
VINA_CPU=${VINA_CPU:-2}
DOCK_CONCURRENCY=${DOCK_CONCURRENCY:-2}
EOF
chmod 600 "$APP/deploy/.env"
cat "$APP/deploy/.env" | sed 's/^/    /'

say "Building and starting (first build takes a few minutes)"
cd "$APP/deploy"
docker compose up -d --build

say "Waiting for the service to report healthy"
for i in $(seq 1 60); do
	if curl -fsS --max-time 5 "http://localhost:80/health" -H "Host: $DOCK_DOMAIN" >/dev/null 2>&1; then
		break
	fi
	sleep 5
done

say "Status"
docker compose ps
echo
echo "Local health check:"
curl -fsS --max-time 10 "http://localhost:80/health" -H "Host: $DOCK_DOMAIN" | sed 's/^/    /' || \
	echo "    not responding yet - check: docker compose -f $APP/deploy/docker-compose.yml logs"
echo
say "Done. Public endpoint: https://$DOCK_DOMAIN/health"
echo "    Redeploy after a git push:  bash $APP/deploy/redeploy.sh"
echo "    Follow logs:                docker compose -f $APP/deploy/docker-compose.yml logs -f"
