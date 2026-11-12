# Docking host

The `smileberta-dock` service runs on a dedicated VPS rather than Render. Vina is CPU-bound
and parallelises well, so cores-per-dollar dominates: Render's free tier gives ~0.1 shared
vCPU (≈30 min per dock), while a €16/mo 4-core box lands around 30–45 s.

`smileberta-api` and `smileberta-web` stay on Render.

## First-time setup

1. Create a Hetzner Cloud server: **CPX31** (4 dedicated vCPU, 8 GB), **Ubuntu 24.04**,
   and attach your SSH key at creation time.
2. SSH in and run:

   ```bash
   apt-get update -qq && apt-get install -y -qq git
   git clone --depth 1 https://github.com/NisargRhino/SmileBERTa-portal.git /opt/smileberta
   export DOCK_DOMAIN=<your-hostname>
   bash /opt/smileberta/deploy/provision.sh
   ```

   `provision.sh` installs Docker, locks the firewall down to SSH/HTTP/HTTPS, enables
   automatic security updates, builds the image and starts everything behind Caddy.
   It is safe to re-run.

3. Verify: `curl https://$DOCK_DOMAIN/health` should report `"docking": true` and a
   `queue` block.

4. Point the frontend at it by updating `PROD_DOCK_API` in `frontend/assets/js/core.js`.
   To test before committing, set `localStorage.sb_dock_api` in the browser console.

### Hostname

HTTPS is mandatory: the site is served over HTTPS and browsers block mixed content, so a
plain-HTTP docking API will not work. Caddy obtains and renews a Let's Encrypt certificate
automatically, which needs a real hostname resolving to the server.

Without a custom domain, use [sslip.io](https://sslip.io): `<ip-with-dashes>.sslip.io`
resolves to that IP with no DNS setup. For `203.0.113.5`, `DOCK_DOMAIN=203-0-113-5.sslip.io`.
Swap in a real subdomain later by editing `deploy/.env` and re-running `provision.sh`.

## Deploying changes

```bash
ssh root@<server>
bash /opt/smileberta/deploy/redeploy.sh
```

This pulls `origin/main` and rebuilds, so **push your changes first** — the server builds
from GitHub, not from your laptop.

## Tuning

Set in `deploy/.env`. The rule of thumb is `DOCK_CONCURRENCY × VINA_CPU = vCPU count`.
Running two jobs at two cores each beats one job on four, because one job's
single-threaded prep (Open Babel, ETKDG) overlaps the other's Vina search.

| Variable | Default | Meaning |
|---|---|---|
| `DOCK_CONCURRENCY` | 2 | Docking jobs running at once |
| `VINA_CPU` | 2 | Cores per Vina process |
| `VINA_TIMEOUT` | 900 | Seconds before a job is killed |
| `VINA_MEM_MB` | 2048 | Hard address-space cap on the Vina child |
| `MAX_JOBS_PER_USER` | 3 | Running + queued jobs per signed-in user |
| `RECEPTOR_CACHE_SIZE` | 24 | Prepared receptors cached in memory |
| `DOCK_NICE` | 0 | Vina nice level; **leave at 0** on dedicated cores |

## Operations

```bash
cd /opt/smileberta/deploy
docker compose ps
docker compose logs -f dock
curl -s localhost:10000/health | python3 -m json.tool
```

## Known gaps

- **Job state is in memory.** A restart loses running jobs. Acceptable because the VPS does
  not sleep (unlike Render free), but a Postgres-backed queue is the fix if it becomes a
  problem. Keep gunicorn at one worker until then.
- **Anonymous users are not rate-limited.** `MAX_JOBS_PER_USER` keys on the account id,
  which is absent for signed-out users, so they share no limit. Needs either IP-based
  limiting or requiring sign-in to dock.
