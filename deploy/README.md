# Docking host

The docking service runs on a dedicated VPS, not on Render — the Render service that used to
host it was deleted on 2026-11-21. Vina is CPU-bound and parallelises well, so cores-per-dollar
dominates: Render's free tier gives ~0.1 shared vCPU (≈30 min per dock), while a €4.35/mo
2-core Hetzner box measures 3.7 s for benzamidine/trypsin and 35 s for imatinib/EGFR.

`smileberta-api` and `smileberta-web` stay on Render.

**Currently running:** a Hetzner **CPX11** (2 vCPU, 2 GB) at `5.161.192.134`, reached as
`https://5-161-192-134.sslip.io`. A CPX31 was specced originally, but the smaller box already
beat the target and was kept. Provisioning auto-sizes to the host, so resizing needs no code
change.

## First-time setup

1. Create a Hetzner Cloud server — **CPX11** (2 vCPU, 2 GB) is enough; take a bigger one only
   if you need more concurrency — on **Ubuntu 24.04**, attaching your SSH key at creation time.
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
ssh root@5.161.192.134 'BRANCH=main bash /opt/smileberta/deploy/redeploy.sh'
```

This pulls `origin/main` and rebuilds, so **push your changes first** — the server builds
from GitHub, not from your laptop.

**Always verify the deploy actually landed; do not trust the script's output.** It once
derived the branch from the host's checked-out `HEAD`, which had been provisioned from a
feature branch, so every redeploy silently re-pulled stale code, reported success and left
the old container running. It now defaults to `main`, but pass `BRANCH` explicitly and check:

```bash
ssh root@5.161.192.134 'docker exec deploy-dock-1 grep -c SOME_NEW_SYMBOL /app/docking.py'
```

The tell for a no-op deploy is `Container deploy-dock-1 Running` instead of `Recreated`,
plus a `CACHED` COPY layer in the build output.

## Tuning

Set in `deploy/.env`. The rule of thumb is `DOCK_CONCURRENCY × VINA_CPU = vCPU count`.
Running two jobs at two cores each beats one job on four, because one job's
single-threaded prep (Open Babel, ETKDG) overlaps the other's Vina search.

| Variable | Default | Meaning |
|---|---|---|
| `DOCK_CONCURRENCY` | 2 (set to **1** on the current CPX11) | Docking jobs running at once |
| `VINA_CPU` | 2 | Cores per Vina process |
| `VINA_TIMEOUT` | 900 | Seconds before a job is killed |
| `VINA_MEM_MB` | 2048 | Hard address-space cap on the Vina child |
| `MAX_JOBS_PER_USER` | 3 | Running + queued jobs per signed-in user |
| `RECEPTOR_CACHE_SIZE` | 24 | Prepared receptors cached in memory |
| `DOCK_NICE` | 0 | Vina nice level; **leave at 0** on dedicated cores |
| `REDOCK_WARM` | 1 | Validate the example structures once in the background after boot |
| `REDOCK_WARM_IDS` | 3PTB,1HSG,1M17,4EY7,6LU7 | Which structures to pre-validate |
| `DATABASE_URL` | *(unset)* | Postgres for the redocking-validation cache; see below |

### Redocking validation

`/dock/validate` redocks a structure's own co-crystallised ligand and reports how far the
top pose lands from the experimental one. Runs go through the same queue at **lower
priority** and cost **half** a rate-limit slot, so validation never delays real docking.

Results are cached by `pdb_id + comp_id + exhaustiveness + poses + vina version`. Set
`DATABASE_URL` in `deploy/.env` to the same Neon database `smileberta-api` uses and the
cache is shared and survives redeploys; leave it unset and the cache lives in the
container's memory, which means the warm-up runs again after every restart. Either way the
feature works — this only decides how often Vina repeats work it has already done.

Validation boxes are 18–26 Å per side. That is inside what the docking page already
lets anyone ask for (up to 40 Å), so it needs no change to `VINA_MEM_MB` or the container
`mem_limit` — but a 26 Å box does cost Vina noticeably more memory for its grid maps than
a typical 18 Å one, so keep an eye on `/health`'s `rss_mb` if you ever lower those limits.

`REDOCK_WARM` validates the five example structures from the docking page a minute after
boot, one at a time, so a first-time visitor gets an instant answer. 6LU7 is in that list
on purpose and is *expected* to be refused: its inhibitor is covalently bound to Cys145,
and redocking a covalent ligand with a non-covalent protocol would mean nothing. Watch it
with `docker compose logs -f dock | grep redock`, and check `/health`'s `redock_cache`.

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
- **Anonymous rate limiting depends on the proxy.** Signed-out callers are limited by client
  IP, read from the **last** `X-Forwarded-For` entry — the one Caddy appends, since the first
  is client-supplied and forgeable. If Caddy ever stopped sending that header, every caller
  would collapse onto the Docker gateway address and share one bucket, locking everyone out at
  the hourly cap. `/health` reports `"forwarded"` so this is visible; it is `true` in
  production. (`MAX_JOBS_PER_USER` still keys on account id and does not cover signed-out users.)
