# Runbook

## Deploy

```bash
git pull
docker compose build
docker compose up -d
docker compose logs -f app        # expect the selftest line, then Streamlit starting
```

The build runs `phishguard.selftest --golden` and fails if the baked bundle does not
reproduce its recorded golden row. A build failure here is good news: it caught a bundle
mismatch before it reached the host.

Verify:

```bash
curl -fsS https://$SITE_ADDRESS/_stcore/health
docker compose exec app python -m phishguard.selftest --golden
```

## Roll back

The artifact bundle is baked into the image, so the image tag fully describes what the
service predicts. Rollback is therefore just the previous tag:

```bash
docker compose down
docker tag phishguard:<previous-sha> phishguard:latest
docker compose up -d
```

No artifact directory to restore, and no ambiguity about which models are answering.

## Flip the kill switch

To stop all outbound fetching without a rebuild — the service degrades to URL-only
features and renders that state rather than erroring:

```bash
FETCH_ENABLED=false docker compose up -d app
```

Reach for this when an upstream complains, when outbound traffic looks anomalous, or when
you need the service up while you investigate. Coverage will read 0% for every lookup and
the app will abstain, which is the correct behaviour: with no page evidence it has nothing
to judge.

## Block a specific host

```bash
FETCH_DENY_HOSTS=abusive.example.com,other.example docker compose up -d app
```

Subdomains of a listed host are covered. This is the narrow response; the kill switch is
the blunt one.

## Rotate the bundle

Bundles are not hot-swapped. Retrain, rebuild, redeploy:

```bash
python -m phishguard.train --profile corrected
docker compose build && docker compose up -d
```

Mounting a bundle instead would decouple the image tag from the behaviour and reintroduce
the "which model is actually serving?" question that baking exists to answer.

## Someone reports being scanned by this host

1. Confirm from the proxy log which source IP drove it, and over what window.
2. `FETCH_ENABLED=false` if it is ongoing. Ask questions after it stops.
3. Add the target to `FETCH_DENY_HOSTS` if it is one destination.
4. If one client is responsible, tighten `FETCH_RATE_PER_SESSION` and add a per-IP limit at
   the proxy. The application's limiter is per *session*, and a session is a cookie — it
   bounds one browser tab, not one client. Both layers are needed.

Note what cannot have happened: the guard only permits public addresses, revalidates every
redirect hop, and connects to the address it validated. A report of internal-network access
originating here would mean a guard defect, so treat it as a security bug and add the case
to the SSRF test table before shipping a fix.

## Container is unhealthy

```bash
docker compose logs --tail=100 app
docker stats --no-stream
```

Most likely causes, in order:

- **OOM-killed.** In-flight memory is `FETCH_CONCURRENCY × FETCH_MAX_BYTES` on top of a
  ~400 MB baseline. If either was raised, raise `mem_limit` or put them back.
- **Selftest failed at boot.** The bundle does not match its manifest. Rebuild rather than
  restarting — a container that will not boot with a bundle it cannot account for is
  behaving correctly.
- **Health check timing out under load.** BLAS threads are capped at 2 deliberately; if
  that was raised on a 2 vCPU host, the KNN kernel starves the web server.

## Do not scale horizontally

`docker compose up --scale app=N` will appear to work and will drop sessions. Streamlit is
WebSocket-stateful, so a reconnect landing on a different replica loses its state. Scaling
this service means solving session affinity at the proxy first.
