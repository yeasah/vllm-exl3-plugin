#!/usr/bin/env python3
"""SQL against the primary host's telemetry (InfluxDB 3, populated by telegraf). Connection details
are read from ~/influx.creds (url:, database:, token: lines); the token is never printed.

usage: influxq.py "SELECT ..." [--format pretty|json|csv|jsonl] [--db metrics_cloud]

--db picks another database on the same server: metrics_cloud holds the cloud instances' telegraf
(tunnelled back over ssh), where nvidia_smi, cpu and net are meaningful and the container's
diskio and mem are not.

Filter by host: this VM is host = 'llm.home.schwide.net' in cpu, mem, diskio, nvidia_smi; the HF
cache's dm-cache is reported by the hypervisor, host = 'gauss', as cumulative counters (difference
them). Timestamps are UTC. Example:
  influxq.py "SELECT date_bin(INTERVAL '2 minutes', time) AS t, \"index\", avg(utilization_gpu)
              FROM nvidia_smi WHERE host = 'llm.home.schwide.net' AND time > now() - INTERVAL '1 hour'
              GROUP BY 1, 2 ORDER BY 1, 2"
"""
import sys, json, urllib.request, os
creds = {}
for line in open(os.path.expanduser("~/influx.creds")):
    if ":" in line:
        k, v = line.split(":", 1)
        creds[k.strip()] = v.strip()
url = creds["url"].rstrip("/") + "/api/v3/query_sql"
fmt = sys.argv[sys.argv.index("--format") + 1] if "--format" in sys.argv else "pretty"
db = sys.argv[sys.argv.index("--db") + 1] if "--db" in sys.argv else creds["database"]
q = sys.argv[1]
req = urllib.request.Request(url, data=json.dumps({"db": db, "q": q, "format": fmt}).encode(),
                             headers={"Authorization": f"Bearer {creds['token']}", "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        sys.stdout.write(r.read().decode())
except urllib.error.HTTPError as e:
    sys.exit(f"HTTP {e.code}: {e.read().decode()[:500]}")
