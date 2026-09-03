#!/usr/bin/env python3
"""
generate_3_log_files.py
========================
Generates 3 realistic log files from 3 different production systems.
Each has a different type of planted incident.

System 1: Nginx web server  → 502 Bad Gateway surge (upstream timeout)
System 2: Kubernetes pod    → OOMKilled crash loop (memory leak)
System 3: PostgreSQL DB     → Deadlock cascade (transaction lock contention)

Run: python3 generate_3_log_files.py
"""

import os
import random

os.makedirs("logs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# System 1: Nginx Web Server — 502 Bad Gateway surge
# Root cause: upstream app server ran out of worker threads
# ─────────────────────────────────────────────────────────────────────────────

NGINX_LOG = """2024-01-15T02:10:01+00:00 [info ] nginx/1.24.0 starting worker processes (4 workers)
2024-01-15T02:10:01+00:00 [info ] accepting connections on 0.0.0.0:443
2024-01-15T02:11:02+00:00 [info ] GET /api/products 200 45ms upstream=app-svc-01:8080
2024-01-15T02:11:03+00:00 [info ] GET /api/users/profile 200 62ms upstream=app-svc-02:8080
2024-01-15T02:11:04+00:00 [info ] POST /api/orders 200 88ms upstream=app-svc-01:8080
2024-01-15T02:11:05+00:00 [info ] GET /api/products?cat=electronics 200 51ms upstream=app-svc-03:8080
2024-01-15T02:11:06+00:00 [info ] GET /api/search?q=laptop 200 72ms upstream=app-svc-02:8080
2024-01-15T02:11:07+00:00 [info ] POST /api/cart/add 200 39ms upstream=app-svc-01:8080
2024-01-15T02:11:08+00:00 [info ] GET /api/recommendations 200 110ms upstream=app-svc-04:8080
2024-01-15T02:11:09+00:00 [info ] GET /health 200 2ms upstream=app-svc-01:8080
2024-01-15T02:12:00+00:00 [info ] upstream health check OK | all 4 backends responding
2024-01-15T02:13:01+00:00 [info ] GET /api/products 200 47ms upstream=app-svc-02:8080
2024-01-15T02:13:02+00:00 [info ] POST /api/checkout 200 221ms upstream=app-svc-01:8080
2024-01-15T02:13:03+00:00 [info ] GET /api/orders/history 200 95ms upstream=app-svc-03:8080
2024-01-15T02:13:04+00:00 [info ] GET /api/users/profile 200 58ms upstream=app-svc-04:8080
2024-01-15T02:14:00+00:00 [info ] connection rate: 42 req/s | avg_latency: 78ms | error_rate: 0%
2024-01-15T02:15:01+00:00 [info ] GET /api/products 200 52ms upstream=app-svc-02:8080
2024-01-15T02:15:02+00:00 [info ] POST /api/orders 200 91ms upstream=app-svc-01:8080
2024-01-15T02:16:00+00:00 [info ] upstream health check OK | all 4 backends responding
2024-01-15T02:17:01+00:00 [info ] GET /api/search?q=phone 200 68ms upstream=app-svc-03:8080
2024-01-15T02:17:02+00:00 [info ] POST /api/cart/add 200 41ms upstream=app-svc-02:8080
2024-01-15T02:18:00+00:00 [info ] connection rate: 38 req/s | avg_latency: 74ms | error_rate: 0%
2024-01-15T02:19:01+00:00 [info ] GET /api/products 200 49ms upstream=app-svc-01:8080
2024-01-15T02:20:00+00:00 [info ] upstream health check OK | all 4 backends responding
2024-01-15T02:20:01+00:00 [info ] connection rate spike detected | req/s: 180 (normal: 40) | possible traffic surge
2024-01-15T02:20:02+00:00 [warn ] upstream app-svc-01:8080 response time degraded | latency: 1820ms (threshold: 500ms)
2024-01-15T02:20:03+00:00 [warn ] upstream app-svc-02:8080 response time degraded | latency: 1940ms
2024-01-15T02:20:04+00:00 [warn ] upstream app-svc-03:8080 response time degraded | latency: 2100ms
2024-01-15T02:20:05+00:00 [warn ] upstream queue depth rising | app-svc-01: 48 pending | app-svc-02: 51 pending
2024-01-15T02:20:06+00:00 [warn ] upstream app-svc-01:8080 worker thread pool EXHAUSTED | active: 50/50 | queued: 62
2024-01-15T02:20:07+00:00 [error] upstream timed out (110: Connection timed out) | GET /api/products | upstream=app-svc-01:8080
2024-01-15T02:20:07+00:00 [error] 502 Bad Gateway | GET /api/products | client: 203.0.113.45
2024-01-15T02:20:08+00:00 [error] upstream timed out (110: Connection timed out) | POST /api/orders | upstream=app-svc-02:8080
2024-01-15T02:20:08+00:00 [error] 502 Bad Gateway | POST /api/orders | client: 198.51.100.22
2024-01-15T02:20:09+00:00 [error] upstream timed out (110: Connection timed out) | GET /api/search | upstream=app-svc-03:8080
2024-01-15T02:20:09+00:00 [error] 502 Bad Gateway | GET /api/search | client: 192.0.2.11
2024-01-15T02:20:10+00:00 [error] upstream timed out (110: Connection timed out) | POST /api/checkout | upstream=app-svc-04:8080
2024-01-15T02:20:10+00:00 [error] 502 Bad Gateway | POST /api/checkout | client: 203.0.113.88 | revenue_at_risk: $450.00
2024-01-15T02:20:11+00:00 [error] error_rate: 87% in last 10s | 28 requests failed | normal: <0.1%
2024-01-15T02:20:12+00:00 [error] all upstream backends unresponsive | serving 503 Service Unavailable to all clients
2024-01-15T02:20:13+00:00 [error] 503 Service Unavailable | GET /api/products | circuit open
2024-01-15T02:20:14+00:00 [error] 503 Service Unavailable | POST /api/cart | circuit open
2024-01-15T02:20:15+00:00 [warn ] retrying upstream app-svc-01:8080 after 10s backoff...
2024-01-15T02:20:45+00:00 [warn ] upstream app-svc-01:8080 recovering | active threads: 48/50 | latency: 820ms
2024-01-15T02:20:48+00:00 [info ] upstream app-svc-01:8080 healthy | latency: 210ms
2024-01-15T02:20:50+00:00 [info ] upstream app-svc-02:8080 healthy | latency: 195ms
2024-01-15T02:20:52+00:00 [info ] upstream app-svc-03:8080 healthy | latency: 188ms
2024-01-15T02:20:54+00:00 [info ] upstream app-svc-04:8080 healthy | latency: 201ms
2024-01-15T02:20:55+00:00 [info ] all upstream backends RECOVERED | resuming normal traffic routing
2024-01-15T02:21:00+00:00 [info ] GET /api/products 200 61ms upstream=app-svc-01:8080 | post-incident
2024-01-15T02:21:01+00:00 [info ] POST /api/orders 200 89ms upstream=app-svc-02:8080 | post-incident
2024-01-15T02:22:00+00:00 [info ] connection rate: 41 req/s | avg_latency: 81ms | error_rate: 0% | NORMAL
2024-01-15T02:23:00+00:00 [info ] incident summary | duration: 48s | requests_failed: 42 | revenue_at_risk: $2,140.00 | root_cause: worker_thread_exhaustion
"""

# ─────────────────────────────────────────────────────────────────────────────
# System 2: Kubernetes Pod — OOMKilled crash loop
# Root cause: memory leak in report generation endpoint, no memory limits set
# ─────────────────────────────────────────────────────────────────────────────

K8S_LOG = """2024-01-15T03:00:01Z INFO  kubelet node=prod-node-04 pod=report-svc-7d9f8b-xk2pq starting container report-service:v2.3.1
2024-01-15T03:00:02Z INFO  kubelet node=prod-node-04 pod=report-svc-7d9f8b-xk2pq container started successfully
2024-01-15T03:00:03Z INFO  report-svc pid=1 listening on :8090 | workers=8 | heap_limit=none
2024-01-15T03:00:15Z INFO  report-svc GET /reports/daily 200 1.2s | mem_rss=128MB
2024-01-15T03:00:30Z INFO  report-svc GET /reports/weekly 200 2.1s | mem_rss=132MB
2024-01-15T03:00:45Z INFO  report-svc POST /reports/export?format=pdf 200 3.4s | mem_rss=141MB
2024-01-15T03:01:00Z INFO  report-svc GET /health 200 2ms | mem_rss=141MB
2024-01-15T03:01:15Z INFO  report-svc GET /reports/daily 200 1.1s | mem_rss=143MB
2024-01-15T03:01:30Z INFO  report-svc POST /reports/export?format=xlsx 200 2.8s | mem_rss=149MB
2024-01-15T03:02:00Z INFO  report-svc GET /health 200 2ms | mem_rss=151MB
2024-01-15T03:02:15Z INFO  report-svc GET /reports/monthly 200 4.2s | mem_rss=158MB
2024-01-15T03:02:30Z INFO  report-svc POST /reports/export?format=pdf 200 3.8s | mem_rss=168MB
2024-01-15T03:03:00Z INFO  report-svc GET /health 200 2ms | mem_rss=174MB
2024-01-15T03:03:15Z INFO  report-svc GET /reports/weekly 200 2.0s | mem_rss=181MB
2024-01-15T03:03:30Z INFO  report-svc POST /reports/export?format=pdf 200 3.6s | mem_rss=192MB
2024-01-15T03:04:00Z INFO  report-svc GET /health 200 3ms | mem_rss=198MB
2024-01-15T03:04:15Z INFO  report-svc GET /reports/daily 200 1.3s | mem_rss=207MB
2024-01-15T03:05:00Z WARN  report-svc memory usage elevated | mem_rss=241MB | gc_pressure=high | heap_objects=8.2M
2024-01-15T03:05:15Z WARN  report-svc POST /reports/export?format=pdf 200 6.1s | mem_rss=268MB | GC pause: 890ms
2024-01-15T03:05:30Z WARN  report-svc memory leak suspected | rss growing 12MB/min | objects not being freed after export
2024-01-15T03:05:45Z WARN  report-svc POST /reports/export?format=xlsx 200 8.2s | mem_rss=312MB | GC pause: 1240ms
2024-01-15T03:06:00Z WARN  report-svc memory usage critical | mem_rss=378MB | heap_objects=28.4M | GC thrashing
2024-01-15T03:06:15Z WARN  kubelet node=prod-node-04 pod=report-svc-7d9f8b-xk2pq memory approaching node limit | rss=401MB
2024-01-15T03:06:30Z WARN  report-svc POST /reports/export?format=pdf 500 timeout | mem_rss=445MB | MEMORY PRESSURE: unable to allocate
2024-01-15T03:06:45Z ERROR report-svc out of memory: Kill process 1 (report-service) score 892 or sacrifice child
2024-01-15T03:06:45Z ERROR kubelet node=prod-node-04 pod=report-svc-7d9f8b-xk2pq OOMKilled | exit_code=137 | mem_at_kill=448MB
2024-01-15T03:06:46Z ERROR kubelet restarting pod report-svc-7d9f8b-xk2pq | restart_count=1 | reason=OOMKilled
2024-01-15T03:06:50Z INFO  report-svc pid=1 restarted | mem_rss=128MB | restart_count=1
2024-01-15T03:07:00Z INFO  report-svc GET /reports/daily 200 1.2s | mem_rss=131MB | [RESTARTED]
2024-01-15T03:07:30Z INFO  report-svc POST /reports/export?format=pdf 200 3.5s | mem_rss=142MB
2024-01-15T03:08:00Z WARN  report-svc memory growing again | mem_rss=189MB | SAME LEAK PATTERN
2024-01-15T03:08:30Z WARN  report-svc mem_rss=268MB | GC pause: 1.1s | export handler not releasing chart buffers
2024-01-15T03:09:00Z WARN  report-svc mem_rss=352MB | restart imminent
2024-01-15T03:09:15Z ERROR kubelet node=prod-node-04 pod=report-svc-7d9f8b-xk2pq OOMKilled | exit_code=137 | restart_count=2
2024-01-15T03:09:20Z WARN  kubelet pod in CrashLoopBackOff | restart_count=2 | back_off=20s
2024-01-15T03:09:45Z ERROR kubelet node=prod-node-04 pod=report-svc-7d9f8b-xk2pq OOMKilled | exit_code=137 | restart_count=3
2024-01-15T03:10:00Z ERROR kubelet pod in CrashLoopBackOff | restart_count=3 | back_off=40s | service DOWN
2024-01-15T03:10:01Z ERROR alertmanager CRITICAL: report-svc CrashLoopBackOff | 3 restarts in 4min | on-call paged
2024-01-15T03:10:05Z INFO  kubelet pod report-svc-7d9f8b-xk2pq | applying memory limit: 512Mi (emergency patch)
2024-01-15T03:10:10Z INFO  report-svc pod restarted with memory limit | mem_limit=512Mi | restart_count=4
2024-01-15T03:10:30Z WARN  report-svc POST /reports/export | mem_rss=189MB | chart buffers accumulating (leak still present)
2024-01-15T03:11:00Z INFO  report-svc GET /health 200 2ms | mem_rss=201MB | pod stable with limit
2024-01-15T03:12:00Z INFO  report-svc service recovering | mem_rss=210MB | within limit | no crash in 2min
2024-01-15T03:15:00Z INFO  report-svc stable | mem_rss=230MB | reports available | fix needed: chart buffer not freed in export handler
"""

# ─────────────────────────────────────────────────────────────────────────────
# System 3: PostgreSQL Database — Deadlock cascade
# Root cause: two transaction types acquiring locks in opposite order
# ─────────────────────────────────────────────────────────────────────────────

PG_LOG = """2024-01-15 04:00:00.001 UTC [12841] LOG:  database system is ready to accept connections
2024-01-15 04:00:00.002 UTC [12841] LOG:  autovacuum launcher started
2024-01-15 04:01:12.221 UTC [12901] LOG:  connection received: host=10.0.1.44 user=app_user database=ecommerce
2024-01-15 04:01:12.225 UTC [12901] LOG:  connection authorized: user=app_user database=ecommerce
2024-01-15 04:01:13.112 UTC [12901] LOG:  statement: BEGIN; UPDATE inventory SET qty=qty-1 WHERE product_id=1001; UPDATE orders SET status='confirmed' WHERE id=5521; COMMIT;
2024-01-15 04:01:13.220 UTC [12901] LOG:  duration: 8.112 ms statement: COMMIT
2024-01-15 04:01:14.001 UTC [12902] LOG:  connection received: host=10.0.1.45 user=app_user database=ecommerce
2024-01-15 04:01:15.112 UTC [12902] LOG:  statement: BEGIN; UPDATE orders SET status='shipped' WHERE id=5510; UPDATE inventory SET last_shipped=NOW() WHERE product_id=1002; COMMIT;
2024-01-15 04:01:15.218 UTC [12902] LOG:  duration: 7.881 ms statement: COMMIT
2024-01-15 04:02:00.001 UTC [12841] LOG:  checkpoint starting: time
2024-01-15 04:02:01.441 UTC [12841] LOG:  checkpoint complete: wrote 128 buffers (0.8%); 0 WAL file(s) added, 0 removed; write=0.421 s
2024-01-15 04:02:45.001 UTC [12903] LOG:  statement: BEGIN; UPDATE inventory SET qty=qty-1 WHERE product_id=1001; -- txn A start
2024-01-15 04:02:45.008 UTC [12904] LOG:  statement: BEGIN; UPDATE orders SET status='confirmed' WHERE id=5522; -- txn B start (opposite lock order)
2024-01-15 04:02:45.010 UTC [12903] LOG:  statement: UPDATE orders SET status='confirmed' WHERE id=5522; -- txn A waiting for row lock held by txn B
2024-01-15 04:02:45.012 UTC [12904] LOG:  statement: UPDATE inventory SET qty=qty-1 WHERE product_id=1001; -- txn B waiting for row lock held by txn A
2024-01-15 04:02:45.088 UTC [12841] WARNING:  process 12903 still waiting for ShareLock on transaction 82441 after 78ms | blocking pid: 12904
2024-01-15 04:02:45.089 UTC [12841] WARNING:  process 12904 still waiting for ShareLock on transaction 82440 after 77ms | blocking pid: 12903
2024-01-15 04:02:45.091 UTC [12841] ERROR:  deadlock detected | DETAIL: Process 12903 waits for ShareLock on transaction 82441; blocked by process 12904. Process 12904 waits for ShareLock on transaction 82440; blocked by process 12903.
2024-01-15 04:02:45.092 UTC [12903] ERROR:  deadlock detected | HINT: See server log for query details.
2024-01-15 04:02:45.093 UTC [12904] LOG:  statement: ROLLBACK -- txn B rolled back as deadlock victim
2024-01-15 04:02:45.094 UTC [12903] LOG:  statement: COMMIT -- txn A proceeded after txn B rolled back
2024-01-15 04:02:46.001 UTC [12905] LOG:  statement: BEGIN; UPDATE inventory SET qty=qty-1 WHERE product_id=1002;
2024-01-15 04:02:46.010 UTC [12906] LOG:  statement: BEGIN; UPDATE orders SET status='confirmed' WHERE id=5523;
2024-01-15 04:02:46.018 UTC [12905] WARNING:  waiting for lock on row in relation "orders"
2024-01-15 04:02:46.019 UTC [12906] WARNING:  waiting for lock on row in relation "inventory"
2024-01-15 04:02:46.082 UTC [12841] ERROR:  deadlock detected | process 12905 waits for ShareLock; blocked by process 12906. Process 12906 waits for ShareLock; blocked by process 12905.
2024-01-15 04:02:46.083 UTC [12905] ERROR:  deadlock detected | ERROR: deadlock detected
2024-01-15 04:02:47.001 UTC [12907] ERROR:  deadlock detected | CONTEXT: txn acquiring inventory lock then orders lock (reverse of order_service txn pattern)
2024-01-15 04:02:47.110 UTC [12841] LOG:  deadlock cascade: 8 transactions rolled back in last 2s | lock_wait_queue: 24 pending
2024-01-15 04:02:48.001 UTC [12841] WARNING:  lock contention critical | waiter_count=31 | longest_wait=2841ms
2024-01-15 04:02:48.500 UTC [12841] ERROR:  max_locks_per_transaction exceeded | lock table full | new transactions cannot acquire locks
2024-01-15 04:02:49.001 UTC [12841] ERROR:  FATAL: all transaction slots are in use | refusing new connections
2024-01-15 04:02:49.500 UTC [12841] ERROR:  app_user connection rejected: connection limit reached | active: 100/100
2024-01-15 04:02:50.001 UTC [12841] ERROR:  app_user connection rejected: connection limit reached | active: 100/100
2024-01-15 04:02:50.500 UTC [12908] ERROR:  database unavailable to application layer | 14 requests failed in 3s
2024-01-15 04:03:00.001 UTC [12841] LOG:  autovacuum: lock contention resolved after 14s | rolled back 41 transactions
2024-01-15 04:03:05.001 UTC [12841] LOG:  lock queue draining | waiter_count=8
2024-01-15 04:03:10.001 UTC [12841] LOG:  lock contention resolved | waiter_count=0 | normal operation resumed
2024-01-15 04:03:12.001 UTC [12901] LOG:  statement: BEGIN; UPDATE inventory SET qty=qty-1 WHERE product_id=1001; UPDATE orders SET status='confirmed' WHERE id=5560; COMMIT; -- fixed order
2024-01-15 04:03:12.010 UTC [12901] LOG:  duration: 9.1 ms | no deadlock | RECOVERED
2024-01-15 04:05:00.001 UTC [12841] LOG:  incident summary: deadlock_cascade | duration=75s | transactions_rolled_back=41 | root_cause=inconsistent_lock_order_inventory_vs_orders
"""

# Write the files
files = {
    "logs/nginx_web_server.log"   : NGINX_LOG.strip(),
    "logs/kubernetes_pods.log"    : K8S_LOG.strip(),
    "logs/postgresql_db.log"      : PG_LOG.strip(),
}

for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)
    lines = content.count("\n") + 1
    print(f"✅ {path}  ({lines} lines)")

print(f"\n{'─'*55}")
print(f"  3 log files generated in logs/")
print(f"  Each has a different planted incident:")
print(f"  1. nginx_web_server.log  → 502 surge (thread exhaustion)")
print(f"  2. kubernetes_pods.log   → OOMKilled crash loop (memory leak)")
print(f"  3. postgresql_db.log     → Deadlock cascade (lock order conflict)")
print(f"{'─'*55}\n")
print(f"  Next: python3 test_all_logs.py")
