#!/usr/bin/env python3
"""
Generate synthetic AcmeCorp Payment Service logs.
Simulates a real incident: DB connection pool exhausted by an analytics batch job.

Root cause planted at line ~63:
    AnalyticsJob held 27 DB connections without releasing (missing DB index → full table scan)
    → Connection pool exhausted → TimeoutExceptions → Payments failed
"""

import os
from datetime import datetime, timedelta

BASE_TIME = datetime(2024, 1, 15, 1, 45, 0)

def ts(seconds):
    """Timestamp string offset from base time."""
    return (BASE_TIME + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S.") + \
           f"{int((seconds % 1) * 1000):03d}"

lines = []

def log(secs, level, thread, component, message):
    level_str = level.ljust(5)
    comp_str  = component.ljust(20)
    lines.append(f"{ts(secs)} {level_str} [payment-svc] [{thread}] {comp_str} - {message}")

# ─── PHASE 1: Normal operations (Lines 1–55) ─────────────────────────────────

payments = [
    ("REQ-10201", "usr_4421", "$89.99",    "HDFC",  "7823", 0,   "AU-882341"),
    ("REQ-10202", "usr_5511", "$234.00",   "ICICI", "4421", 15,  "AU-883572"),
    ("REQ-10203", "usr_2201", "$45.50",    "SBI",   "9912", 30,  "AU-884803"),
    ("REQ-10204", "usr_7731", "$512.00",   "HDFC",  "3341", 55,  "AU-886034"),
    ("REQ-10205", "usr_1102", "$19.99",    "Axis",  "8821", 75,  "AU-887265"),
    ("REQ-10206", "usr_8891", "$1,200.00", "Kotak", "2291", 100, "AU-888496"),
    ("REQ-10207", "usr_3312", "$67.30",    "ICICI", "5571", 130, "AU-889727"),
    ("REQ-10208", "usr_6621", "$330.00",   "HDFC",  "1183", 160, "AU-890958"),
]

threads  = ["thread-01","thread-02","thread-03","thread-04","thread-05","thread-06"]
conns    = ["conn-017","conn-022","conn-031","conn-008","conn-044","conn-011"]
pool_val = 8

for i, (req, user, amt, bank, last4, offset, auth) in enumerate(payments):
    th = threads[i % len(threads)]
    cn = conns[i % len(conns)]
    pool_in  = pool_val + 1
    pool_out = pool_val
    log(offset + 0.0,  "INFO", th, "PaymentProcessor",  f"Processing payment {req} | user: {user} | amount: {amt}")
    log(offset + 0.45, "INFO", th, "DBConnectionPool",  f"Acquired connection {cn} [pool: {pool_in}/50 active]")
    log(offset + 0.66, "INFO", th, "PaymentValidator",  f"Card validation passed {req} | bank: {bank} | last4: {last4}")
    log(offset + 1.21, "INFO", th, "PaymentGateway",    f"Authorization request sent | {req} | gateway: Stripe")
    log(offset + 1.75, "INFO", th, "PaymentGateway",    f"Authorization approved | {req} | auth_code: {auth}")
    log(offset + 1.82, "INFO", th, "PaymentProcessor",  f"Payment {req} COMPLETED | {amt} charged")
    log(offset + 1.84, "INFO", th, "DBConnectionPool",  f"Released connection {cn} [pool: {pool_out}/50 active]")

log(180,   "INFO", "thread-01", "HealthCheck",      "Service health PASSED | DB: OK | Gateway: OK | Cache: OK | pool: 13/50")
log(240,   "INFO", "thread-01", "MetricsCollector", "Metrics snapshot: payments=8 | success_rate=100% | avg_latency=1.81s | pool=26%")
log(270,   "INFO", "thread-02", "PaymentProcessor",  "Processing payment REQ-10209 | user: usr_9901 | amount: $88.75")
log(270.4, "INFO", "thread-02", "DBConnectionPool",  "Acquired connection conn-031 [pool: 14/50 active]")
log(271.2, "INFO", "thread-02", "PaymentGateway",    "Authorization approved | REQ-10209 | auth_code: AU-892189")
log(271.3, "INFO", "thread-02", "PaymentProcessor",  "Payment REQ-10209 COMPLETED | $88.75 charged")
log(271.3, "INFO", "thread-02", "DBConnectionPool",  "Released connection conn-031 [pool: 13/50 active]")

# ─── PHASE 2: Analytics batch job — pool fills up (Lines 56–75) ──────────────

log(300.1, "INFO", "thread-07", "Scheduler",        "Triggering scheduled monthly analytics job | job_id: BATCH-2024-01")
log(300.6, "INFO", "thread-07", "AnalyticsJob",     "Starting BATCH-2024-01 | query: monthly_revenue_report | est_duration: 45min")
log(301.1, "INFO", "thread-07", "AnalyticsJob",     "Acquiring DB connections for parallel analytics queries...")
log(301.5, "INFO", "thread-07", "DBConnectionPool", "Acquired 5 connections for BATCH-2024-01 [pool: 18/50 active]")
log(302.0, "INFO", "thread-07", "AnalyticsJob",     "Running query: monthly_revenue_by_category | table: transactions | est_rows: ~2.4M")
log(302.5, "INFO", "thread-07", "DBConnectionPool", "Acquired 10 more connections for BATCH-2024-01 [pool: 23/50 active]")
log(315.0, "WARN", "thread-07", "DBConnectionPool", "Connection pool at 40% capacity [pool: 20/50 active] | Holder: BATCH-2024-01")

# ← THIS IS THE ROOT CAUSE LINE (line ~63 in the file)
log(345.2, "WARN", "thread-07", "AnalyticsJob",
    "SLOW QUERY DETECTED | full_table_scan on transactions | elapsed: 41s | "
    "rows_scanned: 2,401,882 | connections_held: 27 | "
    "ROOT CAUSE: MISSING INDEX on transactions.date_column")

log(346.0, "WARN", "thread-07", "DBConnectionPool", "Connection pool at 60% capacity [pool: 30/50 active] | 27 connections held by BATCH-2024-01")
log(362.1, "WARN", "thread-07", "DBConnectionPool", "Connection pool at 70% capacity [pool: 35/50 active] | Long-running queries not releasing")
log(378.3, "WARN", "thread-08", "PaymentProcessor",  "REQ-10210 waiting for DB connection | elapsed: 3.2s | queue_depth: 2")
log(382.4, "WARN", "thread-07", "DBConnectionPool", "Connection pool at 80% capacity [pool: 40/50 active] | WARNING: Approaching limit")
log(388.5, "WARN", "thread-08", "PaymentProcessor",  "REQ-10210 still waiting | elapsed: 13.4s | timeout at 30s")
log(394.6, "WARN", "thread-07", "DBConnectionPool", "Connection pool at 90% capacity [pool: 45/50 active] | CRITICAL: 5 connections left")
log(398.7, "WARN", "thread-09", "PaymentProcessor",  "REQ-10211 waiting for DB connection | elapsed: 0.8s")
log(400.8, "WARN", "thread-10", "PaymentProcessor",  "REQ-10212 waiting for DB connection | elapsed: 0.2s | queue_depth: 4")
log(402.9, "ERROR","thread-07", "DBConnectionPool", "Connection pool EXHAUSTED [pool: 50/50 active] — new requests will wait or TIMEOUT")
log(403.0, "WARN", "thread-07", "DBConnectionPool", "3 payment requests queued | avg_wait: 18.2s")
log(405.0, "WARN", "thread-07", "DBConnectionPool", "6 payment requests queued | avg_wait: 20.1s")
log(408.1, "WARN", "thread-08", "PaymentProcessor",  "REQ-10210 waiting | elapsed: 29.1s | timeout in 0.9s")

# ─── PHASE 3: Payment failures cascade (Lines 76–95) ─────────────────────────

log(409.1, "ERROR","thread-08", "PaymentProcessor",  "java.sql.SQLException: Timeout waiting for connection after 30000ms | REQ-10210 | user: usr_4412 | $145.00")
log(409.1, "ERROR","thread-08", "PaymentProcessor",  "Payment REQ-10210 FAILED | user: usr_4412 | $145.00 NOT charged | reason: DB_CONNECTION_TIMEOUT")
log(409.2, "ERROR","thread-09", "PaymentProcessor",  "java.sql.SQLException: Timeout waiting for connection after 30000ms | REQ-10211 | user: usr_7723 | $55.00")
log(409.2, "ERROR","thread-09", "PaymentProcessor",  "Payment REQ-10211 FAILED | user: usr_7723 | $55.00 NOT charged | reason: DB_CONNECTION_TIMEOUT")
log(409.3, "ERROR","thread-10", "PaymentProcessor",  "java.sql.SQLException: Timeout waiting for connection after 30000ms | REQ-10212 | user: usr_8821 | $142.50")
log(409.4, "ERROR","thread-10", "PaymentProcessor",  "Payment REQ-10212 FAILED | user: usr_8821 | $142.50 NOT charged | reason: DB_CONNECTION_TIMEOUT")
log(410.0, "WARN", "thread-01", "CircuitBreaker",   "Payment failure rate: 43% in last 60s | threshold: 20% | CIRCUIT OPENING")
log(410.1, "ERROR","thread-01", "CircuitBreaker",   "CIRCUIT BREAKER OPEN | Rejecting ALL new payment requests | retry_after: 30s")
log(410.5, "ERROR","thread-02", "PaymentGateway",   "CIRCUIT OPEN — Rejecting REQ-10213 | user: usr_6612 | $55.00 blocked")
log(411.0, "ERROR","thread-03", "AlertManager",     "CRITICAL ALERT: 5 payments failed in 2s | Incident: INC-2024-0115-001 | Severity: P1 | PagerDuty notified")
log(411.5, "ERROR","thread-02", "PaymentGateway",   "CIRCUIT OPEN — Rejecting REQ-10214 | user: usr_3341 | $890.00 blocked")
log(412.0, "ERROR","thread-02", "PaymentGateway",   "CIRCUIT OPEN — Rejecting REQ-10215 | user: usr_9912 | $330.00 blocked")
log(413.0, "ERROR","thread-02", "PaymentGateway",   "CIRCUIT OPEN — Rejecting REQ-10216 | user: usr_1122 | $44.00 blocked")
log(414.0, "ERROR","thread-02", "PaymentGateway",   "CIRCUIT OPEN — Rejecting REQ-10217 | user: usr_5512 | $221.50 blocked")
log(415.0, "WARN", "thread-07", "DBConnectionPool", "BATCH-2024-01 queries still running | connections_held: 27 | elapsed: 6min 17s")
log(415.5, "ERROR","thread-04", "AlertManager",     "INC-2024-0115-001 ESCALATED | On-call paged | Revenue at risk: $1,935.00 | 9 payments blocked")

# ─── PHASE 4: Recovery (Lines 96–107) ────────────────────────────────────────

log(513.0, "INFO", "thread-07", "AnalyticsJob",     "BATCH-2024-01 queries completing | 80% done | releasing partial connections")
log(517.0, "INFO", "thread-07", "AnalyticsJob",     "Releasing DB connections | releasing: 27 connections | job_id: BATCH-2024-01")
log(518.0, "INFO", "thread-07", "DBConnectionPool", "Connection pool draining [pool: 40/50 active] | 10 connections released by BATCH-2024-01")
log(520.0, "INFO", "thread-07", "DBConnectionPool", "Connection pool draining [pool: 28/50 active] | 22 connections released")
log(522.0, "INFO", "thread-07", "DBConnectionPool", "Connection pool NORMALIZED [pool: 15/50 active] | All BATCH-2024-01 connections released")
log(524.0, "INFO", "thread-01", "CircuitBreaker",   "Circuit HALF-OPEN | Testing payment service recovery with probe request...")
log(524.5, "INFO", "thread-01", "PaymentProcessor",  "Probe transaction REQ-10220-TEST sent | amount: $0.01 [circuit test]")
log(524.6, "INFO", "thread-01", "DBConnectionPool", "Acquired connection conn-019 [pool: 16/50 active] — immediate success")
log(525.0, "INFO", "thread-01", "PaymentGateway",   "Authorization approved | REQ-10220-TEST | circuit test PASSED")
log(525.5, "INFO", "thread-01", "CircuitBreaker",   "Circuit CLOSED | Payment service RECOVERED | Normal operations resuming")
log(526.0, "INFO", "thread-01", "AlertManager",     "RESOLVED: INC-2024-0115-001 | Duration: 1m 57s | Payments blocked: 9 | Revenue blocked: $1,935.00")
log(527.0, "INFO", "thread-07", "AnalyticsJob",     "BATCH-2024-01 COMPLETED | Total time: 8m 27s (expected 45min — caused by missing index)")

# ─── PHASE 5: Normal operations resume (Lines 108–120) ───────────────────────

for i, (req, user, amt, auth, offset) in enumerate([
    ("REQ-10221", "usr_2201", "$156.00", "AU-991001", 545),
    ("REQ-10222", "usr_5511", "$78.50",  "AU-992002", 570),
    ("REQ-10223", "usr_9901", "$445.00", "AU-993003", 600),
]):
    th = threads[i % len(threads)]
    cn = conns[i % len(conns)]
    log(offset,       "INFO", th, "PaymentProcessor", f"[POST-INCIDENT] Processing payment {req} | user: {user} | amount: {amt}")
    log(offset + 0.4, "INFO", th, "DBConnectionPool", f"Acquired connection {cn} [pool: {11+i}/50 active]")
    log(offset + 0.9, "INFO", th, "PaymentGateway",   f"Authorization approved | {req} | auth_code: {auth}")
    log(offset + 1.0, "INFO", th, "PaymentProcessor", f"[POST-INCIDENT] Payment {req} COMPLETED | {amt} charged | All systems NORMAL")
    log(offset + 1.1, "INFO", th, "DBConnectionPool", f"Released connection {cn} [pool: {10+i}/50 active]")

log(630, "INFO", "thread-01", "MetricsCollector",
    "Recovery metrics: payments_processed=3 | success_rate=100% | pool_utilization=24% | status=HEALTHY")

# ─── Write file ───────────────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)
with open("logs/acmecorp_payment.log", "w") as f:
    for line in lines:
        f.write(line + "\n")

print(f"✅ Generated {len(lines)} log lines → logs/acmecorp_payment.log")
print(f"   Root cause planted at line ~63")
print(f"   Failures at lines ~76–90")
print(f"   Recovery at lines ~96–107")
