#!/usr/bin/env python3
"""
Weaviate object-count drift check between local Pi and the replica primary.

Source .claude/infra.env first so WEAVIATE_HOST / WEAVIATE_PORT (local) and
WEAVIATE_REPLICA_PRIMARY_HOST / WEAVIATE_REPLICA_PRIMARY_PORT (remote) are set:

    set -a; source .claude/infra.env; set +a
    python3 scripts/diff.py

Cron-friendly: prints a per-collection table on stdout, summary on stderr,
exits 0 if all collections drift <= DRIFT_THRESHOLD_PCT, 1 if any exceed it,
2 if config or connection setup fails.

Tunables (env):
    DRIFT_THRESHOLD_PCT   default 0.5
    WEAVIATE_GRPC_PORT    default 50051 (used for both sides)
"""

import os
import sys

import weaviate


DRIFT_THRESHOLD_PCT = float(os.environ.get("DRIFT_THRESHOLD_PCT", "0.5"))


def _connect(host: str, http_port: int, grpc_port: int):
    return weaviate.connect_to_custom(
        http_host=host,
        http_port=http_port,
        http_secure=False,
        grpc_host=host,
        grpc_port=grpc_port,
        grpc_secure=False,
    )


def _collection_counts(client) -> dict:
    """Return {collection_name: object_count}. Counts that fail return -1."""
    counts = {}
    for name in client.collections.list_all():
        try:
            agg = client.collections.get(name).aggregate.over_all(total_count=True)
            counts[name] = agg.total_count
        except Exception as e:
            print(f"  WARN: count failed for {name}: {e}", file=sys.stderr)
            counts[name] = -1
    return counts


def _drift_pct(local: int, primary: int) -> float:
    if primary == 0:
        return 0.0 if local == 0 else 100.0
    return abs(local - primary) / primary * 100.0


def main() -> int:
    local_host = os.environ.get("WEAVIATE_HOST", "localhost")
    local_port = int(os.environ.get("WEAVIATE_PORT", "8080"))
    grpc_port = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))
    primary_host = os.environ.get("WEAVIATE_REPLICA_PRIMARY_HOST", "")
    primary_port = int(os.environ.get("WEAVIATE_REPLICA_PRIMARY_PORT", "8080"))

    if not primary_host:
        print(
            "ERROR: WEAVIATE_REPLICA_PRIMARY_HOST is empty. "
            "Source .claude/infra.env first.",
            file=sys.stderr,
        )
        return 2

    print(f"Local:   {local_host}:{local_port}")
    print(f"Primary: {primary_host}:{primary_port}")
    print(f"Threshold: {DRIFT_THRESHOLD_PCT}%\n")

    try:
        local = _connect(local_host, local_port, grpc_port)
        primary = _connect(primary_host, primary_port, grpc_port)
    except Exception as e:
        print(f"ERROR: connection setup failed: {e}", file=sys.stderr)
        return 2

    try:
        local_counts = _collection_counts(local)
        primary_counts = _collection_counts(primary)
    finally:
        local.close()
        primary.close()

    names = sorted(set(local_counts) | set(primary_counts))

    print(f"{'Collection':<28} {'Local':>10} {'Primary':>10} {'Drift':>10} {'Status':>8}")
    print("-" * 70)

    over_threshold = []
    for name in names:
        l = local_counts.get(name, 0)
        p = primary_counts.get(name, 0)
        pct = _drift_pct(l, p)
        status = "OK" if pct <= DRIFT_THRESHOLD_PCT else "DRIFT"
        if status == "DRIFT":
            over_threshold.append((name, l, p, pct))
        print(f"{name:<28} {l:>10} {p:>10} {pct:>9.2f}% {status:>8}")

    if over_threshold:
        print(
            f"\n{len(over_threshold)} collection(s) over threshold "
            f"({DRIFT_THRESHOLD_PCT}%): "
            + ", ".join(n for n, *_ in over_threshold),
            file=sys.stderr,
        )
        return 1

    print(f"\nAll {len(names)} collections within threshold ({DRIFT_THRESHOLD_PCT}%).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
