"""
generate_demo_data.py — Synthetic IoT Cybersecurity Dataset Generator
======================================================================
Creates a realistic-looking synthetic dataset for demonstration.
All data is SYNTHETIC — it does NOT represent real-world traffic.

The generated dataset mimics the column structure of IoT-23 / BoT-IoT
so that the full pipeline can be demonstrated without a real dataset.
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────
NUM_SAMPLES = 5000
ATTACK_RATIO = 0.35  # 35 % attacks
RANDOM_SEED = 42

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# ── Attack types ─────────────────────────────────────────────
ATTACK_TYPES = [
    "DDoS",
    "DoS",
    "Reconnaissance",
    "Theft",
    "Normal",
]

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "dns", "ssh", "dhcp", "ssl", "smtp", "-"]
CONN_STATES = ["S0", "SF", "REJ", "RSTO", "RSTR", "SH", "SHR", "OTH"]

# ── Helper ───────────────────────────────────────────────────

def _random_ip(rng: np.random.Generator) -> str:
    return ".".join(str(rng.integers(1, 255)) for _ in range(4))


def generate(n: int = NUM_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_attacks = int(n * ATTACK_RATIO)
    n_normal = n - n_attacks

    rows = []
    for i in range(n):
        is_attack = i < n_attacks
        if is_attack:
            attack_type = rng.choice(ATTACK_TYPES[:-1])  # exclude 'Normal'
            label = attack_type
            detailed_label = f"Malicious-{attack_type}"
        else:
            label = "Benign"
            detailed_label = "Benign"

        proto = rng.choice(PROTOCOLS, p=[0.5, 0.35, 0.15])
        service = rng.choice(SERVICES)
        conn_state = rng.choice(CONN_STATES)

        # Normal traffic patterns
        if not is_attack:
            duration = max(0.0, rng.normal(5.0, 3.0))
            orig_bytes = max(0, int(rng.normal(500, 200)))
            resp_bytes = max(0, int(rng.normal(1200, 400)))
            orig_pkts = max(1, int(rng.normal(10, 5)))
            resp_pkts = max(1, int(rng.normal(15, 7)))
            src_port = int(rng.integers(1024, 65535))
            dst_port = int(rng.choice([80, 443, 53, 22, 8080, 993]))
        else:
            # Attacks tend to have anomalous patterns
            if attack_type == "DDoS":
                duration = max(0.0, rng.exponential(0.5))
                orig_bytes = max(0, int(rng.exponential(100)))
                resp_bytes = max(0, int(rng.exponential(50)))
                orig_pkts = max(1, int(rng.exponential(200)))
                resp_pkts = max(0, int(rng.exponential(5)))
                src_port = int(rng.integers(1024, 65535))
                dst_port = int(rng.choice([80, 443, 53]))
            elif attack_type == "DoS":
                duration = max(0.0, rng.exponential(1.0))
                orig_bytes = max(0, int(rng.normal(5000, 2000)))
                resp_bytes = max(0, int(rng.exponential(100)))
                orig_pkts = max(1, int(rng.normal(100, 50)))
                resp_pkts = max(0, int(rng.exponential(3)))
                src_port = int(rng.integers(1024, 65535))
                dst_port = int(rng.choice([80, 443]))
            elif attack_type == "Reconnaissance":
                duration = max(0.0, rng.exponential(0.2))
                orig_bytes = max(0, int(rng.exponential(60)))
                resp_bytes = max(0, int(rng.exponential(40)))
                orig_pkts = max(1, int(rng.exponential(3)))
                resp_pkts = max(0, int(rng.exponential(2)))
                src_port = int(rng.integers(1024, 65535))
                dst_port = int(rng.integers(1, 65535))
            else:  # Theft
                duration = max(0.0, rng.normal(30.0, 15.0))
                orig_bytes = max(0, int(rng.normal(10000, 5000)))
                resp_bytes = max(0, int(rng.normal(500, 200)))
                orig_pkts = max(1, int(rng.normal(50, 20)))
                resp_pkts = max(1, int(rng.normal(10, 5)))
                src_port = int(rng.integers(1024, 65535))
                dst_port = int(rng.choice([21, 22, 443, 3389, 8443]))

        rows.append({
            "ts": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=int(rng.integers(0, 86400 * 30))),
            "uid": f"C{rng.integers(100000, 999999)}",
            "id.orig_h": _random_ip(rng),
            "id.orig_p": src_port,
            "id.resp_h": _random_ip(rng),
            "id.resp_p": dst_port,
            "proto": proto,
            "service": service,
            "duration": round(duration, 4),
            "orig_bytes": orig_bytes,
            "resp_bytes": resp_bytes,
            "conn_state": conn_state,
            "orig_pkts": orig_pkts,
            "resp_pkts": resp_pkts,
            "orig_ip_bytes": orig_bytes + int(rng.integers(20, 60)) * orig_pkts,
            "resp_ip_bytes": resp_bytes + int(rng.integers(20, 60)) * resp_pkts,
            "label": label,
            "detailed-label": detailed_label,
        })

    df = pd.DataFrame(rows)
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main():
    print("=" * 60)
    print("  MLCS-HACK-26 — Synthetic Demo Data Generator")
    print("  ⚠  ALL DATA IS SYNTHETIC / FOR DEMONSTRATION ONLY")
    print("=" * 60)

    os.makedirs(RAW_DIR, exist_ok=True)

    df = generate()

    out_path = os.path.join(RAW_DIR, "demo_iot_traffic.csv")
    df.to_csv(out_path, index=False)

    print(f"\n✓ Generated {len(df)} records → {out_path}")
    print(f"  Normal:  {(df['label'] == 'Benign').sum()}")
    print(f"  Attack:  {(df['label'] != 'Benign').sum()}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Attack types: {df[df['label'] != 'Benign']['label'].value_counts().to_dict()}")
    print("\n⚠  This is DEMO/SYNTHETIC data. Do NOT use results for real security decisions.\n")


if __name__ == "__main__":
    main()
