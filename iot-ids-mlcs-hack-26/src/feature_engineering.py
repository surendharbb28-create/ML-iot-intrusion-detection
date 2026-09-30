"""
feature_engineering.py — Cybersecurity Feature Engineering
==========================================================
Inspects available columns and derives security-relevant features.
Only creates features when the required raw data actually exists.

Every engineered feature is documented in FEATURE_DOCS.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Feature documentation ────────────────────────────────────
FEATURE_DOCS: dict[str, str] = {}


def _register(name: str, doc: str):
    """Register documentation for an engineered feature."""
    FEATURE_DOCS[name] = doc


# ── Individual feature builders ──────────────────────────────
# Each function checks whether the required source columns exist
# and returns the DataFrame unchanged if they don't.

def add_bytes_per_packet(df: pd.DataFrame) -> pd.DataFrame:
    """bytes_per_packet = total_bytes / packet_count"""
    byte_cols = [c for c in df.columns if c.lower() in (
        "orig_bytes", "resp_bytes", "src_bytes", "dst_bytes",
        "sbytes", "dbytes", "totbytes", "total_bytes", "bytes"
    )]
    pkt_cols = [c for c in df.columns if c.lower() in (
        "orig_pkts", "resp_pkts", "spkts", "dpkts",
        "total_pkts", "pkts", "packets", "packet_count"
    )]
    if not byte_cols or not pkt_cols:
        return df

    total_bytes = df[byte_cols].sum(axis=1).astype(float)
    total_pkts = df[pkt_cols].sum(axis=1).replace(0, np.nan).astype(float)
    df["bytes_per_packet"] = (total_bytes / total_pkts).fillna(0)
    _register("bytes_per_packet", "Total bytes divided by total packets; high values may indicate data exfiltration.")
    return df


def add_packets_per_second(df: pd.DataFrame) -> pd.DataFrame:
    """packets_per_second = packet_count / duration"""
    pkt_cols = [c for c in df.columns if c.lower() in (
        "orig_pkts", "resp_pkts", "spkts", "dpkts",
        "total_pkts", "pkts", "packets", "packet_count"
    )]
    dur_cols = [c for c in df.columns if c.lower() in (
        "duration", "dur", "flow_duration", "conn_duration"
    )]
    if not pkt_cols or not dur_cols:
        return df

    total_pkts = df[pkt_cols].sum(axis=1).astype(float)
    duration = df[dur_cols[0]].replace(0, np.nan).astype(float)
    df["packets_per_second"] = (total_pkts / duration).fillna(0)
    _register("packets_per_second", "Packets per second of connection duration; DDoS traffic often shows very high values.")
    return df


def add_bytes_per_second(df: pd.DataFrame) -> pd.DataFrame:
    """bytes_per_second = total_bytes / duration"""
    byte_cols = [c for c in df.columns if c.lower() in (
        "orig_bytes", "resp_bytes", "src_bytes", "dst_bytes",
        "sbytes", "dbytes", "totbytes", "total_bytes", "bytes"
    )]
    dur_cols = [c for c in df.columns if c.lower() in (
        "duration", "dur", "flow_duration", "conn_duration"
    )]
    if not byte_cols or not dur_cols:
        return df

    total_bytes = df[byte_cols].sum(axis=1).astype(float)
    duration = df[dur_cols[0]].replace(0, np.nan).astype(float)
    df["bytes_per_second"] = (total_bytes / duration).fillna(0)
    _register("bytes_per_second", "Bytes per second; anomalously high values may signal data floods or exfiltration.")
    return df


def add_src_dst_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """src_dst_byte_ratio = src_bytes / dst_bytes"""
    src_cols = [c for c in df.columns if c.lower() in (
        "orig_bytes", "src_bytes", "sbytes"
    )]
    dst_cols = [c for c in df.columns if c.lower() in (
        "resp_bytes", "dst_bytes", "dbytes"
    )]
    if not src_cols or not dst_cols:
        return df

    src = df[src_cols[0]].astype(float)
    dst = df[dst_cols[0]].replace(0, np.nan).astype(float)
    df["src_dst_byte_ratio"] = (src / dst).fillna(0)
    _register("src_dst_byte_ratio", "Ratio of source to destination bytes; extreme ratios indicate one-directional traffic (scan, exfil).")
    return df


def add_src_dst_pkt_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """src_dst_pkt_ratio = src_pkts / dst_pkts"""
    src_cols = [c for c in df.columns if c.lower() in (
        "orig_pkts", "spkts", "src_pkts"
    )]
    dst_cols = [c for c in df.columns if c.lower() in (
        "resp_pkts", "dpkts", "dst_pkts"
    )]
    if not src_cols or not dst_cols:
        return df

    src = df[src_cols[0]].astype(float)
    dst = df[dst_cols[0]].replace(0, np.nan).astype(float)
    df["src_dst_pkt_ratio"] = (src / dst).fillna(0)
    _register("src_dst_pkt_ratio", "Ratio of source to destination packets; imbalanced ratios may signal scanning or DoS.")
    return df


def add_is_privileged_port(df: pd.DataFrame) -> pd.DataFrame:
    """Flag connections on well-known ports (<1024)."""
    port_cols = [c for c in df.columns if c.lower() in (
        "id.resp_p", "dport", "dst_port", "dest_port", "resp_p",
        "id.orig_p", "sport", "src_port", "orig_p"
    )]
    for pc in port_cols:
        safe_name = pc.replace(".", "_")
        col_name = f"is_privileged_{safe_name}"
        try:
            df[col_name] = (pd.to_numeric(df[pc], errors="coerce").fillna(9999) < 1024).astype(int)
            _register(col_name, f"1 if port {pc} < 1024 (well-known/privileged port), 0 otherwise.")
        except Exception:
            pass
    return df


def add_connection_state_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary flags for common connection states."""
    state_cols = [c for c in df.columns if c.lower() in (
        "conn_state", "state", "connection_state"
    )]
    if not state_cols:
        return df

    sc = state_cols[0]
    states = df[sc].astype(str).str.upper()
    important_states = ["S0", "SF", "REJ", "RSTO", "RSTR", "SH", "SHR", "OTH"]
    for st in important_states:
        col_name = f"state_{st}"
        df[col_name] = (states == st).astype(int)
        _register(col_name, f"1 if connection state is {st}.")
    return df


# ── Master engineer function ────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature-engineering steps on *df*.

    Only adds features whose required raw columns exist in the data.
    Returns the augmented DataFrame.
    """
    FEATURE_DOCS.clear()  # reset docs for each run

    df = add_bytes_per_packet(df)
    df = add_packets_per_second(df)
    df = add_bytes_per_second(df)
    df = add_src_dst_ratio(df)
    df = add_src_dst_pkt_ratio(df)
    df = add_is_privileged_port(df)
    df = add_connection_state_flags(df)

    # Replace inf / NaN introduced by division
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    n_eng = len(FEATURE_DOCS)
    logger.info("Engineered %d new features: %s", n_eng, list(FEATURE_DOCS.keys()))
    return df


def get_feature_docs() -> dict[str, str]:
    """Return the documentation dict for the most recent engineering run."""
    return dict(FEATURE_DOCS)
