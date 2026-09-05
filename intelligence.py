"""
intelligence.py
================
Advanced analytics layer for the NIDS. Adds four capabilities on top of the
base XGBoost + Isolation Forest hybrid detector:

1. Explainer          - SHAP-grounded per-alert feature attribution, so the
                         LLM remediation report and dashboard cite *why* a
                         flow was flagged instead of just asserting it.
2. AttackFingerprinter - Online clustering ("Attack DNA") of confirmed-attack
                         feature vectors into recurring behavioral families,
                         so repeat campaigns are recognized across time and
                         across different source IPs.
3. build_narratives    - Turns a flat telemetry buffer into per-source-IP
                         incident timelines (contact -> recon -> escalation
                         -> confirmed attack -> contained).
4. DriftMonitor        - Compares live feature distributions against the
                         training-time baseline to surface data/model drift
                         before it silently degrades detection quality.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger("nids_intelligence")

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


# --------------------------------------------------------------------------- #
# 1. Explainability (SHAP)
# --------------------------------------------------------------------------- #
class Explainer:
    """Wraps a SHAP TreeExplainer around the XGBoost model for per-alert
    feature attribution. This grounds both the dashboard UI and the LLM
    remediation prompt in real evidence instead of free-form generation."""

    def __init__(self) -> None:
        self._explainer = None
        self._feature_names: List[str] = []

    def fit(self, xgb_model: Any, feature_names: List[str]) -> None:
        self._feature_names = list(feature_names)
        if shap is None:
            logger.warning("shap package not installed; explanations disabled.")
            return
        try:
            self._explainer = shap.TreeExplainer(xgb_model)
            logger.info("SHAP TreeExplainer initialized for %d features.", len(feature_names))
        except Exception:
            logger.exception("Failed to initialize SHAP TreeExplainer")
            self._explainer = None

    @property
    def is_ready(self) -> bool:
        return self._explainer is not None

    def top_features(self, df_row: pd.DataFrame, k: int = 5) -> List[Dict[str, Any]]:
        """Return the top-k features driving this single-row prediction,
        signed by direction of contribution (positive = pushes toward attack)."""
        if not self.is_ready:
            return []
        try:
            raw = self._explainer.shap_values(df_row)
            values = np.array(raw[1] if isinstance(raw, list) and len(raw) > 1 else raw)
            values = values[0] if values.ndim > 1 else values
        except Exception:
            logger.exception("SHAP computation failed")
            return []

        contributions = list(zip(self._feature_names, df_row.iloc[0].tolist(), values.tolist()))
        contributions.sort(key=lambda t: abs(t[2]), reverse=True)
        return [
            {"feature": name, "value": round(float(val), 4), "impact": round(float(shap_val), 4)}
            for name, val, shap_val in contributions[:k]
        ]


# --------------------------------------------------------------------------- #
# 2. Attack DNA - behavioral clustering of confirmed attacks
# --------------------------------------------------------------------------- #
ATTACK_FAMILY_NAMES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel"]


class AttackFingerprinter:
    """Clusters confirmed-attack feature vectors online (MiniBatchKMeans) so
    the system can recognize recurring 'attack families' across time,
    instead of treating every new alert as an unrelated event."""

    def __init__(self, n_clusters: int = 6, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self._model = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=3, batch_size=32
        )
        self._warm = False
        self._warmup_buffer: List[np.ndarray] = []
        self._cluster_examples: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def observe(self, feature_vector: np.ndarray, context: Dict[str, Any]) -> int:
        """Feed one confirmed-attack feature vector, return its cluster id."""
        with self._lock:
            x = np.asarray(feature_vector, dtype=float).reshape(1, -1)

            if not self._warm:
                self._warmup_buffer.append(x[0])
                if len(self._warmup_buffer) < max(self.n_clusters * 2, 8):
                    # Not enough attack samples yet for stable clustering.
                    cluster_id = len(self._warmup_buffer) % self.n_clusters
                    self._record(cluster_id, context)
                    return cluster_id
                # Enough samples collected: bootstrap the model in one shot.
                self._model.partial_fit(np.vstack(self._warmup_buffer))
                self._warm = True

            self._model.partial_fit(x)
            cluster_id = int(self._model.predict(x)[0])
            self._record(cluster_id, context)
            return cluster_id

    def _record(self, cluster_id: int, context: Dict[str, Any]) -> None:
        entry = self._cluster_examples.setdefault(
            cluster_id, {"count": 0, "services": defaultdict(int), "flags": defaultdict(int)}
        )
        entry["count"] += 1
        entry["services"][context.get("service", "other")] += 1
        entry["flags"][context.get("flag", "SF")] += 1

    def family_name(self, cluster_id: int) -> str:
        return f"Family-{ATTACK_FAMILY_NAMES[cluster_id % len(ATTACK_FAMILY_NAMES)]}"

    def summary(self) -> List[Dict[str, Any]]:
        with self._lock:
            out = []
            for cid, entry in sorted(self._cluster_examples.items()):
                dom_service = max(entry["services"].items(), key=lambda kv: kv[1])[0] if entry["services"] else "n/a"
                dom_flag = max(entry["flags"].items(), key=lambda kv: kv[1])[0] if entry["flags"] else "n/a"
                out.append(
                    {
                        "cluster_id": cid,
                        "family_name": self.family_name(cid),
                        "observed_count": entry["count"],
                        "dominant_service": dom_service,
                        "dominant_flag": dom_flag,
                    }
                )
            return out


# --------------------------------------------------------------------------- #
# 3. Incident narrative timelines
# --------------------------------------------------------------------------- #
STAGE_LABELS = {
    "contact": "🟢 First Contact",
    "recon": "🔍 Reconnaissance / Probing",
    "escalation": "⚠️ Escalation",
    "anomaly": "🟡 Statistical Anomaly",
    "confirmed_attack": "🚨 Confirmed Attack",
}


def _classify_stage(event: Dict[str, Any]) -> str:
    """Classify a single traffic event into a narrative stage label."""
    is_attack = bool(event.get("is_attack", False))
    if not is_attack:
        return "contact"
    flag = event.get("flag", "SF")
    xgb_hit = event.get("xgb_prediction") == 1
    iso_hit = event.get("iso_prediction") == 1
    if flag in ("REJ", "S0") and not xgb_hit:
        return "recon"
    if xgb_hit and iso_hit:
        return "confirmed_attack"
    if xgb_hit:
        return "escalation"
    return "anomaly"


def build_narratives(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group a flat list of telemetry events into per-source-IP incident
    narratives/timelines, ordered by risk (most attacks first)."""
    by_ip: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        by_ip[ev.get("src_ip", "Unknown")].append(ev)

    narratives = []
    for src_ip, evs in by_ip.items():
        evs_sorted = sorted(evs, key=lambda e: e.get("timestamp", ""))
        stages = []
        for ev in evs_sorted:
            stage = _classify_stage(ev)
            stages.append(
                {
                    "timestamp": ev.get("timestamp"),
                    "stage": stage,
                    "label": STAGE_LABELS.get(stage, stage),
                    "dst_ip": ev.get("dst_ip"),
                    "service": ev.get("service"),
                    "flag": ev.get("flag"),
                    "attack_family": ev.get("attack_family"),
                }
            )
        attack_count = sum(1 for e in evs_sorted if e.get("is_attack"))
        risk = "critical" if attack_count >= 5 else "high" if attack_count >= 2 else "low" if attack_count == 1 else "none"
        narratives.append(
            {
                "src_ip": src_ip,
                "event_count": len(evs_sorted),
                "attack_count": attack_count,
                "first_seen": evs_sorted[0].get("timestamp") if evs_sorted else None,
                "last_seen": evs_sorted[-1].get("timestamp") if evs_sorted else None,
                "stages": stages,
                "risk_level": risk,
            }
        )

    narratives.sort(key=lambda n: n["attack_count"], reverse=True)
    return narratives


# --------------------------------------------------------------------------- #
# 4. Model health / drift monitoring
# --------------------------------------------------------------------------- #
class DriftMonitor:
    """Compares live feature distributions against the training-time
    baseline (mean/std, captured in feature_schema.json) and tracks the
    live prediction mix, so operators can see model/data health without
    digging through raw logs."""

    def __init__(self, baseline_stats: Dict[str, Dict[str, float]] | None = None) -> None:
        self.baseline_stats: Dict[str, Dict[str, float]] = baseline_stats or {}
        self._live_sums: Dict[str, float] = defaultdict(float)
        self._live_count = 0
        self._pred_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def set_baseline(self, baseline_stats: Dict[str, Dict[str, float]]) -> None:
        self.baseline_stats = baseline_stats or {}

    def observe(self, numeric_row: pd.Series, is_attack: bool) -> None:
        with self._lock:
            self._live_count += 1
            for col, val in numeric_row.items():
                try:
                    self._live_sums[col] += float(val)
                except (TypeError, ValueError):
                    continue
            self._pred_counts["attack" if is_attack else "normal"] += 1

    def report(self) -> Dict[str, Any]:
        with self._lock:
            n = self._live_count
            drift_rows = []
            if n > 0:
                for feat, base in self.baseline_stats.items():
                    if feat not in self._live_sums:
                        continue
                    live_mean = self._live_sums[feat] / n
                    base_mean = base.get("mean", 0.0)
                    base_std = base.get("std", 0.0) or 1e-6
                    z = (live_mean - base_mean) / base_std
                    drift_rows.append(
                        {
                            "feature": feat,
                            "baseline_mean": round(base_mean, 4),
                            "live_mean": round(live_mean, 4),
                            "drift_z_score": round(z, 3),
                            "flag": "drift" if abs(z) > 3 else "ok",
                        }
                    )
                drift_rows.sort(key=lambda r: abs(r["drift_z_score"]), reverse=True)

            total_preds = sum(self._pred_counts.values())
            return {
                "samples_observed": n,
                "prediction_mix": dict(self._pred_counts),
                "attack_rate": round(self._pred_counts.get("attack", 0) / total_preds, 4) if total_preds else 0.0,
                "baseline_available": bool(self.baseline_stats),
                "drift": drift_rows[:10],
            }
