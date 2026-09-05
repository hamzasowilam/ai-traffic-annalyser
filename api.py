import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Security,
    UploadFile,
    status,
)
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from scapy.all import IP, TCP, UDP, rdpcap
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from intelligence import AttackFingerprinter, DriftMonitor, Explainer, build_narratives


# ===========================================================
# Configuration
# ===========================================================
class Settings(BaseSettings):
    artifacts_dir: Path = Path("ml/artifacts")
    xgb_model_file: str = "xgboost_traffic_model.pkl"
    iso_model_file: str = "isolation_forest_model.pkl"
    encoder_file: str = "encoder.pkl"
    schema_file: str = "feature_schema.json"

    xgb_model_sha256: Optional[str] = None
    iso_model_sha256: Optional[str] = None
    encoder_sha256: Optional[str] = None

    api_key: Optional[str] = None
    require_auth: bool = False

    rate_limit: str = "200/minute"
    max_features: int = 200

    # SECURITY: never hardcode secrets here. Set via env var NIDS_GROQ_API_KEY
    # (or a .env file loaded by pydantic-settings) or the GROQ_API_KEY fallback
    # read in RemediationService below. A previous version of this file shipped
    # a live key as the default value here — if that key was ever committed to
    # a repo, treat it as compromised and rotate it in the Groq console.
    groq_api_key: Optional[str] = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model_cache_ttl_seconds: int = 3600

    # Attack DNA clustering
    attack_fingerprint_clusters: int = 6

    log_level: str = "INFO"
    model_config = SettingsConfigDict(env_prefix="NIDS_")


settings = Settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("nids_api")

if not settings.require_auth:
    logger.warning(
        "NIDS_REQUIRE_AUTH is not enabled — the API is running WITHOUT authentication. "
        "Set NIDS_REQUIRE_AUTH=true and NIDS_API_KEY=<secret> before exposing this "
        "service beyond localhost."
    )


# ===========================================================
# In-Memory Telemetry Ring Buffer (Thread-Safe)
# ===========================================================
MAX_LOG_BUFFER = 500
traffic_logs_lock = threading.Lock()
traffic_logs: Deque[Dict[str, Any]] = deque(maxlen=MAX_LOG_BUFFER)


def record_traffic_event(
    src_ip: str,
    dst_ip: str,
    service: str,
    flag: str,
    pred_result: Dict[str, Any],
) -> None:
    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "service": service,
        "flag": flag,
        "status": pred_result.get("status", "Unknown"),
        "is_attack": pred_result.get("is_attack", False),
        "xgb_prediction": pred_result.get("xgb_prediction", 0),
        "iso_prediction": pred_result.get("iso_prediction", 0),
        "top_features": pred_result.get("top_features", []),
        "attack_cluster": pred_result.get("attack_cluster"),
        "attack_family": pred_result.get("attack_family"),
    }
    with traffic_logs_lock:
        traffic_logs.append(event)


# ===========================================================
# Custom Exceptions & Verification
# ===========================================================
class SchemaValidationError(ValueError):
    """Raised when payload does not match schema."""


class ArtifactIntegrityError(RuntimeError):
    """Raised when SHA-256 verification fails."""


def _verify_checksum(path: Path, expected_sha256: Optional[str]) -> None:
    if not expected_sha256:
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ArtifactIntegrityError(
            f"Checksum mismatch for {path.name}: expected {expected_sha256}, got {digest}"
        )


# ===========================================================
# ML Model Service
# ===========================================================
class ModelService:
    def __init__(self) -> None:
        self.xgb = None
        self.iso = None
        self.encoder = None
        self.feature_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self._loaded = False

        # Intelligence layer: explainability, attack clustering, drift.
        self.explainer = Explainer()
        self.fingerprinter = AttackFingerprinter(n_clusters=settings.attack_fingerprint_clusters)
        self.drift_monitor = DriftMonitor()

    def load(self) -> None:
        d = settings.artifacts_dir
        xgb_path = d / settings.xgb_model_file
        iso_path = d / settings.iso_model_file
        enc_path = d / settings.encoder_file
        schema_path = d / settings.schema_file

        for p in (xgb_path, iso_path, enc_path, schema_path):
            if not p.is_file():
                raise FileNotFoundError(f"Required artifact missing: {p}")

        _verify_checksum(xgb_path, settings.xgb_model_sha256)
        _verify_checksum(iso_path, settings.iso_model_sha256)
        _verify_checksum(enc_path, settings.encoder_sha256)

        self.xgb = joblib.load(xgb_path)
        self.iso = joblib.load(iso_path)
        self.encoder = joblib.load(enc_path)

        with open(schema_path, "r") as f:
            schema = json.load(f)

        try:
            self.feature_columns = list(schema["feature_columns"])
            self.categorical_columns = list(schema.get("categorical_columns", []))
        except KeyError as e:
            raise SchemaValidationError(f"feature_schema.json missing key: {e}") from e

        if not self.feature_columns:
            raise SchemaValidationError("feature_schema.json has no feature_columns")

        # Wire up the intelligence layer now that the model + schema exist.
        self.explainer.fit(self.xgb, self.feature_columns)
        baseline_stats = schema.get("baseline_stats", {})
        if not baseline_stats:
            logger.warning(
                "feature_schema.json has no baseline_stats — retrain with the "
                "updated ml/train.py to enable model-health drift monitoring."
            )
        self.drift_monitor.set_baseline(baseline_stats)

        self._loaded = True

    def unload(self) -> None:
        self.xgb = self.iso = self.encoder = None
        self.feature_columns = []
        self.categorical_columns = []
        self._loaded = False

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def _build_dataframe(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(rows)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = "other" if col in self.categorical_columns else 0

        df = df[self.feature_columns]

        if self.categorical_columns and self.encoder is not None:
            try:
                df[self.categorical_columns] = self.encoder.transform(
                    df[self.categorical_columns]
                )
            except ValueError as e:
                raise SchemaValidationError(f"Invalid categorical value: {e}") from e

        numeric_df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        if np.isinf(numeric_df.values).any():
            raise SchemaValidationError("Input contains Inf values")

        return numeric_df

    def predict(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.is_ready:
            raise RuntimeError("Model service is not ready")

        df = self._build_dataframe(rows)

        xgb_preds = self.xgb.predict(df).astype(int)
        iso_raw = self.iso.predict(df).astype(int)
        iso_preds = np.where(iso_raw == -1, 1, 0)

        results = []
        for i, (xgb_p, iso_p) in enumerate(zip(xgb_preds, iso_preds)):
            is_attack = bool(xgb_p == 1 or iso_p == 1)
            row_df = df.iloc[[i]]

            # Model-health / drift telemetry (every row, attack or not).
            self.drift_monitor.observe(row_df.iloc[0], is_attack)

            result: Dict[str, Any] = {
                "is_attack": is_attack,
                "status": "Attack Detected!" if is_attack else "Normal Traffic",
                "xgb_prediction": int(xgb_p),
                "iso_prediction": int(iso_p),
                "top_features": [],
                "attack_cluster": None,
                "attack_family": None,
            }

            if is_attack:
                # SHAP explainability: ground the alert in real feature attribution.
                result["top_features"] = self.explainer.top_features(row_df, k=5)

                # Attack DNA: cluster this confirmed attack's feature vector.
                original_row = rows[i] if i < len(rows) else {}
                cluster_id = self.fingerprinter.observe(
                    row_df.iloc[0].to_numpy(),
                    context={
                        "service": original_row.get("service", "other"),
                        "flag": original_row.get("flag", "SF"),
                    },
                )
                result["attack_cluster"] = cluster_id
                result["attack_family"] = self.fingerprinter.family_name(cluster_id)

            results.append(result)
        return results


model_service = ModelService()


# ===========================================================
# LLM Remediation Service
# ===========================================================
class RemediationService:
    PREFERRED_MODEL_PREFIXES = ("llama-3.3", "llama-3.1", "deepseek-r1", "llama3")

    def __init__(self) -> None:
        self.client: Optional[Any] = None
        self._cached_model_id: Optional[str] = None
        self._cache_loaded_at: float = 0.0
        self._cache_ttl = settings.groq_model_cache_ttl_seconds
        self._lock = threading.Lock()
        self.api_key = settings.groq_api_key or os.environ.get("GROQ_API_KEY")

        if not self.api_key or OpenAI is None:
            return

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=settings.groq_base_url,
                max_retries=0,
            )
            logger.info("Groq-compatible LLM client initialized.")
        except Exception:
            self.client = None

    def _select_model_from_list(self, model_ids: List[str]) -> Optional[str]:
        for prefix in self.PREFERRED_MODEL_PREFIXES:
            for model_id in model_ids:
                if model_id.startswith(prefix):
                    return model_id
        return model_ids[0] if model_ids else None

    def _refresh_active_model(self, force: bool = False) -> Optional[str]:
        if self.client is None:
            return None

        now = time.time()
        with self._lock:
            if not force and self._cached_model_id and (now - self._cache_loaded_at) < self._cache_ttl:
                return self._cached_model_id

            try:
                response = self.client.models.list()
                model_ids = [m.id for m in response.data]
            except Exception as e:
                logger.error("Could not query Groq /models endpoint: %s", e)
                return self._cached_model_id

            selected = self._select_model_from_list(model_ids)
            if selected:
                self._cached_model_id = selected
                self._cache_loaded_at = now
            return self._cached_model_id

    @staticmethod
    def _deterministic_report(src_ip: str, dst_ip: str, service: str, flag: str) -> str:
        return (
            f"[Automated Remediation - Fallback] Attack detected from {src_ip} "
            f"targeting {service} (flag: {flag}).\n\n"
            f"**Immediate Remediation**\n"
            f"- Linux: `sudo iptables -A INPUT -s {src_ip} -j DROP`\n"
            f"- Windows: `netsh advfirewall firewall add rule name=\"Block_{src_ip}\" "
            f"dir=in action=block remoteip={src_ip}`\n"
        )

    def _build_prompt(self, src_ip: str, dst_ip: str, service: str, flag: str, details: Dict[str, Any]) -> str:
        top_features = details.get("top_features") or []
        if top_features:
            evidence_lines = "\n".join(
                f"  - {f.get('feature')}: value={f.get('value')}, SHAP impact={f.get('impact')} "
                f"({'pushes toward ATTACK' if f.get('impact', 0) > 0 else 'pushes toward normal'})"
                for f in top_features
            )
            evidence_block = f"- Model Evidence (SHAP top contributing features):\n{evidence_lines}"
        else:
            evidence_block = "- Model Evidence: not available for this alert."

        attack_family = details.get("attack_family")
        family_line = f"- Behavioral Cluster: {attack_family} (matches a previously observed attack pattern)\n" if attack_family else ""

        return f"""You are an elite Cybersecurity Incident Response Specialist.
An active network intrusion attempt was intercepted by our ML-NIDS:
- Attacker IP (Source): {src_ip}
- Destination Target: {dst_ip}
- Service Targeted: {service}
- TCP/UDP Flag: {flag}
{family_line}{evidence_block}
- Raw Detection Info: {json.dumps(details)}

Ground your analysis in the SHAP evidence above where possible — reference the
specific feature(s) that drove the detection rather than speaking generically.
Provide a concise, professional threat mitigation summary:
1. Threat Analysis: Likely attack vector and immediate risk, referencing the evidence.
2. Immediate Remediation: Exact firewall command (iptables for Linux and netsh for Windows) to block the attacker IP immediately.
3. Hardening Advice: 1 short actionable recommendation.
Keep the response formatted strictly in markdown."""

    def _call_llm(self, model_id: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a professional Cyber Defense Assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()

    def generate_analysis(
        self, src_ip: str, dst_ip: str, service: str, flag: str, details: Dict[str, Any]
    ) -> str:
        if self.client is None:
            return self._deterministic_report(src_ip, dst_ip, service, flag)

        model_id = self._refresh_active_model()
        if not model_id:
            return self._deterministic_report(src_ip, dst_ip, service, flag)

        prompt = self._build_prompt(src_ip, dst_ip, service, flag, details)
        try:
            return self._call_llm(model_id, prompt)
        except Exception as exc:
            logger.warning("LLM remediation failed for %s -> %s: %s", src_ip, dst_ip, exc)
            try:
                new_model_id = self._refresh_active_model(force=True)
                if new_model_id and new_model_id != model_id:
                    return self._call_llm(new_model_id, prompt)
            except Exception as retry_exc:
                logger.warning("LLM remediation retry failed for %s -> %s: %s", src_ip, dst_ip, retry_exc)
            return self._deterministic_report(src_ip, dst_ip, service, flag)


remediation_service = RemediationService()


# ===========================================================
# Webhook Alert
# ===========================================================
def send_soc_alert(src_ip: str, dst_ip: str, report: str) -> None:
    webhook_url = os.environ.get("NIDS_WEBHOOK_URL", "")
    if not webhook_url:
        return
    payload = {
        "content": f"🚨 **CRITICAL NIDS ALERT** 🚨\n**Attacker IP:** `{src_ip}` ➡️ **Target IP:** `{dst_ip}`\n\n**AI Report:**\n```text\n{report}\n```"
    }
    try:
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception as e:
        logger.error("Failed to send SOC alert webhook: %s", e)


# ===========================================================
# Lifespan
# ===========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML artifacts...")
    try:
        model_service.load()
        logger.info("All artifacts loaded successfully.")
    except Exception:
        logger.exception("Failed to load artifacts")
        raise
    yield
    model_service.unload()
    logger.info("Artifacts cleared from memory.")


app = FastAPI(
    title="NIDS Hybrid Prediction API with Telemetry Feed",
    description=(
        "API for detecting intrusions and serving real-time telemetry logs, "
        "with SHAP explainability, attack-family clustering, incident "
        "narrative timelines, and model-health drift monitoring."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str = Security(_api_key_header)) -> None:
    if not settings.require_auth:
        return
    if not settings.api_key:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Service misconfigured")
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")


# ===========================================================
# Schemas
# ===========================================================
class TrafficData(BaseModel):
    features: Dict[str, Any]
    src_ip: Optional[str] = "203.0.113.5"
    dst_ip: Optional[str] = "192.168.1.50"

    @field_validator("features")
    @classmethod
    def _limit_feature_count(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if len(v) > settings.max_features:
            raise ValueError(f"Too many feature keys ({len(v)}); max is {settings.max_features}")
        return v


class BatchTrafficData(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1, max_length=500)


class FeatureAttribution(BaseModel):
    feature: str
    value: float
    impact: float


class PredictionResponse(BaseModel):
    is_attack: bool
    status: str
    xgb_prediction: int
    iso_prediction: int
    top_features: List[FeatureAttribution] = []
    attack_cluster: Optional[int] = None
    attack_family: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]


class ThreatAnalysisRequest(BaseModel):
    src_ip: str
    dst_ip: str
    service: str = "other"
    flag: str = "SF"
    detection_details: Dict[str, Any] = {}


class ThreatAnalysisResponse(BaseModel):
    threat_analysis_report: str


# ===========================================================
# Endpoints
# ===========================================================
@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "active" if model_service.is_ready else "degraded",
        "model_loaded": model_service.is_ready,
        "remediation_llm_configured": remediation_service.client is not None,
    }


@app.get("/logs", dependencies=[Depends(require_api_key)])
def get_traffic_logs() -> List[Dict[str, Any]]:
    """Returns the latest in-memory traffic detection buffer for the Streamlit dashboard."""
    with traffic_logs_lock:
        return list(traffic_logs)


@app.delete("/logs", dependencies=[Depends(require_api_key)])
def clear_traffic_logs() -> Dict[str, str]:
    """Clears the live telemetry buffer."""
    with traffic_logs_lock:
        traffic_logs.clear()
    return {"message": "Telemetry logs cleared."}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(settings.rate_limit)
def predict_traffic(request: Request, data: TrafficData) -> Dict[str, Any]:
    try:
        result = model_service.predict([data.features])[0]
        
        # Ingest into live telemetry log buffer
        record_traffic_event(
            src_ip=data.src_ip or "Unknown",
            dst_ip=data.dst_ip or "Local",
            service=data.features.get("service", "other"),
            flag=data.features.get("flag", "SF"),
            pred_result=result,
        )
        return result
    except SchemaValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Unhandled prediction error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(settings.rate_limit)
def predict_traffic_batch(request: Request, data: BatchTrafficData) -> Dict[str, Any]:
    try:
        results = model_service.predict(data.records)
        for row, res in zip(data.records, results):
            record_traffic_event(
                src_ip=row.get("src_ip", "BatchSource"),
                dst_ip=row.get("dst_ip", "TargetHost"),
                service=row.get("service", "other"),
                flag=row.get("flag", "SF"),
                pred_result=res,
            )
        return {"results": results}
    except SchemaValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Unhandled batch prediction error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/remediate", response_model=ThreatAnalysisResponse)
def generate_remediation(payload: ThreatAnalysisRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    report = remediation_service.generate_analysis(
        src_ip=payload.src_ip,
        dst_ip=payload.dst_ip,
        service=payload.service,
        flag=payload.flag,
        details=payload.detection_details,
    )
    background_tasks.add_task(send_soc_alert, payload.src_ip, payload.dst_ip, report)
    return {"threat_analysis_report": report}


# ===========================================================
# Intelligence Endpoints: Explainability, Attack DNA, Timelines, Drift
# ===========================================================
@app.get("/timeline", dependencies=[Depends(require_api_key)])
def get_all_timelines() -> List[Dict[str, Any]]:
    """Per-source-IP incident narratives built from the live telemetry
    buffer: first contact -> recon -> escalation -> confirmed attack."""
    with traffic_logs_lock:
        events = list(traffic_logs)
    return build_narratives(events)


@app.get("/timeline/{src_ip}", dependencies=[Depends(require_api_key)])
def get_timeline_for_ip(src_ip: str) -> Dict[str, Any]:
    with traffic_logs_lock:
        events = [e for e in traffic_logs if e.get("src_ip") == src_ip]
    narratives = build_narratives(events)
    if not narratives:
        raise HTTPException(status_code=404, detail=f"No telemetry recorded for {src_ip}")
    return narratives[0]


@app.get("/attack-families", dependencies=[Depends(require_api_key)])
def get_attack_families() -> List[Dict[str, Any]]:
    """Attack DNA: behavioral clusters discovered among confirmed attacks,
    so recurring campaigns are recognized instead of treated as isolated
    one-off events."""
    return model_service.fingerprinter.summary()


@app.get("/model/health", dependencies=[Depends(require_api_key)])
def get_model_health() -> Dict[str, Any]:
    """Model-health / drift report: live feature distributions vs. the
    training-time baseline, plus the live prediction mix."""
    report = model_service.drift_monitor.report()
    report["model_loaded"] = model_service.is_ready
    report["explainability_available"] = model_service.explainer.is_ready
    report["attack_families_discovered"] = len(model_service.fingerprinter.summary())
    return report


@app.post(
    "/predict/pcap",
    dependencies=[Depends(require_api_key)],
)
async def analyze_pcap_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.endswith(".pcap"):
        raise HTTPException(status_code=400, detail="Only .pcap files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        packets = rdpcap(tmp_path)
        records = []

        for pkt in packets:
            if IP in pkt:
                service = "other"
                flag = "SF"
                if TCP in pkt:
                    flags = str(pkt[TCP].flags)
                    flag = "S0" if "S" in flags and "A" not in flags else "SF"
                    service = "http" if pkt[TCP].dport in [80, 443] else "other"

                records.append({
                    "service": service,
                    "flag": flag,
                    "src_bytes": len(pkt[IP].payload),
                    "dst_bytes": 0,
                    "count": 1,
                    "srv_count": 1,
                    "serror_rate": 1.0 if flag == "S0" else 0.0,
                    "same_srv_rate": 1.0,
                    "src_ip": str(pkt[IP].src),
                    "dst_ip": str(pkt[IP].dst),
                })

        if not records:
            return {"total_packets": 0, "attacks_found": 0, "results_sample": []}

        results = model_service.predict(records)
        attacks_count = 0
        for rec, res in zip(records, results):
            if res["is_attack"]:
                attacks_count += 1
            record_traffic_event(
                src_ip=rec["src_ip"],
                dst_ip=rec["dst_ip"],
                service=rec["service"],
                flag=rec["flag"],
                pred_result=res,
            )

        return {
            "total_packets": len(records),
            "attacks_found": attacks_count,
            "results_sample": results[:100],
        }
    except Exception as e:
        logger.exception("PCAP processing failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
