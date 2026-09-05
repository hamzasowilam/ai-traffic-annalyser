"""
AI-Powered NIDS & SOC Terminal Dashboard (Integrated Live Backend Sync).
"""

from __future__ import annotations

import ipaddress
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------------------------------------------------------- #
# Config & Secrets
# --------------------------------------------------------------------------- #
def _resolve_api_key() -> Optional[str]:
    api_key = os.environ.get("NIDS_DASHBOARD_API_KEY")
    if api_key:
        return api_key
    try:
        if hasattr(st, "secrets") and "NIDS_API_KEY" in st.secrets:
            return str(st.secrets["NIDS_API_KEY"])
    except Exception:
        pass
    return None


API_URL: str = os.environ.get("NIDS_DASHBOARD_API_URL", "http://127.0.0.1:8000")
API_KEY: Optional[str] = _resolve_api_key()
PREDICT_TIMEOUT: float = float(os.environ.get("NIDS_DASHBOARD_PREDICT_TIMEOUT", "5.0"))
REMEDIATE_TIMEOUT: float = float(os.environ.get("NIDS_DASHBOARD_REMEDIATE_TIMEOUT", "25.0"))

SAFE_IPS = {"127.0.0.1", "::1", "192.168.1.1", "192.168.1.50", "localhost"}

st.set_page_config(
    page_title="SOC Cyber Defense Terminal",
    page_icon="⚡",
    layout="wide",
)

HACKER_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600;700&family=Orbitron:wght@500;700;900&display=swap');

* {
    font-family: 'Fira Code', monospace !important;
}
h1, h2, h3, .stTitle {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 1.5px;
}
.stApp {
    background-color: #060a0f;
    color: #c9d1d9;
}
[data-testid="stSidebar"] {
    background-color: #0b1118;
    border-right: 1px solid #00ff6633;
}
h1 {
    color: #00ff66 !important;
    text-shadow: 0 0 10px rgba(0, 255, 102, 0.4), 0 0 20px rgba(0, 255, 102, 0.2);
}
h2, h3, h4 {
    color: #00e5ff !important;
}
.stTextInput input, .stNumberInput input, div[data-baseweb="select"] > div {
    background-color: #0d1520 !important;
    color: #00ff66 !important;
    border: 1px solid #00ff6655 !important;
    border-radius: 4px;
}
.stTextInput input:focus, div[data-baseweb="select"] > div:focus-within {
    border-color: #00ff66 !important;
    box-shadow: 0 0 8px rgba(0, 255, 102, 0.6) !important;
}
button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, #0d261a 0%, #05140d 100%) !important;
    color: #00ff66 !important;
    border: 1px solid #00ff66 !important;
    box-shadow: 0 0 10px rgba(0, 255, 102, 0.2);
    transition: all 0.3s ease;
    font-weight: 700;
    text-transform: uppercase;
}
button[kind="primary"]:hover, .stButton > button:hover {
    background: #00ff66 !important;
    color: #000000 !important;
    box-shadow: 0 0 20px rgba(0, 255, 102, 0.8);
}
[data-testid="stMetric"] {
    background-color: #0b141d;
    border: 1px solid #00e5ff44;
    border-radius: 6px;
    padding: 10px 15px;
    box-shadow: 0 0 8px rgba(0, 229, 255, 0.1);
}
[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-size: 0.85rem !important;
}
[data-testid="stMetricValue"] {
    color: #00e5ff !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 1.5rem !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #00ff6633;
}
.stTabs [data-baseweb="tab"] {
    background-color: #091017;
    border: 1px solid #1f2937;
    border-radius: 4px 4px 0 0;
    color: #8b949e;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background-color: #0d1e16 !important;
    color: #00ff66 !important;
    border: 1px solid #00ff66 !important;
    border-bottom: none !important;
}
.threat-box {
    background-color: #1f0a0d;
    border: 1px solid #ff0055;
    border-left: 6px solid #ff0055;
    padding: 15px;
    border-radius: 4px;
    color: #ff3366;
    margin: 10px 0;
    box-shadow: 0 0 12px rgba(255, 0, 85, 0.2);
}
.clean-box {
    background-color: #081d11;
    border: 1px solid #00ff66;
    border-left: 6px solid #00ff66;
    padding: 15px;
    border-radius: 4px;
    color: #00ff66;
    margin: 10px 0;
    box-shadow: 0 0 12px rgba(0, 255, 102, 0.2);
}
</style>
"""
st.markdown(HACKER_THEME_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# API Client
# --------------------------------------------------------------------------- #
class APIClientError(Exception):
    pass

class APIUnreachableError(APIClientError):
    pass

class APIRequestError(APIClientError):
    pass

@st.cache_resource
def _get_session() -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=2,
        backoff_factor=0.3,
        status_forcelist=(502, 503, 504),
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    if API_KEY:
        session.headers.update({"X-API-Key": API_KEY})
    return session


def _post(path: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    session = _get_session()
    try:
        response = session.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise APIUnreachableError(f"Backend timed out after {timeout}s calling {path}.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise APIUnreachableError(f"Could not connect to backend at {API_URL}.") from exc

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        raise APIRequestError(f"{path} returned {response.status_code}") from exc

    return response.json()


@st.cache_data(ttl=3)
def fetch_health() -> Optional[Dict[str, Any]]:
    try:
        session = _get_session()
        response = session.get(f"{API_URL}/health", timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def fetch_backend_logs() -> List[Dict[str, Any]]:
    """Fetches the latest real-time packet logs collected by the sniffer."""
    try:
        session = _get_session()
        response = session.get(f"{API_URL}/logs", timeout=2)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []


def clear_backend_logs() -> None:
    try:
        session = _get_session()
        session.delete(f"{API_URL}/logs", timeout=2)
    except Exception:
        pass


@st.cache_data(ttl=4)
def fetch_timelines() -> List[Dict[str, Any]]:
    """Per-source-IP incident narratives (Attack Timeline tab)."""
    try:
        session = _get_session()
        response = session.get(f"{API_URL}/timeline", timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []


@st.cache_data(ttl=4)
def fetch_attack_families() -> List[Dict[str, Any]]:
    """Attack DNA: behavioral clusters discovered among confirmed attacks."""
    try:
        session = _get_session()
        response = session.get(f"{API_URL}/attack-families", timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []


@st.cache_data(ttl=4)
def fetch_model_health() -> Optional[Dict[str, Any]]:
    """Model-health / drift report vs. training-time baseline."""
    try:
        session = _get_session()
        response = session.get(f"{API_URL}/model/health", timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def predict_traffic(features: Dict[str, Any], src_ip: str = "203.0.113.5", dst_ip: str = "192.168.1.50") -> Dict[str, Any]:
    return _post("/predict", {"features": features, "src_ip": src_ip, "dst_ip": dst_ip}, timeout=PREDICT_TIMEOUT)


def remediate_threat(
    src_ip: str, dst_ip: str, service: str, flag: str, detection_details: Dict[str, Any]
) -> Dict[str, Any]:
    return _post(
        "/remediate",
        {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "service": service,
            "flag": flag,
            "detection_details": detection_details,
        },
        timeout=REMEDIATE_TIMEOUT,
    )


# --------------------------------------------------------------------------- #
# Firewall Actions
# --------------------------------------------------------------------------- #
def validate_ip(value: str) -> Optional[str]:
    try:
        ipaddress.ip_address(value.strip())
        return None
    except ValueError:
        return f"'{value}' is not a valid IPv4/IPv6 address."


def execute_firewall_rule(target_ip: str, action: str = "block") -> tuple[bool, str]:
    clean_ip = target_ip.strip()
    try:
        parsed_ip = str(ipaddress.ip_address(clean_ip))
    except ValueError:
        return False, f"Invalid IP format: '{clean_ip}'"

    if action == "block" and (parsed_ip in SAFE_IPS or clean_ip in SAFE_IPS):
        return False, f"Rule aborted: {parsed_ip} is protected in Static Whitelist."

    os_name = platform.system().lower()
    rule_name = f"NIDS_Block_{parsed_ip}"

    try:
        if "windows" in os_name:
            if action == "block":
                cmd = [
                    "netsh", "advfirewall", "firewall", "add", "rule",
                    f"name={rule_name}", "dir=in", "action=block", f"remoteip={parsed_ip}"
                ]
            else:
                cmd = ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule_name}"]
        else:
            flag = "-A" if action == "block" else "-D"
            cmd = ["sudo", "-n", "iptables", flag, "INPUT", "-s", parsed_ip, "-j", "DROP"]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout.strip() or f"Firewall {action} applied successfully."
    except Exception as exc:
        return False, str(exc)


# --------------------------------------------------------------------------- #
# State Model
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisRecord:
    src_ip: str
    dst_ip: str
    service: str
    flag: str
    is_attack: bool
    status: str
    xgb_prediction: int
    iso_prediction: int
    remediation_report: Optional[str] = None
    blocked: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    top_features: List[Dict[str, Any]] = field(default_factory=list)
    attack_cluster: Optional[int] = None
    attack_family: Optional[str] = None


def _init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "blocked_ips" not in st.session_state:
        st.session_state.blocked_ips = set()


_init_session_state()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown("### `[ SYSTEM_NODE_STATUS ]`")
        health = fetch_health()
        if health is None:
            st.error("❌ BACKEND OFFLINE // CHECK PORT 8000")
        elif health.get("status") == "active":
            st.success("🟢 CORE ACTIVE // FASTAPI ONLINE")
            c1, c2 = st.columns(2)
            c1.caption(f"ML: `{health.get('model_loaded')}`")
            c2.caption(f"LLM: `{health.get('remediation_llm_configured')}`")
        else:
            st.warning("⚠️ BACKEND DEGRADED")

        st.markdown("---")
        st.markdown("### `[ AUTONOMOUS_SOAR ]`")
        auto_block = st.toggle("⚡ Auto-Block Firewall", value=False)
        auto_llm = st.toggle("🤖 Auto-LLM Intel Report", value=True)
        live_polling = st.toggle("🔄 Live Auto-Refresh (5s)", value=True)

        st.markdown("---")
        st.markdown("### `[ PACKET_FORGE ]`")
        sim_service = st.selectbox("Protocol / Service", ["http", "smtp", "ftp", "dns", "private", "other"])
        sim_flag = st.selectbox("TCP Control Flags", ["SF", "S0", "REJ", "RSTR", "SH"])
        sim_src_bytes = st.number_input("Payload Src Bytes", min_value=0, value=250)
        sim_dst_bytes = st.number_input("Payload Dst Bytes", min_value=0, value=0)
        sim_count = st.number_input("Concurrent Conn Count", min_value=1, value=50)

    return {
        "service": sim_service,
        "flag": sim_flag,
        "src_bytes": sim_src_bytes,
        "dst_bytes": sim_dst_bytes,
        "count": sim_count,
        "auto_block": auto_block,
        "auto_llm": auto_llm,
        "live_polling": live_polling,
    }


# --------------------------------------------------------------------------- #
# Tab 1: Live Analysis
# --------------------------------------------------------------------------- #
def render_analysis_tab(sim: Dict[str, Any]) -> None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### `[ PACKET_INTERCEPTION_CONSOLE ]`")
        with st.form("analyze_form"):
            attacker_ip = st.text_input("Source IP (Attacker/Client)", value="203.0.113.5")
            target_ip = st.text_input("Destination IP (Target Host)", value="192.168.1.50")
            submitted = st.form_submit_button("⚡ Execute Deep Inspection", use_container_width=True)

        if submitted:
            src_err = validate_ip(attacker_ip)
            dst_err = validate_ip(target_ip)
            if src_err or dst_err:
                st.session_state.last_error = " ".join(e for e in (src_err, dst_err) if e)
            else:
                _run_analysis(attacker_ip.strip(), target_ip.strip(), sim)

    with col2:
        st.markdown("#### `[ THREAT_INSPECTION_TELEMETRY ]`")
        if st.session_state.last_error:
            st.error(st.session_state.last_error)
        elif st.session_state.history:
            _render_result(st.session_state.history[-1])
        else:
            st.info("Terminal idle. Send a packet from the console to begin inspection.")


def _run_analysis(attacker_ip: str, target_ip: str, sim: Dict[str, Any]) -> None:
    st.session_state.last_error = None
    features = {
        "service": sim["service"],
        "flag": sim["flag"],
        "src_bytes": sim["src_bytes"],
        "dst_bytes": sim["dst_bytes"],
        "count": sim["count"],
        "srv_count": sim["count"],
        "serror_rate": 1.0 if sim["flag"] == "S0" else 0.0,
        "same_srv_rate": 1.0,
    }

    try:
        with st.spinner(">> Extracting features & invoking ML classifier..."):
            pred = predict_traffic(features, src_ip=attacker_ip, dst_ip=target_ip)
    except APIClientError as exc:
        st.session_state.last_error = f"Inference pipeline failed: {exc}"
        return

    record = AnalysisRecord(
        src_ip=attacker_ip,
        dst_ip=target_ip,
        service=sim["service"],
        flag=sim["flag"],
        is_attack=bool(pred.get("is_attack", False)),
        status=pred.get("status", "Unknown"),
        xgb_prediction=int(pred.get("xgb_prediction", 0)),
        iso_prediction=int(pred.get("iso_prediction", 0)),
        top_features=pred.get("top_features", []),
        attack_cluster=pred.get("attack_cluster"),
        attack_family=pred.get("attack_family"),
    )

    if record.is_attack and sim.get("auto_block", False):
        success, _ = execute_firewall_rule(record.src_ip, action="block")
        if success:
            record.blocked = True
            st.session_state.blocked_ips.add(record.src_ip)

    if record.is_attack and sim.get("auto_llm", True):
        try:
            with st.spinner(">> Threat confirmed. Synthesizing AI incident report..."):
                rem = remediate_threat(
                    src_ip=attacker_ip,
                    dst_ip=target_ip,
                    service=sim["service"],
                    flag=sim["flag"],
                    detection_details={
                        "xgb": record.xgb_prediction,
                        "iso": record.iso_prediction,
                        "flag": sim["flag"],
                        "top_features": record.top_features,
                        "attack_family": record.attack_family,
                    },
                )
            record.remediation_report = rem.get("threat_analysis_report", "No report available.")
        except APIClientError as exc:
            record.remediation_report = f"⚠️ LLM Dispatch Error: {exc}"

    st.session_state.history.append(record)


def _render_result(record: AnalysisRecord) -> None:
    if record.is_attack:
        st.markdown(f'<div class="threat-box">🚨 <b>ANOMALY / ATTACK DETECTED:</b> {record.status}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="clean-box">🟢 <b>TRAFFIC CLEAN:</b> {record.status}</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("XGBoost Alert", "MALICIOUS" if record.xgb_prediction == 1 else "BENIGN")
    m2.metric("Isolation Forest", "ANOMALY" if record.iso_prediction == 1 else "NORMAL")
    m3.metric("Attack Family (DNA)", record.attack_family or "—")

    if record.is_attack and record.top_features:
        st.markdown("---")
        st.markdown("#### `[ SHAP_EXPLAINABILITY // WHY_FLAGGED ]`")
        df_shap = pd.DataFrame(record.top_features)
        df_shap["direction"] = df_shap["impact"].apply(lambda v: "Attack ▲" if v > 0 else "Normal ▼")
        fig_shap = px.bar(
            df_shap,
            x="impact",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={"Attack ▲": "#ff0055", "Normal ▼": "#00ff66"},
            title="Top Contributing Features (SHAP)",
            template="plotly_dark",
        )
        fig_shap.update_layout(paper_bgcolor="#060a0f", plot_bgcolor="#060a0f", height=280)
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption(
            "Positive impact pushes the model toward *attack*; negative impact pushes toward *normal*. "
            "This is the actual evidence behind the classification, not just the model's raw verdict."
        )

    if record.is_attack:
        st.markdown("---")
        st.markdown("#### `[ ACTIVE_INCIDENT_CONTAINMENT ]`")

        if record.blocked or record.src_ip in st.session_state.blocked_ips:
            st.success(f"🔒 Host Firewall: Inbound rule active for `{record.src_ip}` [DROP].")
        else:
            if st.button(f"🛡️ ISOLATE & DROP IP: {record.src_ip}", type="primary", use_container_width=True):
                with st.spinner("Executing OS kernel firewall rule..."):
                    success, output = execute_firewall_rule(record.src_ip, action="block")
                    if success:
                        record.blocked = True
                        st.session_state.blocked_ips.add(record.src_ip)
                        st.rerun()
                    else:
                        st.error(f"Execution failed: {output}")

    if record.is_attack and record.remediation_report:
        st.markdown("---")
        st.markdown("#### `[ AI_SOC_INCIDENT_REPORT ]`")
        st.code(record.remediation_report, language="markdown")


# --------------------------------------------------------------------------- #
# Tab 2: Batch Simulator
# --------------------------------------------------------------------------- #
def render_batch_tab(auto_block: bool = False) -> None:
    st.markdown("#### `[ TRAFFIC_BURST_SIMULATOR ]`")
    scenario = st.selectbox(
        "Attack Scenario Preset",
        [
            "SYN Flood Attack Burst (DoS)",
            "Aggressive Port Reconnaissance (Scan)",
            "Benign Web Browsing Stream",
        ],
    )
    batch_size = st.slider("Burst Packet Count", min_value=5, max_value=50, value=15)
    
    if st.button("🚀 INITIATE BURST TRANSMISSION", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(batch_size):
            if "SYN Flood" in scenario:
                ip_suffix = (i % 5) + 1
                features = {
                    "service": "private", "flag": "S0", "src_bytes": 0, "dst_bytes": 0,
                    "count": 150 + i * 10, "srv_count": 150 + i * 10, "serror_rate": 1.0, "same_srv_rate": 1.0
                }
                src_ip = f"198.51.100.{ip_suffix}"
            elif "Port Reconnaissance" in scenario:
                features = {
                    "service": "other", "flag": "REJ", "src_bytes": 0, "dst_bytes": 0,
                    "count": 80 + i, "srv_count": 1, "serror_rate": 0.0, "same_srv_rate": 0.1
                }
                src_ip = "198.51.100.99"
            else:
                features = {
                    "service": "http", "flag": "SF", "src_bytes": 450 + (i * 20), "dst_bytes": 3200,
                    "count": 5, "srv_count": 5, "serror_rate": 0.0, "same_srv_rate": 1.0
                }
                src_ip = f"10.0.0.{i+10}"

            try:
                pred = predict_traffic(features, src_ip=src_ip, dst_ip="192.168.1.100")
                is_atk = bool(pred.get("is_attack", False))
                is_blocked = False

                if is_atk and auto_block:
                    success, _ = execute_firewall_rule(src_ip, action="block")
                    if success:
                        is_blocked = True
                        st.session_state.blocked_ips.add(src_ip)

                rec = AnalysisRecord(
                    src_ip=src_ip,
                    dst_ip="192.168.1.100",
                    service=features["service"],
                    flag=features["flag"],
                    is_attack=is_atk,
                    status=pred.get("status", "Unknown"),
                    xgb_prediction=int(pred.get("xgb_prediction", 0)),
                    iso_prediction=int(pred.get("iso_prediction", 0)),
                    blocked=is_blocked,
                )
                st.session_state.history.append(rec)
            except Exception:
                pass

            progress_bar.progress((i + 1) / batch_size)
            status_text.text(f">> Packet {i+1}/{batch_size} processed.")
            time.sleep(0.03)

        st.success(f"✅ Ingestion complete: {batch_size} simulated packets analyzed.")


# --------------------------------------------------------------------------- #
# Tab 3: SOC Analytics & Live Sniffer Synchronization
# --------------------------------------------------------------------------- #
def render_visualizer_tab(auto_block: bool = False) -> None:
    st.markdown("#### `[ SOC_METRICS_&_TELEMETRY ]`")

    # 1. Fetch live logs from FastAPI backend
    backend_logs = fetch_backend_logs()
    if backend_logs:
        synced_records = []
        for log in backend_logs:
            is_atk = bool(log.get("is_attack", False))
            s_ip = log.get("src_ip", "Unknown")
            is_blocked = s_ip in st.session_state.blocked_ips

            if is_atk and auto_block and not is_blocked:
                success, _ = execute_firewall_rule(s_ip, action="block")
                if success:
                    is_blocked = True
                    st.session_state.blocked_ips.add(s_ip)

            synced_records.append(
                AnalysisRecord(
                    src_ip=s_ip,
                    dst_ip=log.get("dst_ip", "TargetHost"),
                    service=log.get("service", "other"),
                    flag=log.get("flag", "SF"),
                    is_attack=is_atk,
                    status=log.get("status", "Unknown"),
                    xgb_prediction=int(log.get("xgb_prediction", 0)),
                    iso_prediction=int(log.get("iso_prediction", 0)),
                    blocked=is_blocked,
                    timestamp=log.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    top_features=log.get("top_features", []),
                    attack_cluster=log.get("attack_cluster"),
                    attack_family=log.get("attack_family"),
                )
            )
        st.session_state.history = synced_records

    history: List[AnalysisRecord] = st.session_state.history
    
    total = len(history)
    attacks = sum(1 for r in history if r.is_attack)
    clean = total - attacks
    blocked = sum(1 for r in history if r.blocked or r.src_ip in st.session_state.blocked_ips)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TOTAL SAMPLES", total)
    c2.metric("ATTACKS DETECTED", attacks)
    c3.metric("CLEAN SAMPLES", clean)
    c4.metric("FIREWALL BLOCKS", blocked)

    if not history:
        st.info("No live telemetry recorded yet. Live sniffer packets will stream here automatically.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        data_pie = pd.DataFrame({
            "Classification": ["Clean", "Attack"],
            "Count": [clean, attacks]
        })
        fig_pie = px.pie(
            data_pie,
            values="Count",
            names="Classification",
            title="Attack vs Normal Traffic Ratio",
            color="Classification",
            color_discrete_map={"Clean": "#00ff66", "Attack": "#ff0055"},
            template="plotly_dark",
        )
        fig_pie.update_layout(paper_bgcolor="#060a0f", plot_bgcolor="#060a0f")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        df_hist = pd.DataFrame([{"Service": r.service, "Attack": 1 if r.is_attack else 0} for r in history])
        service_counts = df_hist.groupby("Service").sum().reset_index()
        fig_bar = px.bar(
            service_counts,
            x="Service",
            y="Attack",
            title="Attacks Identified per Service Protocol",
            color_discrete_sequence=["#00e5ff"],
            template="plotly_dark",
        )
        fig_bar.update_layout(paper_bgcolor="#060a0f", plot_bgcolor="#060a0f")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("#### `[ AUDIT_TRAIL_LOGS ]`")

    df_export = pd.DataFrame(
        [
            {
                "Timestamp": r.timestamp,
                "Source IP": r.src_ip,
                "Dest IP": r.dst_ip,
                "Service": r.service,
                "Flag": r.flag,
                "Result": r.status,
                "XGB Alert": "YES" if r.xgb_prediction == 1 else "NO",
                "ISO Anomaly": "YES" if r.iso_prediction == 1 else "NO",
                "Attack Family": r.attack_family or "—",
                "Blocked": "ACTIVE" if (r.blocked or r.src_ip in st.session_state.blocked_ips) else "NO",
            }
            for r in reversed(history)
        ]
    )
    st.dataframe(df_export, use_container_width=True)

    c_d1, c_d2 = st.columns(2)
    with c_d1:
        st.download_button(
            label="💾 EXPORT LOGS (CSV)",
            data=df_export.to_csv(index=False).encode("utf-8"),
            file_name=f"soc_nids_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c_d2:
        if st.button("🗑️ CLEAR LOG BUFFER", use_container_width=True):
            clear_backend_logs()
            st.session_state.history = []
            st.rerun()


# --------------------------------------------------------------------------- #
# Tab 4: Firewall Management Console
# --------------------------------------------------------------------------- #
def render_firewall_tab() -> None:
    st.markdown("#### `[ HOST_FIREWALL_RULESET_MANAGER ]`")
    blocked_ips = list(st.session_state.blocked_ips)

    if not blocked_ips:
        st.info("No IP addresses are currently blocked by this session.")
        return

    st.write(f"Active Containment Rules: `{len(blocked_ips)}` Targets Isolated")

    for ip in blocked_ips:
        col_ip, col_btn = st.columns([3, 1])
        col_ip.markdown(f"🚫 `DROP ALL` from Source: **{ip}**")
        if col_btn.button(f"🔓 UNBLOCK {ip}", key=f"unblock_{ip}"):
            with st.spinner(f"Removing firewall rule for {ip}..."):
                success, out = execute_firewall_rule(ip, action="unblock")
                if success:
                    st.session_state.blocked_ips.remove(ip)
                    for r in st.session_state.history:
                        if r.src_ip == ip:
                            r.blocked = False
                    st.success(f"Rule removed for {ip}")
                    st.rerun()
                else:
                    st.error(f"Failed to unblock: {out}")


# --------------------------------------------------------------------------- #
# Tab 5: Incident Narrative Timeline
# --------------------------------------------------------------------------- #
def render_timeline_tab() -> None:
    st.markdown("#### `[ INCIDENT_NARRATIVE_TIMELINE ]`")
    st.caption(
        "Groups raw telemetry per source IP into a story: first contact → recon → "
        "escalation → confirmed attack, instead of a flat, undifferentiated log."
    )

    narratives = fetch_timelines()
    if not narratives:
        st.info("No incident narratives yet. Run some traffic through the Analysis or Burst Simulator tabs.")
        return

    risk_colors = {"critical": "🔴", "high": "🟠", "low": "🟡", "none": "🟢"}

    for n in narratives:
        icon = risk_colors.get(n["risk_level"], "⚪")
        with st.expander(
            f"{icon} `{n['src_ip']}` — {n['attack_count']}/{n['event_count']} events flagged "
            f"— risk: {n['risk_level'].upper()} — {n['first_seen']} → {n['last_seen']}",
            expanded=(n["risk_level"] in ("critical", "high")),
        ):
            for stage in n["stages"]:
                family_tag = f" `[{stage['attack_family']}]`" if stage.get("attack_family") else ""
                st.markdown(
                    f"**{stage['timestamp']}** — {stage['label']} → `{stage['service']}` "
                    f"(flag `{stage['flag']}`) targeting `{stage['dst_ip']}`{family_tag}"
                )


# --------------------------------------------------------------------------- #
# Tab 6: Attack DNA (Behavioral Clustering)
# --------------------------------------------------------------------------- #
def render_attack_dna_tab() -> None:
    st.markdown("#### `[ ATTACK_DNA // BEHAVIORAL_FINGERPRINTING ]`")
    st.caption(
        "Confirmed attacks are clustered online (MiniBatchKMeans) into recurring behavioral "
        "families, so a repeat campaign is recognized even from a new source IP."
    )

    families = fetch_attack_families()
    if not families:
        st.info("No attack families discovered yet — confirmed attacks build this profile over time.")
        return

    df_fam = pd.DataFrame(families)
    col1, col2 = st.columns([1, 1])

    with col1:
        fig = px.bar(
            df_fam,
            x="family_name",
            y="observed_count",
            color="dominant_flag",
            title="Attack Family Population",
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="#060a0f", plot_bgcolor="#060a0f")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(
            df_fam.rename(
                columns={
                    "family_name": "Family",
                    "observed_count": "Observed Count",
                    "dominant_service": "Dominant Service",
                    "dominant_flag": "Dominant Flag",
                }
            )[["Family", "Observed Count", "Dominant Service", "Dominant Flag"]],
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------------------------------- #
# Tab 7: Model Health & Drift
# --------------------------------------------------------------------------- #
def render_model_health_tab() -> None:
    st.markdown("#### `[ MODEL_HEALTH // DRIFT_MONITORING ]`")
    st.caption(
        "Live feature distributions vs. the training-time baseline — catches data/model drift "
        "before it silently degrades detection quality."
    )

    health = fetch_model_health()
    if not health:
        st.warning("Could not reach the model-health endpoint. Is the backend online?")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples Observed", health.get("samples_observed", 0))
    c2.metric("Live Attack Rate", f"{health.get('attack_rate', 0) * 100:.1f}%")
    c3.metric("Explainability", "ON" if health.get("explainability_available") else "OFF")
    c4.metric("Attack Families", health.get("attack_families_discovered", 0))

    if not health.get("baseline_available"):
        st.warning(
            "No training-time baseline found in feature_schema.json. Retrain with the updated "
            "ml/train.py to enable drift scoring."
        )
        return

    drift_rows = health.get("drift", [])
    if not drift_rows:
        st.info("Not enough live traffic yet to compute drift.")
        return

    df_drift = pd.DataFrame(drift_rows)
    fig = px.bar(
        df_drift,
        x="feature",
        y="drift_z_score",
        color="flag",
        color_discrete_map={"drift": "#ff0055", "ok": "#00e5ff"},
        title="Feature Drift (Live Mean vs. Training Baseline, in std-deviations)",
        template="plotly_dark",
    )
    fig.update_layout(paper_bgcolor="#060a0f", plot_bgcolor="#060a0f")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("|z| > 3 is flagged as drift — the live feature mean has moved 3+ std-deviations from training.")

    st.dataframe(
        df_drift.rename(
            columns={
                "feature": "Feature",
                "baseline_mean": "Baseline Mean",
                "live_mean": "Live Mean",
                "drift_z_score": "Drift (z-score)",
                "flag": "Status",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------------------------------- #
# Main Entry Point
# --------------------------------------------------------------------------- #
st.title("⚡ AI-NIDS // SOC COMMAND TERMINAL")
st.caption("Autonomous Threat Detection • Isolation Forest Anomaly Engine • LLM Incident Orchestration")

sim_inputs = render_sidebar()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 LIVE ANALYSIS & MITIGATION",
    "⚡ BURST SIMULATOR",
    "📊 TELEMETRY & AUDIT",
    "🛡️ FIREWALL RULES",
    "🕒 INCIDENT TIMELINE",
    "🧬 ATTACK DNA",
    "🩺 MODEL HEALTH",
])

with tab1:
    render_analysis_tab(sim_inputs)
with tab2:
    render_batch_tab(auto_block=sim_inputs.get("auto_block", False))
with tab3:
    render_visualizer_tab(auto_block=sim_inputs.get("auto_block", False))
with tab4:
    render_firewall_tab()
with tab5:
    render_timeline_tab()
with tab6:
    render_attack_dna_tab()
with tab7:
    render_model_health_tab()

# Periodic Auto-Polling for Streamlit
if sim_inputs.get("live_polling", False):
    time.sleep(5)
    st.rerun()