"""Streamlit frontend for the classifier API."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(ttl=30)
def fetch_classes() -> list[dict]:
    """Cache /classes response."""
    try:
        r = requests.get(f"{API_BASE}/classes", timeout=5)
        r.raise_for_status()
        return r.json().get("classes", [])
    except Exception:
        return []


@st.cache_data(ttl=30)
def fetch_health() -> dict:
    """Cache /health response."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"status": "unreachable", "models_loaded": [], "device": "?"}


def predict_single(uploaded_file, model_name: str) -> dict | None:
    """Hit POST /predict."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(),
                          uploaded_file.type or "image/jpeg")}
        params = {"model": model_name} if model_name else {}
        r = requests.post(f"{API_BASE}/predict", files=files,
                          params=params, timeout=30)
        if r.status_code >= 400:
            st.error(f"API error {r.status_code}: {r.text}")
            return None
        return r.json()
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def predict_batch(files, model_name: str) -> dict | None:
    """Hit POST /predict-batch."""
    try:
        multi = [("files", (f.name, f.getvalue(), f.type or "image/jpeg"))
                 for f in files]
        params = {"model": model_name} if model_name else {}
        r = requests.post(f"{API_BASE}/predict-batch", files=multi,
                          params=params, timeout=120)
        if r.status_code >= 400:
            st.error(f"API error {r.status_code}: {r.text}")
            return None
        return r.json()
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def render_predictions(preds: list[dict]) -> None:
    """Bar chart for top-K."""
    df = pd.DataFrame(preds)
    if df.empty:
        return
    df = df.rename(columns={"display_name": "Person",
                              "confidence": "Confidence"})
    df = df[["Person", "Confidence"]].set_index("Person")
    st.bar_chart(df, horizontal=True, height=240)


def sidebar() -> str:
    """Render sidebar; return chosen model name."""
    st.sidebar.title("Settings")
    health = fetch_health()
    st.sidebar.markdown(f"**API:** `{API_BASE}`")
    st.sidebar.markdown(f"**Status:** {health.get('status', '?')}")
    st.sidebar.markdown(f"**Device:** {health.get('device', '?')}")
    models = health.get("models_loaded", [])
    chosen = st.sidebar.selectbox("Model", options=models or [""])

    with st.sidebar.expander("About this project"):
        st.markdown(
            "Production-grade classifier for **30 Pakistani public figures** "
            "(politicians + military). End-to-end MLOps with DVC, MLflow, "
            "Docker, GitHub Actions, and Airflow."
        )
    with st.sidebar.expander("View all 30 classes"):
        classes = fetch_classes()
        if classes:
            st.dataframe(pd.DataFrame(classes), hide_index=True,
                         use_container_width=True)
    return chosen


def tab_single(model_name: str) -> None:
    """Single-image tab."""
    uploaded = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=False,
    )
    if uploaded is None:
        st.info("Drag a face image into the uploader above.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption=uploaded.name, use_column_width=True)
    with col_b:
        with st.spinner("Predicting..."):
            result = predict_single(uploaded, model_name)
        if not result:
            return
        st.markdown(f"**Model:** `{result['model_used']}`  "
                    f"**Inference:** {result['inference_time_ms']:.1f} ms")
        st.markdown("**Top predictions:**")
        render_predictions(result["predictions"])


def tab_batch(model_name: str) -> None:
    """Batch tab."""
    files = st.file_uploader("Upload multiple images",
                                type=["jpg", "jpeg", "png", "webp", "bmp"],
                                accept_multiple_files=True)
    if not files:
        return
    with st.spinner(f"Predicting {len(files)} images..."):
        result = predict_batch(files, model_name)
    if not result:
        return
    cols = st.columns(3)
    for i, item in enumerate(result["results"]):
        with cols[i % 3]:
            matching = next((f for f in files if f.name == item["filename"]),
                            None)
            if matching:
                st.image(Image.open(matching), caption=item["filename"],
                         use_column_width=True)
            if item.get("error"):
                st.error(item["error"])
            elif item.get("predictions"):
                top = item["predictions"][0]
                st.success(f"{top['display_name']} ({top['confidence']:.2%})")


def tab_performance() -> None:
    """Model performance tab."""
    results_dir = PROJECT_ROOT / "results"
    csv = results_dir / "model_comparison.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No model comparison data yet — train models first.")
    for png in sorted(results_dir.glob("*.png")):
        st.image(str(png), caption=png.name, use_column_width=True)


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="Pak Public Figures Recognition",
                       layout="wide", page_icon="PK")
    st.title("Pakistani Public Figures Recognition")
    model_name = sidebar()
    tab1, tab2, tab3 = st.tabs(["Single Image", "Batch Upload",
                                  "Model Performance"])
    with tab1:
        tab_single(model_name)
    with tab2:
        tab_batch(model_name)
    with tab3:
        tab_performance()


if __name__ == "__main__":
    main()
