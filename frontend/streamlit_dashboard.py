"""Simple Streamlit dashboard for metrics and explanations."""

import os
import requests
import streamlit as st

API_URL = os.environ.get("METRICS_API_URL", "http://localhost:8000")

st.title("Monitoring Dashboard")

# metrics
metrics = requests.get(f"{API_URL}/metrics/evaluation").json()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precision", f"{metrics.get('precision', 0):.2f}")
col2.metric("Recall", f"{metrics.get('recall', 0):.2f}")
col3.metric("Latency", f"{metrics.get('latency', 0):.2f}")
col4.metric("Fairness", f"{metrics.get('fairness', 0):.2f}")

# explanations
st.header("Explanations")
expl = requests.get(f"{API_URL}/metrics/explanations").json()
for e in expl:
    st.write(e.get("text"))
