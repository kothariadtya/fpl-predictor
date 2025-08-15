import random
import time
from typing import Dict, Set, List, Tuple

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from data_loader import load_for_app

players = load_for_app()
players_full = players.copy()

# --------------------------------------------------------------------------------------
# Streamlit setup
# --------------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="FPL Team Predictor")
st.title("⚽ Fantasy Premier League – AI Team Predictor (Personal)")

# --------------------------------------------------------------------------------------
# Polite networking helpers (avoid blocks)
# --------------------------------------------------------------------------------------
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
]

def safe_get(url: str, timeout: int = 10, retries: int = 3, sleep_base: float = 1.0) -> requests.Response | None:
    """GET with random UA, backoff, and graceful failure."""
    for i in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": random.choice(USER_AGENTS)}, timeout=timeout)
            if r.status_code == 200 and r.text:
                return r
        except Exception:
            pass
        time.sleep(sleep_base * (1 + 0.5 * random.random()) * (i + 1))
    return None

# --------------------------------------------------------------------------------------
