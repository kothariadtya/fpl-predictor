# data_loader.py
# Build a rich player dataframe with FPL + fixtures + injuries + mentality + preseason boost
# Safe for Streamlit Cloud (no newspaper3k, polite scraping)

from __future__ import annotations
import random
import time
from typing import Dict, Set, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------
# Networking helpers (polite)
# ----------------------------
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

# ----------------------------
# FPL data
# ----------------------------
def load_fpl_raw() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = "https://fantasy.premierleague.com/api"
    r_boot = safe_get(f"{base}/bootstrap-static/")
    r_fix = safe_get(f"{base}/fixtures/")
    if r_boot is None or r_fix is None:
        raise RuntimeError("Failed to fetch FPL API.")

    boot = r_boot.json()
    fixtures = r_fix.json()
    elements = pd.DataFrame(boot["elements"])
    teams = pd.DataFrame(boot["teams"])
    fixtures_df = pd.DataFrame(fixtures)
    return elements, teams, fixtures_df

def current_gameweek(fixtures_df: pd.DataFrame) -> int | None:
    ev = fixtures_df["event"].dropna().astype(int)
    return int(ev.min()) if not ev.empty else None

def team_fixture_difficulty_and_dgw(elements: pd.DataFrame,
                                    fixtures_df: pd.DataFrame) -> Tuple[Dict[int,int], Set[int]]:
    """
    Returns:
      - next_fixture_difficulty per element id (1..5)
      - set of team_ids that have a double GW (any event with >1 match)
    """
    next_diff: Dict[int,int] = {}
    dgw_teams: Set[int] = set()

    # precompute counts per team/event
    if "event" in fixtures_df.columns:
        tmp = fixtures_df.dropna(subset=["event"]).copy()
        ev_counts = (
            pd.concat([
                tmp[["event", "team_h"]].rename(columns={"team_h":"team"}),
                tmp[["event", "team_a"]].rename(columns={"team_a":"team"})
            ])
            .groupby(["event", "team"])
            .size()
            .reset_index(name="n")
        )
        dgw_teams = set(ev_counts.loc[ev_counts["n"] > 1, "team"].unique())

    # map next GW difficulty per player
    for _, row in elements.iterrows():
        team_id = int(row["team"])
        pid = int(row["id"])
        team_fixt = fixtures_df[
            ((fixtures_df["team_h"] == team_id) | (fixtures_df["team_a"] == team_id))
            & fixtures_df["event"].notna()
        ]
        if team_fixt.empty:
            next_diff[pid] = 3
            continue
        ngw = int(team_fixt["event"].min())
        gw_rows = team_fixt[team_fixt["event"] == ngw]
        # take first match difficulty (if DGW, we'll add a separate factor below)
        fi = gw_rows.iloc[0]
        if int(fi["team_h"]) == team_id:
            difficulty = int(fi["team_h_difficulty"])
        else:
            difficulty = int(fi["team_a_difficulty"])
        next_diff[pid] = difficulty

    return next_diff, dgw_teams

# ----------------------------
# Injuries (Premier Injuries)
# ----------------------------
def fetch_injured_players() -> Set[str]:
    url = "https://www.premierinjuries.com/injury-table.php"
    r = safe_get(url, timeout=10, retries=3, sleep_base=1.2)
    injured: Set[str] = set()
    if r is None:
        return injured  # fallback: empty set

    soup = BeautifulSoup(r.text, "html.parser")
    for row in soup.select("table tr"):
        cols = row.find_all("td")
        if len(cols) >= 1:
            nm = cols[0].get_text(" ", strip=True)
            nm = nm.split("\n")[0].strip()
            if nm:
                injured.add(nm.lower())
    return injured

# ----------------------------
# Mentality via news snippets
# ----------------------------
NEWS_SITES = [
    "https://www.bbc.com/sport/football",
    "https://www.theguardian.com/football",
    "https://www.skysports.com/football/news",
]

def player_mentality_from_news(names: List[str], max_links_per_site: int = 10) -> Dict[str,str]:
    analyzer = SentimentIntensityAnalyzer()
    mentality = {n: "Stable" for n in names}

    for site in NEWS_SITES:
        rs = safe_get(site, timeout=10, retries=3, sleep_base=1.0)
        if rs is None:
            continue
        soup = BeautifulSoup(rs.text, "html.parser")
        links = soup.find_all("a", href=True)
        article_links = []
        for a in links:
            href = a["href"]
            if "/football" in href and len(href) > 20:
                if href.startswith("/"):
                    href = f"https://{site.split('/')[2]}{href}"
                if href not in article_links:
                    article_links.append(href)

        for link in article_links[:max_links_per_site]:
            ar = safe_get(link, timeout=10, retries=2, sleep_base=1.2)
            if ar is None:
                continue
            soup2 = BeautifulSoup(ar.text, "html.parser")
            desc = soup2.find("meta", attrs={"name": "description"})
            if desc and "content" in desc.attrs:
                text = desc["content"]
            else:
                ps = soup2.find_all("p")
                text = " ".join(p.get_text(" ", strip=True) for p in ps[:2])

            score = analyzer.polarity_scores(text).get("compound", 0.0)

            # Negative article mentioning a player → Volatile
            if score < -0.3:
                tlow = text.lower()
                for n in names:
                    if n.lower() in tlow:
                        mentality[n] = "Volatile"
            time.sleep(1.2 + random.random() * 0.8)  # polite pause

    return mentality

# ----------------------------
# Pre-season boost (GW ≤ 5)
# ----------------------------
def preseason_boost_from_news(names: List[str]) -> Dict[str, float]:
    """
    Lightweight approach: look for 'pre-season' / 'friendly' mentions in articles/snippets.
    Positive tone → slight boost; negative tone → slight penalty.
    """
    analyzer = SentimentIntensityAnalyzer()
    boost = {n: 1.0 for n in names}

    for site in NEWS_SITES:
        rs = safe_get(site, timeout=10, retries=3, sleep_base=1.0)
        if rs is None:
            continue
        soup = BeautifulSoup(rs.text, "html.parser")
        links = soup.find_all("a", href=True)
        article_links = []
        for a in links:
            href = a["href"]
            if "/football" in href and len(href) > 20:
                if href.startswith("/"):
                    href = f"https://{site.split('/')[2]}{href}"
                if href not in article_links:
                    article_links.append(href)

        for link in article_links[:10]:
            ar = safe_get(link, timeout=10, retries=2, sleep_base=1.2)
            if ar is None:
                continue
            soup2 = BeautifulSoup(ar.text, "html.parser")
            # Prefer meta
            text = ""
            desc = soup2.find("meta", attrs={"name": "description"})
            if desc and "content" in desc.attrs:
                text = desc["content"]
            else:
                ps = soup2.find_all("p")
                text = " ".join(p.get_text(" ", strip=True) for p in ps[:3])

            low = text.lower()
            if ("pre-season" in low) or ("preseason" in low) or ("friendly" in low):
                score = analyzer.polarity_scores(text).get("compound", 0.0)
                for n in names:
                    if n.lower() in low:
                        if score > 0.3:
                            boost[n] = max(boost[n], 1.10)  # +10%
                        elif score < -0.3:
                            boost[n] = min(boost[n], 0.90)  # -10%
            time.sleep(1.0 + random.random() * 0.7)

    return boost

# ----------------------------
# Master loader (returns ready DF)
# ----------------------------
def load_players_df(apply_preseason_if_gw_le5: bool = True) -> pd.DataFrame:
    elements, teams, fixtures_df = load_fpl_raw()

    # Map fixture difficulty & DGW
    next_diff_map, dgw_teams = team_fixture_difficulty_and_dgw(elements, fixtures_df)

    # Merge team names & basic fields
    team_min = teams[["id", "name", "short_name", "strength"]]
    df = elements.merge(team_min, left_on="team", right_on="id", suffixes=("", "_team"))

    # Basic transforms
    df["Price"] = df["now_cost"] / 10.0
    df["Ownership"] = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0.0)
    df["Differential"] = df["Ownership"] < 10.0
    df["FixtureDifficulty"] = df["id"].map(next_diff_map).fillna(3).astype(int)
    df["DoubleGW"] = df["team"].apply(lambda t: bool(t in dgw_teams))

    # Injuries
    injured = fetch_injured_players()
    df["InjuryRisk"] = df["web_name"].str.lower().apply(lambda n: "Yes" if n in injured else "No")

    # Mentality via news
    names_list = df["web_name"].tolist()
    mentality_map = player_mentality_from_news(names_list)
    df["Mentality"] = df["web_name"].apply(lambda n: mentality_map.get(n, "Stable"))

    # Rotation risk (simple): low minutes recently OR low chance of next round
    df["chance_of_playing_next_round"] = pd.to_numeric(df["chance_of_playing_next_round"], errors="coerce").fillna(100)
    df["RotationRisk"] = ((df["minutes"] < 90) | (df["chance_of_playing_next_round"] < 75)).map({True: "Yes", False: "No"})

    # Factors
    df["FatigueFactor"] = df["minutes"].apply(lambda m: 0.95 if m > 270 else 1.0)
    df["ChemistryFactor"] = pd.to_numeric(df["influence"], errors="coerce").fillna(0.0).apply(lambda x: 1.0 if x > 300 else 0.95)
    df["InjuryFactor"] = df["InjuryRisk"].map({"Yes": 0.85, "No": 1.0})
    df["MentalityFactor"] = df["Mentality"].map({"Volatile": 0.95, "Stable": 1.0})
    df["RotationFactor"] = df["RotationRisk"].map({"Yes": 0.92, "No": 1.0})
    df["FixtureFactor"] = df["FixtureDifficulty"].apply(lambda x: 1.05 if x <= 2 else (0.95 if x >= 4 else 1.0))
    df["DoubleGWFactor"] = df["DoubleGW"].map({True: 1.15, False: 1.0})

    # Pre-season factor (GW ≤ 5) using news-based proxy
    gw = current_gameweek(fixtures_df)
    df["PreSeasonFactor"] = 1.0
    if gw is not None and gw <= 5 and apply_preseason_if_gw_le5:
        preseason_map = preseason_boost_from_news(names_list)
        df["PreSeasonFactor"] = df["web_name"].apply(lambda n: preseason_map.get(n, 1.0))

    # Predicted points (composite)
    df["form"] = pd.to_numeric(df["form"], errors="coerce").fillna(0.0)
    df["PredictedPoints"] = (
        df["form"]
        * df["FatigueFactor"]
        * df["ChemistryFactor"]
        * df["InjuryFactor"]
        * df["MentalityFactor"]
        * df["RotationFactor"]
        * df["FixtureFactor"]
        * df["DoubleGWFactor"]
        * df["PreSeasonFactor"]
    )
    df["PPM"] = (df["PredictedPoints"] / df["Price"]).replace([pd.NA, pd.NaT, float("inf")], 0)

    # Keep columns the app expects (plus extras)
    keep_cols = [
        "id","web_name","team","element_type","now_cost","minutes","form","influence","selected_by_percent",
        "Price","Ownership","Differential",
        "FixtureDifficulty","DoubleGW",
        "InjuryRisk","Mentality","RotationRisk",
        "FatigueFactor","ChemistryFactor","InjuryFactor","MentalityFactor","RotationFactor",
        "FixtureFactor","DoubleGWFactor","PreSeasonFactor",
        "PredictedPoints","PPM","name","short_name"
    ]
    out = df[keep_cols].copy()
    return out

# Convenience entry point for Streamlit
def load_for_app() -> pd.DataFrame:
    """One-liner used by app.py"""
    return load_players_df(apply_preseason_if_gw_le5=True)
# Data loader placeholder

def load_data():
    pass
