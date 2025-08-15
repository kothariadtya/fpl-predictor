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
# Data loading from FPL
# --------------------------------------------------------------------------------------
@st.cache_data
def load_fpl_raw() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = "https://fantasy.premierleague.com/api"
    r_boot = safe_get(f"{base}/bootstrap-static/")
    r_fix = safe_get(f"{base}/fixtures/")
    if r_boot is None or r_fix is None:
        raise RuntimeError("Failed to fetch FPL API. Try again in a minute.")
    boot = r_boot.json()
    fixtures = r_fix.json()
    elements = pd.DataFrame(boot["elements"])
    teams = pd.DataFrame(boot["teams"])
    fixtures_df = pd.DataFrame(fixtures)
    return elements, teams, fixtures_df

def current_gameweek(fixtures_df: pd.DataFrame) -> int | None:
    if "event" not in fixtures_df.columns:
        return None
    ev = fixtures_df["event"].dropna()
    if ev.empty:
        return None
    return int(ev.min())

def team_fixture_difficulty_and_dgw(elements: pd.DataFrame,
                                    fixtures_df: pd.DataFrame) -> Tuple[Dict[int,int], Set[int]]:
    """
    Returns:
      - next_fixture_difficulty per element id (1..5)
      - set of team_ids that have a double GW (any event with >1 match)
    """
    next_diff: Dict[int,int] = {}
    dgw_teams: Set[int] = set()

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

    for _, row in elements.iterrows():
        team_id = int(row["team"])
        pid = int(row["id"])
        team_fixt = fixtures_df[
            ((fixtures_df.get("team_h") == team_id) | (fixtures_df.get("team_a") == team_id))
            & fixtures_df.get("event").notna()
        ]
        if team_fixt.empty:
            next_diff[pid] = 3
            continue
        ngw = int(team_fixt["event"].min())
        gw_rows = team_fixt[team_fixt["event"] == ngw]
        fi = gw_rows.iloc[0]
        if int(fi["team_h"]) == team_id:
            difficulty = int(fi["team_h_difficulty"])
        else:
            difficulty = int(fi["team_a_difficulty"])
        next_diff[pid] = difficulty

    return next_diff, dgw_teams

# --------------------------------------------------------------------------------------
# Injuries (Premier Injuries table)
# --------------------------------------------------------------------------------------
@st.cache_data
def fetch_injured_players() -> Set[str]:
    url = "https://www.premierinjuries.com/injury-table.php"
    r = safe_get(url, timeout=10, retries=3, sleep_base=1.2)
    injured: Set[str] = set()
    if r is None:
        return injured
    soup = BeautifulSoup(r.text, "html.parser")
    for row in soup.select("table tr"):
        cols = row.find_all("td")
        if len(cols) >= 1:
            nm = cols[0].get_text(" ", strip=True)
            nm = nm.split("\n")[0].strip()
            if nm:
                injured.add(nm.lower())
    return injured

# --------------------------------------------------------------------------------------
# Mentality via news snippets (headline/description/first paras)
# --------------------------------------------------------------------------------------
NEWS_SITES = [
    "https://www.bbc.com/sport/football",
    "https://www.theguardian.com/football",
    "https://www.skysports.com/football/news",
]

@st.cache_resource
def get_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

def player_mentality_from_news(names: List[str], max_links_per_site: int = 10) -> Dict[str,str]:
    analyzer = get_sentiment_analyzer()
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
            if score < -0.3:
                tlow = text.lower()
                for n in names:
                    if n.lower() in tlow:
                        mentality[n] = "Volatile"
            time.sleep(1.1 + random.random() * 0.7)
    return mentality

# --------------------------------------------------------------------------------------
# Pre-season boost (GW ≤ 5) using article mentions ("pre‑season"/"friendly")
# --------------------------------------------------------------------------------------
def preseason_boost_from_news(names: List[str]) -> Dict[str, float]:
    analyzer = get_sentiment_analyzer()
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
            time.sleep(1.0 + random.random() * 0.6)
    return boost

# --------------------------------------------------------------------------------------
# Master loader -> returns prepared DataFrame for app + picker
# --------------------------------------------------------------------------------------
@st.cache_data
def load_players_df() -> pd.DataFrame:
    elements, teams, fixtures_df = load_fpl_raw()

    # fixture difficulty & DGW
    next_diff_map, dgw_teams = team_fixture_difficulty_and_dgw(elements, fixtures_df)

    # merge team info
    team_min = teams[["id", "name", "short_name", "strength"]]
    df = elements.merge(team_min, left_on="team", right_on="id", suffixes=("", "_team"))

    # basics
    df["Price"] = df["now_cost"] / 10.0
    df["Ownership"] = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0.0)
    df["Differential"] = df["Ownership"] < 10.0
    df["FixtureDifficulty"] = df["id"].map(next_diff_map).fillna(3).astype(int)
    df["DoubleGW"] = df["team"].apply(lambda t: bool(t in dgw_teams))

    # injuries
    injured = fetch_injured_players()
    df["InjuryRisk"] = df["web_name"].str.lower().apply(lambda n: "Yes" if n in injured else "No")

    # mentality
    names_list = df["web_name"].tolist()
    mentality_map = player_mentality_from_news(names_list)
    df["Mentality"] = df["web_name"].apply(lambda n: mentality_map.get(n, "Stable"))

    # rotation risk
    df["chance_of_playing_next_round"] = pd.to_numeric(df["chance_of_playing_next_round"], errors="coerce").fillna(100)
    df["RotationRisk"] = ((df["minutes"] < 90) | (df["chance_of_playing_next_round"] < 75)).map({True: "Yes", False: "No"})

    # factors
    df["FatigueFactor"]   = df["minutes"].apply(lambda m: 0.95 if m > 270 else 1.0)
    df["ChemistryFactor"] = pd.to_numeric(df["influence"], errors="coerce").fillna(0.0).apply(lambda x: 1.0 if x > 300 else 0.95)
    df["InjuryFactor"]    = df["InjuryRisk"].map({"Yes": 0.85, "No": 1.0})
    df["MentalityFactor"] = df["Mentality"].map({"Volatile": 0.95, "Stable": 1.0})
    df["RotationFactor"]  = df["RotationRisk"].map({"Yes": 0.92, "No": 1.0})
    df["FixtureFactor"]   = df["FixtureDifficulty"].apply(lambda x: 1.05 if x <= 2 else (0.95 if x >= 4 else 1.0))
    df["DoubleGWFactor"]  = df["DoubleGW"].map({True: 1.15, False: 1.0})

    # pre-season factor if early season
    gw = current_gameweek(fixtures_df)
    df["PreSeasonFactor"] = 1.0
    if gw is not None and gw <= 5:
        preseason_map = preseason_boost_from_news(names_list)
        df["PreSeasonFactor"] = df["web_name"].apply(lambda n: preseason_map.get(n, 1.0))
# OPTIONAL: blend a learned model if available
try:
    import joblib
    model = joblib.load("fpl_model.pkl")
    feats = pd.DataFrame({
        "form": pd.to_numeric(players["form"], errors="coerce").fillna(0),
        "influence": pd.to_numeric(players["influence"], errors="coerce").fillna(0),
        "ict_index": pd.to_numeric(players["ict_index"], errors="coerce").fillna(0),
        "threat": pd.to_numeric(players["threat"], errors="coerce").fillna(0),
        "creativity": pd.to_numeric(players["creativity"], errors="coerce").fillna(0),
        "minutes": players["minutes"].fillna(0),
        "now_cost": players["now_cost"].fillna(0),
    })
    ml_points = model.predict(feats)
    # blend: 60% rule-based, 40% model (tune as you like)
    players["PredictedPoints"] = 0.6 * players["PredictedPoints"] + 0.4 * ml_points
except Exception as e:
    # model not present or failed → ignore and keep rule-based score
    pass

    # predicted points
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

    # positions map
    df["Position"] = df["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
    return df

# --------------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------------
players = load_players_df()
players_full = players.copy()   # keep unfiltered for optimizer

# --------------------------------------------------------------------------------------
# Sidebar + table (filters affect only table, NOT optimizer)
# --------------------------------------------------------------------------------------
selected_position = st.sidebar.selectbox("Position", ["ALL", "GK", "DEF", "MID", "FWD"])
table_df = players.copy()
if selected_position != "ALL":
    table_df = table_df[table_df["Position"] == selected_position]
if st.sidebar.checkbox("Show only Differentials (<10%)"):
    table_df = table_df[table_df["Differential"] == True]

top_n = st.sidebar.slider("Show top N players", 5, 60, 15)
st.subheader(f"Top {top_n} FPL Player Picks")
cols_to_show = [
    "web_name","team","Position","Price","form","Ownership","Differential",
    "PredictedPoints","InjuryRisk","Mentality","FixtureDifficulty","DoubleGW","PPM"
]
st.dataframe(table_df.sort_values("PredictedPoints", ascending=False)[cols_to_show].head(top_n), use_container_width=True)

# --------------------------------------------------------------------------------------
# Best 15-Man FPL Team Selector (robust, DataFrame-only, respects all rules)
# --------------------------------------------------------------------------------------
def build_best_team(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Greedy FPL-compliant squad builder:
      - 15 players: 2 GK, 5 DEF, 5 MID, 3 FWD
      - Budget 100.0
      - Max 3 per real club team
      - Ranks by PPM, then PredictedPoints
    """
    budget = 100.0
    need = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    team_limit = 3

    # work on a clean DF with only required cols
    cols = ["web_name","team","Position","Price","PredictedPoints","PPM"]
    df = df_full[cols].dropna(subset=["Price","PredictedPoints","PPM"]).copy()

    # sort by efficiency then raw points
    pos_sorted = {
        p: df[df["Position"] == p].sort_values(["PPM","PredictedPoints"], ascending=False).reset_index(drop=True)
        for p in need.keys()
    }

    squad_rows: List[Dict] = []
    team_counts: Dict = {}

    def total_cost(rows): return sum(r["Price"] for r in rows)
    def can_add(rowd: Dict):
        if team_counts.get(rowd["team"], 0) >= team_limit:
            return False
        if total_cost(squad_rows) + rowd["Price"] > budget:
            return False
        return True

    # primary pick by position
    for p in ["GK","DEF","MID","FWD"]:
        picked = 0
        for _, r in pos_sorted[p].iterrows():
            rowd = r.to_dict()
            if can_add(rowd):
                squad_rows.append(rowd)
                team_counts[rowd["team"]] = team_counts.get(rowd["team"], 0) + 1
                picked += 1
                if picked == need[p]:
                    break

    # fill any gaps (cheapest feasible options)
    for p in need:
        while sum(1 for r in squad_rows if r["Position"] == p) < need[p]:
            pool = pos_sorted[p].sort_values("Price", ascending=True)
            placed = False
            for _, r in pool.iterrows():
                rowd = r.to_dict()
                if any((x["web_name"] == rowd["web_name"]) for x in squad_rows):
                    continue
                if can_add(rowd):
                    squad_rows.append(rowd)
                    team_counts[rowd["team"]] = team_counts.get(rowd["team"], 0) + 1
                    placed = True
                    break
            if not placed:
                # downgrade worst in this position to free money
                same_pos = [x for x in squad_rows if x["Position"] == p]
                if not same_pos:
                    break
                worst = min(same_pos, key=lambda x: (x["PPM"], x["PredictedPoints"]))
                squad_rows.remove(worst)
                team_counts[worst["team"]] -= 1

    # if still over budget, iteratively downgrade worst value picks
    iters = 0
    while total_cost(squad_rows) > budget and iters < 80:
        iters += 1
        cand = max(squad_rows, key=lambda x: (x["Price"] / max(x["PredictedPoints"], 0.01)))
        pos = cand["Position"]
        # try cheaper replacement in same position
        team_counts[cand["team"]] -= 1
        squad_rows.remove(cand)
        pool = pos_sorted[pos].sort_values("Price", ascending=True)
        placed = False
        for _, r in pool.iterrows():
            rowd = r.to_dict()
            if any((x["web_name"] == rowd["web_name"]) for x in squad_rows):
                continue
            if team_counts.get(rowd["team"], 0) >= team_limit:
                continue
            if total_cost(squad_rows) + rowd["Price"] <= budget:
                squad_rows.append(rowd)
                team_counts[rowd["team"]] = team_counts.get(rowd["team"], 0) + 1
                placed = True
                break
        if not placed:
            # put back to avoid losing a slot
            squad_rows.append(cand)
            team_counts[cand["team"]] = team_counts.get(cand["team"], 0) + 1
            break

    return pd.DataFrame(squad_rows)

if st.button("📋 Build My FPL Team"):
    team_df = build_best_team(players_full)
    team_df = team_df.sort_values("PredictedPoints", ascending=False).reset_index(drop=True)
    # Simple armband: top = C, second = VC
    roles = ["Captain","Vice-Captain"] + ["First Team"] * 9 + ["Sub"] * 3
    if len(team_df) < len(roles):
        roles = roles[:len(team_df)]
    team_df["Role"] = roles

    st.subheader("🔝 Your Best 15‑Man Team (Auto‑Picked)")
    st.dataframe(team_df[["web_name","Position","team","Price","PredictedPoints","Role"]], use_container_width=True)
