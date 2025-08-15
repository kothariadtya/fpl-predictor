# data_loader.py
# Loads & builds a rich player dataframe for the app:
# - FPL API (players, teams, fixtures)
# - Fixture Difficulty + Double GW
# - Injuries (Premier Injuries scrape)
# - Mentality from article snippets (BBC/Guardian/Sky) via VADER
# - **Pre-season boost for GW <= 5** (friendlies/news)
# - Fatigue, chemistry, rotation
# - Robust PredictedPoints (handles GW1 zeros) + PPM

from __future__ import annotations
import random
import time
from typing import Dict, Set, List, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def build_next_opponent_maps(fixtures_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, str], Dict[int, int]]:
    """
    For each team_id, find the next gameweek (min event) and the opponent team_id.
    Returns:
      team_next_opp: {team_id -> opponent_team_id}
      team_home_away: {team_id -> 'H' or 'A'}
      team_next_gw: {team_id -> event_number}
    """
    team_next_opp: Dict[int, int] = {}
    team_home_away: Dict[int, str] = {}
    team_next_gw: Dict[int, int] = {}

    fx = fixtures_df.dropna(subset=["event"]).copy()
    if fx.empty:
        return team_next_opp, team_home_away, team_next_gw

    # For each team, choose the earliest upcoming event (next GW)
    teams_in_fixtures = pd.unique(pd.concat([fx["team_h"], fx["team_a"]], ignore_index=True))
    for t in teams_in_fixtures:
        t_fixt = fx[(fx["team_h"] == t) | (fx["team_a"] == t)]
        if t_fixt.empty:
            continue
        next_gw = int(t_fixt["event"].min())
        gw_rows = t_fixt[t_fixt["event"] == next_gw].sort_index()
        # If DGW, just take the first listing; your DoubleGWFactor still captures the bonus
        row = gw_rows.iloc[0]
        if int(row["team_h"]) == int(t):
            opp = int(row["team_a"]); ha = "H"
        else:
            opp = int(row["team_h"]); ha = "A"
        team_next_opp[int(t)] = opp
        team_home_away[int(t)] = ha
        team_next_gw[int(t)] = next_gw

    return team_next_opp, team_home_away, team_next_gw



# -------- polite networking --------
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
]

def safe_get(url: str, timeout: int = 10, retries: int = 3, sleep_base: float = 1.0):
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

# -------- FPL API --------
def load_fpl_raw() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = "https://fantasy.premierleague.com/api"
    r_boot = safe_get(f"{base}/bootstrap-static/")
    r_fix  = safe_get(f"{base}/fixtures/")
    if r_boot is None or r_fix is None:
        raise RuntimeError("Failed to fetch FPL API. Try again in a minute.")
    boot = r_boot.json()
    elements = pd.DataFrame(boot["elements"])
    teams    = pd.DataFrame(boot["teams"])
    fixtures = pd.DataFrame(r_fix.json())
    return elements, teams, fixtures

def current_gameweek(fixtures_df: pd.DataFrame) -> int | None:
    if "event" not in fixtures_df.columns: return None
    ev = fixtures_df["event"].dropna()
    return int(ev.min()) if not ev.empty else None

def team_fixture_difficulty_and_dgw(elements: pd.DataFrame,
                                    fixtures_df: pd.DataFrame) -> Tuple[Dict[int,int], Set[int]]:
    """Return per-player next fixture difficulty (1..5) and set of team_ids with DGW in any event."""
    next_diff: Dict[int,int] = {}
    dgw_teams: Set[int] = set()

    if "event" in fixtures_df.columns:
        tmp = fixtures_df.dropna(subset=["event"]).copy()
        ev_counts = (
            pd.concat([
                tmp[["event","team_h"]].rename(columns={"team_h":"team"}),
                tmp[["event","team_a"]].rename(columns={"team_a":"team"})
            ])
            .groupby(["event","team"]).size().reset_index(name="n")
        )
        dgw_teams = set(ev_counts.loc[ev_counts["n"] > 1, "team"].unique())

    for _, row in elements.iterrows():
        team_id = int(row["team"]); pid = int(row["id"])
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
        next_diff[pid] = int(fi["team_h_difficulty"] if int(fi["team_h"]) == team_id else fi["team_a_difficulty"])
    return next_diff, dgw_teams

# -------- injuries --------
def fetch_injured_players() -> Set[str]:
    url = "https://www.premierinjuries.com/injury-table.php"
    r = safe_get(url, timeout=10, retries=3, sleep_base=1.2)
    injured: Set[str] = set()
    if r is None: return injured
    soup = BeautifulSoup(r.text, "html.parser")
    for row in soup.select("table tr"):
        tds = row.find_all("td")
        if not tds: continue
        nm = tds[0].get_text(" ", strip=True).split("\n")[0].strip()
        if nm: injured.add(nm.lower())
    return injured

# -------- news sentiment --------
NEWS_SITES = [
    "https://www.bbc.com/sport/football",
    "https://www.theguardian.com/football",
    "https://www.skysports.com/football/news",
]

def analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()

def player_mentality_from_news(names: List[str], max_links_per_site: int = 10) -> Dict[str,str]:
    sia = analyzer()
    mentality = {n: "Stable" for n in names}
    for site in NEWS_SITES:
        rs = safe_get(site, timeout=10, retries=3, sleep_base=1.0)
        if rs is None: continue
        soup = BeautifulSoup(rs.text, "html.parser")
        links = soup.find_all("a", href=True)
        arts = []
        for a in links:
            href = a["href"]
            if "/football" in href and len(href) > 20:
                if href.startswith("/"): href = f"https://{site.split('/')[2]}{href}"
                if href not in arts: arts.append(href)
        for link in arts[:max_links_per_site]:
            ar = safe_get(link, timeout=10, retries=2, sleep_base=1.2)
            if ar is None: continue
            s2 = BeautifulSoup(ar.text, "html.parser")
            desc = s2.find("meta", attrs={"name":"description"})
            text = desc["content"] if (desc and "content" in desc.attrs) else \
                   " ".join(p.get_text(" ", strip=True) for p in s2.find_all("p")[:2])
            score = sia.polarity_scores(text).get("compound", 0.0)
            if score < -0.3:
                low = text.lower()
                for n in names:
                    if n.lower() in low: mentality[n] = "Volatile"
            time.sleep(1.0 + random.random()*0.7)
    return mentality

# -------- pre-season (GW <= 5) --------
def preseason_boost_from_news(names: List[str]) -> Dict[str,float]:
    """
    Friendlies/pre-season proxy from news:
      - If article mentions 'pre-season'/'preseason'/'friendly' + player name
        and sentiment is positive → boost 1.10
        and negative → 0.90
    """
    sia = analyzer()
    boost = {n: 1.0 for n in names}
    for site in NEWS_SITES:
        rs = safe_get(site, timeout=10, retries=3, sleep_base=1.0)
        if rs is None: continue
        soup = BeautifulSoup(rs.text, "html.parser")
        arts = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/football" in href and len(href) > 20:
                if href.startswith("/"): href = f"https://{site.split('/')[2]}{href}"
                if href not in arts: arts.append(href)
        for link in arts[:10]:
            ar = safe_get(link, timeout=10, retries=2, sleep_base=1.2)
            if ar is None: continue
            s2 = BeautifulSoup(ar.text, "html.parser")
            desc = s2.find("meta", attrs={"name":"description"})
            text = desc["content"] if (desc and "content" in desc.attrs) else \
                   " ".join(p.get_text(" ", strip=True) for p in s2.find_all("p")[:3])
            low = text.lower()
            if ("pre-season" in low) or ("preseason" in low) or ("friendly" in low):
                sc = sia.polarity_scores(text).get("compound", 0.0)
                for n in names:
                    if n.lower() in low:
                        if sc > 0.3:  boost[n] = max(boost[n], 1.10)
                        elif sc < -0.3: boost[n] = min(boost[n], 0.90)
            time.sleep(1.0 + random.random()*0.6)
    return boost

# -------- main builder --------
def load_players_df(apply_preseason_if_gw_le5: bool = True) -> pd.DataFrame:
    elements, teams, fixtures_df = load_fpl_raw()
    next_diff_map, dgw_teams = team_fixture_difficulty_and_dgw(elements, fixtures_df)
# Build next-opponent maps
    team_next_opp, team_home_away, team_next_gw = build_next_opponent_maps(fixtures_df)

    df = elements.merge(teams[["id","name","short_name","strength"]],
                        left_on="team", right_on="id", suffixes=("", "_team"))
# Map readable opponent + H/A
# opponent short_name requires a map from team_id -> short_name
team_id_to_short = dict(zip(teams["id"], teams["short_name"]))

df["TeamShort"] = df["short_name"]          # your club short name
df["NextGW"]    = df["team"].map(team_next_gw).fillna(-1).astype(int)
df["HomeAway"]  = df["team"].map(team_home_away).fillna("")
df["OppTeamId"] = df["team"].map(team_next_opp).fillna(-1).astype(int)
df["OppTeam"]   = df["OppTeamId"].map(team_id_to_short).fillna("")

# A compact “Next” column like: "BHA (H)" or "MCI (A)"
df["NextOpponent"] = df.apply(
    lambda r: f"{r['OppTeam']} ({r['HomeAway']})" if r["OppTeam"] else "",
    axis=1
)

    # basics
    df["Price"] = df["now_cost"] / 10.0
    df["Ownership"] = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0.0)
    df["Differential"] = df["Ownership"] < 10.0
    df["FixtureDifficulty"] = df["id"].map(next_diff_map).fillna(3).astype(int)
    df["DoubleGW"] = df["team"].apply(lambda t: t in dgw_teams)

    # injuries
    injured = fetch_injured_players()
    df["InjuryRisk"] = df["web_name"].str.lower().apply(lambda n: "Yes" if n in injured else "No")

    # mentality
    names = df["web_name"].tolist()
    mentality_map = player_mentality_from_news(names)
    df["Mentality"] = df["web_name"].apply(lambda n: mentality_map.get(n, "Stable"))

    # rotation risk
    df["chance_of_playing_next_round"] = pd.to_numeric(df["chance_of_playing_next_round"], errors="coerce").fillna(100)
    df["RotationRisk"] = ((df["minutes"] < 90) | (df["chance_of_playing_next_round"] < 75)).map({True:"Yes", False:"No"})

    # multipliers
    df["FatigueFactor"]   = df["minutes"].apply(lambda m: 0.95 if m > 270 else 1.0)
    df["ChemistryFactor"] = pd.to_numeric(df["influence"], errors="coerce").fillna(0.0).apply(lambda x: 1.0 if x > 300 else 0.95)
    df["InjuryFactor"]    = df["InjuryRisk"].map({"Yes":0.85, "No":1.0})
    df["MentalityFactor"] = df["Mentality"].map({"Volatile":0.95, "Stable":1.0})
    df["RotationFactor"]  = df["RotationRisk"].map({"Yes":0.92, "No":1.0})
    df["FixtureFactor"]   = df["FixtureDifficulty"].apply(lambda x: 1.05 if x <= 2 else (0.95 if x >= 4 else 1.0))
    df["DoubleGWFactor"]  = df["DoubleGW"].map({True:1.15, False:1.0})

    # ---------- Robust scoring (handles GW1 zeros) ----------
    df["form_num"] = pd.to_numeric(df["form"], errors="coerce").fillna(0.0)
    df["ppg"]      = pd.to_numeric(df.get("points_per_game", 0), errors="coerce").fillna(0.0)
    df["ict_num"]  = pd.to_numeric(df.get("ict_index", 0), errors="coerce").fillna(0.0)

    # Fallback when form==0 (common in GW1): 70% last-season PPG + 30% scaled ICT
    fallback_form = 0.7 * df["ppg"] + 0.3 * (df["ict_num"] / 10.0)

    # Pre-season factor (multiplier) + additive nudge in GW <= 5
    gw = current_gameweek(fixtures_df)
    df["PreSeasonFactor"] = 1.0
    preseason_add = 0.0
    if gw is not None and gw <= 5 and apply_preseason_if_gw_le5:
        pre = preseason_boost_from_news(names)  # 0.90 / 1.0 / 1.10
        df["PreSeasonFactor"] = df["web_name"].apply(lambda n: pre.get(n, 1.0))
        # Convert positive pre-season signal into a small additive bump so 0.0 form doesn't zero it out
        preseason_add = 0.5  # +0.5 pts if positive signal

    # BaseForm: if FPL form available use it; otherwise use fallback; then add small preseason nudge when boosted
    df["BaseForm"] = np.where(df["form_num"] > 0, df["form_num"], fallback_form)
    df["BaseForm"] = df["BaseForm"] + np.where(df["PreSeasonFactor"] > 1.0, preseason_add, 0.0)
    # If negative preseason (0.90), add a tiny negative nudge to dampen hype
    df["BaseForm"] = df["BaseForm"] - np.where(df["PreSeasonFactor"] < 1.0, 0.3, 0.0)

    # Final PredictedPoints: multiplicative factors * BaseForm
    df["PredictedPoints"] = (
        df["BaseForm"]
        * df["FatigueFactor"]
        * df["ChemistryFactor"]
        * df["InjuryFactor"]
        * df["MentalityFactor"]
        * df["RotationFactor"]
        * df["FixtureFactor"]
        * df["DoubleGWFactor"]
        * df["PreSeasonFactor"]
    )

    # Value metric: guard against zero price
    df["PPM"] = np.where(df["Price"] > 0, df["PredictedPoints"] / df["Price"], 0.0)

    # positions
    df["Position"] = df["element_type"].map({1:"GK", 2:"DEF", 3:"MID", 4:"FWD"})
    return df

def load_for_app() -> pd.DataFrame:
    return load_players_df(apply_preseason_if_gw_le5=True)
