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

# ---- Sidebar filters ONLY for the display table (not for optimizer) ----
selected_position = st.sidebar.selectbox("Position", ["ALL", "GK", "DEF", "MID", "FWD"])

table_df = players.copy()
if selected_position != "ALL":
    table_df = table_df[table_df["Position"] == selected_position]

if st.sidebar.checkbox("Show only Differentials (<10%)"):
    table_df = table_df[table_df["Differential"] == True]

top_n = st.sidebar.slider("Show top N players", 5, 60, 15)
st.subheader(f"Top {top_n} FPL Player Picks")
cols_to_show = [
    "web_name", "TeamShort", "Position", "Price", "form", "Ownership", "Differential",
    "PredictedPoints", "FixtureDifficulty", "DoubleGW", "NextOpponent", "InjuryRisk", "RotationRisk", "Mentality", "PPM"
]

st.dataframe(
    table_df.sort_values("PredictedPoints", ascending=False)[cols_to_show].head(top_n),
    use_container_width=True
)
# ----------------------------
# Best 15-Man FPL Team Selector (robust, DataFrame-only, respects all rules)
# ----------------------------
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

    cols = ["web_name", "TeamShort", "Position", "Price", "form", "Ownership", "Differential",
    "PredictedPoints", "FixtureDifficulty", "DoubleGW", "NextOpponent", "InjuryRisk", "RotationRisk", "Mentality", "PPM"]
    df = df_full[cols].dropna(subset=["Price","PredictedPoints","PPM"]).copy()

    pos_sorted = {
        p: df[df["Position"] == p].sort_values(["PPM","PredictedPoints"], ascending=False).reset_index(drop=True)
        for p in need.keys()
    }

    squad_rows = []
    team_counts = {}

    def total_cost(rows): return sum(r["Price"] for r in rows)
    def can_add(rowd):
        if team_counts.get(rowd["team"], 0) >= team_limit:
            return False
        if total_cost(squad_rows) + rowd["Price"] > budget:
            return False
        return True

    # pick by position
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

    # fill gaps cheaply if any
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
                # downgrade worst value in this position to free money
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
            squad_rows.append(cand)
            team_counts[cand["team"]] = team_counts.get(cand["team"], 0) + 1
            break

    return pd.DataFrame(squad_rows)
# ----------------------------
# Best XI + Bench + Chip Planner
# ----------------------------
VALID_FORMATIONS = [
    (3, 4, 3),
    (3, 5, 2),
    (4, 4, 2),
    (4, 3, 3),
    (5, 3, 2),
    (5, 4, 1),
]

def choose_best_xi(squad: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gks = squad[squad["Position"] == "GK"].sort_values("PredictedPoints", ascending=False)
    defs = squad[squad["Position"] == "DEF"].sort_values("PredictedPoints", ascending=False)
    mids = squad[squad["Position"] == "MID"].sort_values("PredictedPoints", ascending=False)
    fwds = squad[squad["Position"] == "FWD"].sort_values("PredictedPoints", ascending=False)

    if len(gks) < 2 or len(defs) < 5 or len(mids) < 5 or len(fwds) < 3:
        xi = squad.sort_values("PredictedPoints", ascending=False).head(11)
        bench = squad.loc[~squad.index.isin(xi.index)].sort_values("PredictedPoints", ascending=False)
        return xi, bench

    gk1 = gks.iloc[0:1]
    gk2 = gks.iloc[1:2]

    best_total = -1.0
    best_combo = None
    for d, m, f in VALID_FORMATIONS:
        if len(defs) < d or len(mids) < m or len(fwds) < f:
            continue
        xi_try = pd.concat([gk1, defs.head(d), mids.head(m), fwds.head(f)])
        total = xi_try["PredictedPoints"].sum()
        if total > best_total:
            best_total = total
            best_combo = xi_try

    if best_combo is None:
        xi = squad.sort_values("PredictedPoints", ascending=False).head(11)
        bench = squad.loc[~squad.index.isin(xi.index)].sort_values("PredictedPoints", ascending=False)
        return xi, bench

    xi = best_combo
    bench_pool = squad.loc[~squad.index.isin(xi.index)].copy()
    # Bench order: GK2 first, then two lowest-outfield by PredictedPoints
    outfield_bench = bench_pool[bench_pool["Position"] != "GK"].sort_values("PredictedPoints")
    bench = pd.concat([gk2, outfield_bench.head(2)], ignore_index=False)
    return xi, bench_pool.sort_values("PredictedPoints", ascending=False)

def count_risks(df: pd.DataFrame) -> int:
    injuries = (df["InjuryRisk"] == "Yes").sum()
    rotation = (df["RotationRisk"] == "Yes").sum()
    volatile = (df["Mentality"] == "Volatile").sum()
    return int(injuries + rotation + volatile)

def recommend_chips(xi: pd.DataFrame, bench_full: pd.DataFrame) -> dict:
    xi_sorted = xi.sort_values("PredictedPoints", ascending=False).reset_index(drop=True)
    cap = xi_sorted.iloc[0]
    vc  = xi_sorted.iloc[1] if len(xi_sorted) > 1 else None

    bench_strength = bench_full[bench_full["Position"] != "GK"].sort_values("PredictedPoints", ascending=False).head(3)["PredictedPoints"].sum()

    full15 = pd.concat([xi, bench_full], ignore_index=True)
    risk_count = count_risks(full15)

    cap_dgw = bool(cap.get("DoubleGW", False))
    cap_pts = float(cap["PredictedPoints"])
    vc_pts  = float(vc["PredictedPoints"]) if vc is not None else 0.0

    # Triple Captain
    tc_ok = False; tc_reason = ""
    if cap_dgw and cap_pts >= 8.0:
        tc_ok = True; tc_reason = f"Captain has a DGW and high projection ({cap_pts:.1f})."
    elif cap_pts >= 10.0 and (cap_pts - vc_pts) >= 1.5:
        tc_ok = True; tc_reason = f"Captain projects very high ({cap_pts:.1f}) with clear lead over VC ({vc_pts:.1f})."
    else:
        tc_reason = f"Hold TC: {cap['web_name']} at {cap_pts:.1f} pts (DGW={cap_dgw})."

    # Bench Boost
    bb_ok = False; bb_reason = ""
    if bench_strength >= 16.0 and risk_count <= 3:
        bb_ok = True; bb_reason = f"Bench projects {bench_strength:.1f} pts with low risk ({risk_count} risks)."
    else:
        bb_reason = f"Hold BB: Bench {bench_strength:.1f} pts or risks too high ({risk_count})."

    # Wildcard
    wc_ok = False; wc_reason = ""
    if risk_count >= 6:
        wc_ok = True; wc_reason = f"Wildcard flagged: {risk_count} risky players."
    else:
        wc_reason = f"No Wildcard: risk count acceptable ({risk_count})."

    return {
        "Triple Captain": (tc_ok, tc_reason, cap["web_name"]),
        "Bench Boost": (bb_ok, bb_reason, None),
        "Wildcard": (wc_ok, wc_reason, None),
    }

# --------------------------------------------------------------------------------------
# Streamlit setup
# --------------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="FPL Team Predictor")
st.title("BLRush Predictor Run")

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
if st.button("📋 Build My FPL Team"):
    # Build 15-man squad from the FULL dataset (ignores table filters)
    squad15 = build_best_team(players_full).reset_index(drop=True)

    # Best XI + Bench
    xi, bench_table = choose_best_xi(squad15)

    # Captain & VC in XI
    xi_sorted = xi.sort_values("PredictedPoints", ascending=False).reset_index(drop=True)
    if len(xi_sorted) >= 2:
        xi_sorted.loc[0, "Role"] = "Captain"
        xi_sorted.loc[1, "Role"] = "Vice-Captain"
    xi_sorted.loc[2:, "Role"] = "First Team"

    # Chip recommendations
    chips = recommend_chips(xi_sorted, bench_table)

    st.subheader("🔝 Starting XI (best formation)")
    st.dataframe(xi_sorted[["web_name","Position","TeamShort","NextOpponent","Price","PredictedPoints","InjuryRisk","RotationRisk","Mentality"]].head(4),
                 use_container_width = True

    st.subheader("🛋 Bench (ordered strongest first)")
    st.dataframe(
        bench_table[["web_name","Position","TeamShort","NextOpponent","Price","PredictedPoints","InjuryRisk","RotationRisk","Mentality"]].head(4),
        use_container_width=True
    )

    st.subheader("🧠 Chip Planner (this GW)")
    tc_ok, tc_reason, tc_cap = chips["Triple Captain"]
    bb_ok, bb_reason, _      = chips["Bench Boost"]
    wc_ok, wc_reason, _      = chips["Wildcard"]

    st.markdown(f"- **Triple Captain**: {'✅ Play' if tc_ok else '⏳ Hold'} — {tc_reason} (Captain: **{tc_cap}**)")
    st.markdown(f"- **Bench Boost**: {'✅ Play' if bb_ok else '⏳ Hold'} — {bb_reason}")
    st.markdown(f"- **Wildcard**: {'✅ Play' if wc_ok else '⏳ Hold'} — {wc_reason}")
