
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(layout="wide")
st.title("⚽ Fantasy Premier League - AI Team Predictor (Personal Use)")

@st.cache_data
def load_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url)
    data = r.json()
    elements = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])[["id", "name", "strength", "short_name"]]
    fixtures = requests.get("https://fantasy.premierleague.com/api/fixtures/").json()
    fixtures_df = pd.DataFrame(fixtures)

    next_fixture_difficulty = {}
    double_gw_teams = set()

    for _, row in elements.iterrows():
        player_team_id = row["team"]
        player_id = row["id"]
        player_fixtures = fixtures_df[
            ((fixtures_df["team_h"] == player_team_id) | (fixtures_df["team_a"] == player_team_id)) &
            (fixtures_df["event"] != None)
        ]
        gw_counts = player_fixtures["event"].value_counts()
        if any(gw_counts > 1):
            double_gw_teams.add(player_team_id)

        if not player_fixtures.empty:
            next_gw = player_fixtures["event"].min()
            gw_fixture = player_fixtures[player_fixtures["event"] == next_gw].iloc[0]
            difficulty = gw_fixture["team_h_difficulty"] if gw_fixture["team_h"] == player_team_id else gw_fixture["team_a_difficulty"]
            next_fixture_difficulty[player_id] = difficulty
        else:
            next_fixture_difficulty[player_id] = 3  # Neutral

    elements["FixtureDifficulty"] = elements["id"].map(next_fixture_difficulty)
    elements["DoubleGW"] = elements["team"].apply(lambda x: x in double_gw_teams)
    elements = elements.merge(teams, left_on="team", right_on="id", suffixes=("", "_team"))
    elements["Price"] = elements["now_cost"] / 10
    elements["Ownership"] = elements["selected_by_percent"].astype(float)
    elements["Differential"] = elements["Ownership"] < 10
    return elements

@st.cache_data
def fetch_injured_players():
    url = "https://www.premierinjuries.com/injury-table.php"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, 'html.parser')
    injured = set()
    for row in soup.select("table tr"):
        cols = row.find_all("td")
        if len(cols) >= 2:
            name = cols[0].text.strip().split("\n")[0]
            if name:
                injured.add(name.lower())
    return injured

def fetch_player_mental_states(players):
    analyzer = SentimentIntensityAnalyzer()
    urls = [
        "https://www.bbc.com/sport/football",
        "https://www.theguardian.com/football",
        "https://www.skysports.com/football/news"
    ]
    mentality = {name: "Stable" for name in players["web_name"]}

    for url in urls:
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            links = soup.find_all('a', href=True)
            article_links = []

            for a in links:
                href = a["href"]
                if "/football" in href and len(href) > 20:
                    if href.startswith("/"):
                        href = "https://" + url.split("/")[2] + href
                    if href not in article_links:
                        article_links.append(href)

            for link in article_links[:10]:
                try:
                    time.sleep(1.5)
                    art = requests.get(link, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                    soup2 = BeautifulSoup(art.text, 'html.parser')
                    desc = soup2.find("meta", attrs={"name": "description"})
                    if desc and "content" in desc.attrs:
                        text = desc["content"]
                    else:
                        paragraphs = soup2.find_all("p")
                        text = " ".join(p.text for p in paragraphs[:2])
                    score = analyzer.polarity_scores(text)["compound"]
                    if score < -0.3:
                        for name in players["web_name"]:
                            if name.lower() in text.lower():
                                mentality[name] = "Volatile"
                except:
                    continue
        except:
            continue
    return mentality

# Load data
players = load_fpl_data()
injured_set = fetch_injured_players()
mentality_map = fetch_player_mental_states(players)

# Add calculated columns
players["InjuryRisk"] = players["web_name"].apply(lambda x: "Yes" if x.lower() in injured_set else "No")
players["Mentality"] = players["web_name"].apply(lambda x: mentality_map.get(x, "Stable"))
players["Minutes"] = players["minutes"]
players["FatigueFactor"] = players["Minutes"].apply(lambda m: 0.95 if m > 270 else 1.0)
players["ChemistryFactor"] = players["influence"].astype(float).apply(lambda x: 1.0 if x > 300 else 0.95)
players["InjuryFactor"] = players["InjuryRisk"].apply(lambda x: 0.85 if x == "Yes" else 1.0)
players["MentalityFactor"] = players["Mentality"].apply(lambda x: 0.95 if x == "Volatile" else 1.0)
players["FixtureFactor"] = players["FixtureDifficulty"].apply(lambda x: 1.05 if x <= 2 else (0.95 if x >= 4 else 1.0))
players["DoubleGWFactor"] = players["DoubleGW"].apply(lambda x: 1.15 if x else 1.0)

players["PredictedPoints"] = (
    players["form"].astype(float)
    * players["FatigueFactor"]
    * players["ChemistryFactor"]
    * players["InjuryFactor"]
    * players["MentalityFactor"]
    * players["FixtureFactor"]
    * players["DoubleGWFactor"]
)
players["PPM"] = players["PredictedPoints"] / players["Price"]

# Sidebar filters
positions = players["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
players["Position"] = positions
selected_position = st.sidebar.selectbox("Position", ["ALL", "GK", "DEF", "MID", "FWD"])
if selected_position != "ALL":
    players = players[players["Position"] == selected_position]

if st.sidebar.checkbox("Show only Differentials (<10%)"):
    players = players[players["Differential"] == True]

# Show top N players
top_n = st.sidebar.slider("Show top N players", 5, 50, 15)
st.subheader(f"Top {top_n} FPL Player Picks")
cols_to_show = [
    "web_name", "team", "Position", "Price", "form", "Ownership", "Differential",
    "PredictedPoints", "InjuryRisk", "Mentality", "FixtureDifficulty", "DoubleGW", "PPM"
]
st.dataframe(players.sort_values("PredictedPoints", ascending=False)[cols_to_show].head(top_n))
