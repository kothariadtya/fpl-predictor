# ✅ Streamlit FPL Team Predictor App with Injury & Sentiment Data
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article

st.set_page_config(page_title="FPL Team Predictor", layout="wide")
st.title("⚽ Fantasy Premier League - AI Team Picker")

# --- Load real-time FPL data ---
@st.cache_data
def load_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    data = requests.get(url).json()
    elements = pd.DataFrame(data['elements'])
    teams = pd.DataFrame(data['teams'])
    positions = pd.DataFrame(data['element_types'])

    elements['Team'] = elements['team'].map(teams.set_index('id')['name'])
    elements['Position'] = elements['element_type'].map(positions.set_index('id')['singular_name_short'])
    elements['Ownership'] = elements['selected_by_percent'].astype(float)
    elements['IsDifferential'] = elements['Ownership'] < 10
    elements['Price'] = elements['now_cost'] / 10

    elements['PredictedPoints'] = elements['form'].astype(float) * (
        elements['minutes'].apply(lambda m: 0.95 if m > 270 else 1.0)
    ) * (
        elements['influence'].astype(float).apply(lambda x: 1.0 if x > 300 else 0.95)
    )

    elements['PPM'] = elements['PredictedPoints'] / elements['Price']
    return elements

# --- Scrape injury data ---
def fetch_injuries():
    url = "https://www.premierinjuries.com/injury-table.php"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')

    injury_data = []
    for row in soup.select("tr.player-name-row"):
        try:
            player = row.select_one(".player-name").text.strip()
            reason = row.select_one(".player-injury").text.strip()
            injury_data.append(player)
        except:
            continue
    return injury_data

# --- Scrape news headlines for mentality sentiment ---
def fetch_sentiment_players():
    urls = [
        "https://www.bbc.com/sport/football",
        "https://www.theguardian.com/football",
        "https://www.skysports.com/football/news"
    ]
    sentiment_dict = {}
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text.lower()
            for name in players['web_name']:
                if name.lower() in text:
                    sentiment_dict[name] = "Volatile"
        except:
            continue
    return sentiment_dict

players = load_fpl_data()
injured_players = fetch_injuries()
sentiment_map = fetch_sentiment_players()

# Flag injury and morale
players['InjuryRisk'] = players['web_name'].apply(lambda name: name in injured_players)
players['InjuryMultiplier'] = players['InjuryRisk'].apply(lambda x: 0.85 if x else 1.0)
players['Mentality'] = players['web_name'].apply(lambda x: sentiment_map.get(x, 'Stable'))
players['MoraleMultiplier'] = players['Mentality'].apply(lambda x: 0.95 if x == 'Volatile' else 1.0)

# Adjust predictions
players['PredictedPoints'] *= players['InjuryMultiplier'] * players['MoraleMultiplier']

# --- Filters ---
st.sidebar.header("🔍 Filters")
position = st.sidebar.multiselect("Position", options=players['Position'].unique(), default=players['Position'].unique())
differential_only = st.sidebar.checkbox("Show only differentials (<10% owned)")
min_price = st.sidebar.slider("Minimum Price", 4.0, 12.5, 4.0)
max_price = st.sidebar.slider("Maximum Price", 4.0, 12.5, 12.5)

# --- Apply Filters ---
filtered = players[players['Position'].isin(position)]
filtered = filtered[(filtered['Price'] >= min_price) & (filtered['Price'] <= max_price)]
if differential_only:
    filtered = filtered[filtered['IsDifferential'] == True]

# --- Display Results ---
st.subheader("📋 Recommended Players")
st.dataframe(filtered.sort_values("PredictedPoints", ascending=False)[[
    "web_name", "Team", "Position", "Price", "Ownership", "InjuryRisk", "Mentality", "PredictedPoints", "PPM"
]].rename(columns={
    "web_name": "Name", "Price": "Price (M)"
}).reset_index(drop=True), use_container_width=True)

# --- Suggested Captain & VC ---
best = filtered.sort_values("PredictedPoints", ascending=False).head(2)
if len(best) >= 2:
    st.markdown("---")
    st.markdown(f"### 🧢 Suggested Captain: **{best.iloc[0]['web_name']}** ({round(best.iloc[0]['PredictedPoints'], 2)} pts)")
    st.markdown(f"### 👑 Vice-Captain: **{best.iloc[1]['web_name']}** ({round(best.iloc[1]['PredictedPoints'], 2)} pts)")
