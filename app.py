# ✅ Streamlit FPL Team Predictor App with Injury Data
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

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

players = load_fpl_data()
injured_players = fetch_injuries()

# Flag injury risk
players['InjuryRisk'] = players['web_name'].apply(lambda name: name in injured_players)
players['InjuryMultiplier'] = players['InjuryRisk'].apply(lambda x: 0.85 if x else 1.0)
players['PredictedPoints'] *= players['InjuryMultiplier']

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
    "web_name", "Team", "Position", "Price", "Ownership", "InjuryRisk", "PredictedPoints", "PPM"
]].rename(columns={
    "web_name": "Name", "Price": "Price (M)"
}).reset_index(drop=True), use_container_width=True)

# --- Suggested Captain & VC ---
best = filtered.sort_values("PredictedPoints", ascending=False).head(2)
if len(best) >= 2:
    st.markdown("---")
    st.markdown(f"### 🧢 Suggested Captain: **{best.iloc[0]['web_name']}** ({round(best.iloc[0]['PredictedPoints'], 2)} pts)")
    st.markdown(f"### 👑 Vice-Captain: **{best.iloc[1]['web_name']}** ({round(best.iloc[1]['PredictedPoints'], 2)} pts)")
