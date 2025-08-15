
# 🧩 Best 15-Man FPL Team Selector
def build_best_team(df):
    from itertools import combinations
    budget = 100.0
    squad = []
    team_count = {}
    gks, defs, mids, fwds = [], [], [], []

    for _, row in df.iterrows():
        if row["Position"] == "GK":
            gks.append(row)
        elif row["Position"] == "DEF":
            defs.append(row)
        elif row["Position"] == "MID":
            mids.append(row)
        elif row["Position"] == "FWD":
            fwds.append(row)

    # Sort by predicted points per price
    gks.sort(key=lambda x: -x["PPM"])
    defs.sort(key=lambda x: -x["PPM"])
    mids.sort(key=lambda x: -x["PPM"])
    fwds.sort(key=lambda x: -x["PPM"])

    squad = gks[:2] + defs[:5] + mids[:5] + fwds[:3]
    total_cost = sum(p["Price"] for p in squad)
    if total_cost > budget:
        squad = sorted(squad, key=lambda x: -x["PPM"])
        for i in range(len(squad)-1, -1, -1):
            for alt in df[df["Position"] == squad[i]["Position"]].sort_values("PPM", ascending=False).itertuples():
                if alt.web_name == squad[i]["web_name"]:
                    continue
                if (total_cost - squad[i]["Price"] + alt.Price) <= budget:
                    if sum(1 for p in squad if p["team"] == alt.team) < 3:
                        total_cost = total_cost - squad[i]["Price"] + alt.Price
                        squad[i] = alt
                        break

    return pd.DataFrame(squad)

if st.button("📋 Build My FPL Team"):
    team_df = build_best_team(players)
    team_df = team_df.sort_values("PredictedPoints", ascending=False)
    team_df["Role"] = ["Captain", "Vice-Captain"] + ["First Team"] * 9 + ["Sub"] * 3
    st.subheader("🔝 Your Best 15-Man Team (Auto-Picked)")
    st.dataframe(team_df[["web_name", "Position", "team", "Price", "PredictedPoints", "Role"]])
