import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
from pybaseball import statcast_batter, pitching_stats

st.set_page_config(page_title="MLB HR Model", layout="wide")

st.title("⚾ MLB HR Matchup Predictor (GitHub + Streamlit)")

# -----------------------------
# SETTINGS
# -----------------------------
days = st.slider("Recent Form Days", 7, 30, 7)
min_prob = st.slider("Min HR Probability", 0, 100, 50)

# -----------------------------
# MLB SCHEDULE + PITCHERS
# -----------------------------
@st.cache_data
def get_games():
    today = date.today()
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher"
    data = requests.get(url).json()

    games = []

    for d in data.get("dates", []):
        for g in d.get("games", []):
            games.append({
                "home_team": g["teams"]["home"]["team"]["name"],
                "away_team": g["teams"]["away"]["team"]["name"],
                "home_pitcher": g["teams"]["home"].get("probablePitcher", {}).get("fullName"),
                "away_pitcher": g["teams"]["away"].get("probablePitcher", {}).get("fullName"),
            })

    return pd.DataFrame(games)

games = get_games()

st.subheader("📅 Today's MLB Games")
st.dataframe(games)

# -----------------------------
# STATCAST DATA
# -----------------------------
@st.cache_data
def load_data(days):
    end = date.today()
    start = end - timedelta(days=days)

    df = statcast_batter(start_dt=start, end_dt=end)

    df = df[[
        "player_name",
        "launch_speed",
        "launch_angle",
        "events"
    ]].copy()

    df["HR"] = df["events"] == "home_run"
    return df

raw = load_data(days)

# -----------------------------
# HITTER STATS
# -----------------------------
def build_stats(df):
    g = df.groupby("player_name")

    stats = pd.DataFrame()
    stats["PA"] = g.size()

    stats["Barrel%"] = g.apply(
        lambda x: np.mean(
            (x["launch_speed"] > 98) & (x["launch_angle"].between(26, 30))
        )
    ) * 100

    stats["HardHit%"] = g.apply(
        lambda x: np.mean(x["launch_speed"] >= 95)
    ) * 100

    stats["FlyBall%"] = g.apply(
        lambda x: np.mean(x["launch_angle"] > 25)
    ) * 100

    return stats.reset_index()

hitters = build_stats(raw)

# -----------------------------
# SIMULATED MATCHUP DATA (SAFE VERSION)
# -----------------------------
np.random.seed(42)

hitters["Pitcher HR/9"] = np.random.uniform(1.0, 2.2, len(hitters))
hitters["Park Factor"] = np.random.uniform(0.85, 1.25, len(hitters))

def weather_boost():
    temp = np.random.uniform(10, 35)
    wind = np.random.uniform(0, 20)

    boost = 1.0
    if temp > 25:
        boost += 0.1
    if wind > 12:
        boost += 0.1

    return boost

hitters["Weather"] = [weather_boost() for _ in range(len(hitters))]

# -----------------------------
# HR MODEL
# -----------------------------
def model(row):
    return (
        row["Barrel%"] * 0.30 +
        row["HardHit%"] * 0.20 +
        row["FlyBall%"] * 0.20 +
        row["Pitcher HR/9"] * 20 +
        row["Park Factor"] * 10 +
        row["Weather"] * 10
    )

hitters["HR Score"] = hitters.apply(model, axis=1)
hitters["HR Probability"] = (
    hitters["HR Score"] / hitters["HR Score"].max()
) * 100

# -----------------------------
# FILTER PICKS
# -----------------------------
filtered = hitters[
    (hitters["PA"] > 20) &
    (hitters["HR Probability"] >= min_prob)
]

st.subheader("🔥 Top HR Picks")
st.dataframe(filtered.sort_values("HR Probability", ascending=False).head(10))

# -----------------------------
# CHART
# -----------------------------
st.subheader("📊 HR Leaderboard")
st.bar_chart(
    hitters.set_index("player_name")["HR Probability"]
    .sort_values(ascending=False)
    .head(15)
)