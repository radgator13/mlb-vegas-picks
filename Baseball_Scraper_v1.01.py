import pandas as pd
import numpy as np
import requests
import os
from pybaseball import statcast
from sklearn.linear_model import LogisticRegression, LinearRegression
import streamlit as st
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Team Name Mapping ---
TEAM_NAME_MAP = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH"
}

ODDS_API_KEY = "8c20c59342e07c830e73aa8e6506b1c3"
ODDS_URL = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?regions=us&markets=spreads,totals,h2h&apiKey={ODDS_API_KEY}"

st.set_page_config(layout="wide")
st.title("⚾ MLB Picks vs Vegas")

target_date = st.date_input("Select Game Date:")
start_date = target_date - pd.Timedelta(days=60)
end_date = pd.to_datetime(target_date)

show_all = st.checkbox("🔍 Show all matchups (disable edge filter)", value=False)

@st.cache_resource(show_spinner=False)
def get_statcast_cached(start, end):
    df = statcast(start_dt=start.strftime('%Y-%m-%d'), end_dt=end.strftime('%Y-%m-%d'))
    df = df.dropna(subset=['home_team', 'away_team', 'launch_speed', 'launch_angle'])
    df['game_date'] = pd.to_datetime(df['game_date'])
    return df

statcast_df = get_statcast_cached(start_date, end_date)

def get_team_features(df):
    stats = df.groupby(['game_date', 'home_team']).agg(
        launch_speed=('launch_speed', 'mean'),
        launch_angle=('launch_angle', 'mean')
    ).reset_index()
    stats['team'] = stats['home_team']
    return stats.groupby('team').tail(2).groupby('team')[['launch_speed', 'launch_angle']].mean().reset_index()

team_rolling = get_team_features(statcast_df)

@st.cache_resource(show_spinner=False)
def fetch_vegas_odds():
    response = requests.get(ODDS_URL)
    if response.status_code != 200:
        st.error("Failed to pull odds from The Odds API")
        return pd.DataFrame()
    games = response.json()
    rows = []
    for g in games:
        row = {
            'home_team': g['home_team'],
            'away_team': g['away_team'],
            'matchup': f"{g['away_team']} @ {g['home_team']}",
            'spread': None,
            'total_line': None,
            'home_odds': None
        }
        for book in g.get('bookmakers', []):
            for market in book.get('markets', []):
                if market['key'] == 'spreads':
                    for o in market['outcomes']:
                        if o['name'] == g['home_team']:
                            row['spread'] = o.get('point')
                if market['key'] == 'totals' and market['outcomes']:
                    row['total_line'] = market['outcomes'][0].get('point')
                if market['key'] == 'h2h':
                    for o in market['outcomes']:
                        if o['name'] == g['home_team']:
                            row['home_odds'] = o['price']
        rows.append(row)
    return pd.DataFrame(rows)

odds_df = fetch_vegas_odds()
odds_df['home_team'] = odds_df['home_team'].map(TEAM_NAME_MAP)
odds_df['away_team'] = odds_df['away_team'].map(TEAM_NAME_MAP)
odds_df.dropna(subset=['home_team', 'away_team'], inplace=True)

def train_models(statcast_df, team_rolling):
    matchups = statcast_df.groupby(['game_date', 'home_team', 'away_team']).agg(
        home_score=('home_score', 'first'),
        away_score=('away_score', 'first')
    ).reset_index()

    matchups = matchups.merge(
        team_rolling.rename(columns={'team': 'home_team', 'launch_speed': 'home_launch_speed', 'launch_angle': 'home_launch_angle'}),
        on='home_team', how='inner')

    matchups = matchups.merge(
        team_rolling.rename(columns={'team': 'away_team', 'launch_speed': 'away_launch_speed', 'launch_angle': 'away_launch_angle'}),
        on='away_team', how='inner')

    matchups['diff_launch_speed'] = matchups['home_launch_speed'] - matchups['away_launch_speed']
    matchups['diff_launch_angle'] = matchups['home_launch_angle'] - matchups['away_launch_angle']
    matchups['is_home_win'] = (matchups['home_score'] > matchups['away_score']).astype(int)
    matchups['total_runs'] = matchups['home_score'] + matchups['away_score']

    X = matchups[['diff_launch_speed', 'diff_launch_angle']]
    y_win = matchups['is_home_win']
    y_total = matchups['total_runs']

    win_model = LogisticRegression(max_iter=1000).fit(X, y_win)
    total_model = LinearRegression().fit(X, y_total)
    return win_model, total_model

win_model, total_model = train_models(statcast_df, team_rolling)

def implied_prob(odds):
    return 100 / (abs(odds) + 100) if odds < 0 else abs(odds) / (abs(odds) + 100)

model_rows = []
for _, row in odds_df.iterrows():
    ht = row['home_team']
    at = row['away_team']

    default_stats = pd.Series({'launch_speed': 88.0, 'launch_angle': 12.0})

    home = team_rolling[team_rolling['team'] == ht].iloc[0] if ht in team_rolling['team'].values else default_stats
    away = team_rolling[team_rolling['team'] == at].iloc[0] if at in team_rolling['team'].values else default_stats

    features = pd.DataFrame([{
        'diff_launch_speed': home['launch_speed'] - away['launch_speed'],
        'diff_launch_angle': home['launch_angle'] - away['launch_angle']
    }])

    win_prob = win_model.predict_proba(features)[0][1]
    model_total = total_model.predict(features)[0]
    vegas_total = row['total_line']
    spread = row['spread']

    try:
        vegas_win_prob = implied_prob(row['home_odds']) if row['home_odds'] is not None else None
        edge_win = win_prob - vegas_win_prob if vegas_win_prob else None
    except:
        edge_win = None

    edge_total = model_total - vegas_total if vegas_total else None
    ou_pick = "Over" if edge_total and edge_total > 0 else "Under" if edge_total else "N/A"
    confidence = "High" if edge_win and edge_win > 0.10 else "Medium" if edge_win and edge_win > 0.07 else "Low"

    model_rows.append({
        'Game': row['matchup'],
        'Home Win % (Model)': f"{win_prob:.2%}",
        'Vegas Spread (Home)': f"{spread}" if spread is not None else "N/A",
        'Vegas Moneyline': row['home_odds'],
        'Win % Edge': edge_win,
        'Total Runs (Model)': round(model_total, 2),
        'Vegas Total Line': vegas_total,
        'O/U Pick': ou_pick,
        'Confidence': confidence,
        'Recommended': "🔥 Strong Pick" if edge_win and edge_win > 0.10 else ""
    })

results_df = pd.DataFrame(model_rows)

# 🔍 Diagnostic fallback
if results_df.empty:
    st.warning("⚠️ No predictions were generated. The odds or statcast data may be incomplete for this date.")
    st.stop()

# Ensure Win % Edge column exists
if 'Win % Edge' not in results_df.columns:
    results_df['Win % Edge'] = None

results_df['Win % Edge Num'] = pd.to_numeric(results_df['Win % Edge'], errors='coerce')

if not show_all:
    results_df = results_df[results_df['Win % Edge Num'] > 0.05]

if results_df.empty:
    st.warning("⚠️ No predictions passed the 5% edge filter.")
    st.stop()

results_df = results_df.sort_values(by='Win % Edge Num', ascending=False)
results_df['Win % Edge'] = results_df['Win % Edge Num'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else 'N/A')
results_df.drop(columns=['Win % Edge Num'], inplace=True)

# --- Render HTML Table (Centered) ---
def render_html_table(df):
    html = "<style>table { width: 100%; text-align: center; } th, td { text-align: center; padding: 8px; }</style>"
    html += df.to_html(index=False, escape=False)
    return html

st.subheader("📊 Model vs Vegas Picks")
st.markdown(render_html_table(results_df), unsafe_allow_html=True)

st.download_button(
    "Download Picks as CSV",
    data=results_df.to_csv(index=False),
    file_name="mlb_model_vs_vegas.csv"
)
