import sqlite3
import pandas as pd

conn = sqlite3.connect("data/nba.sqlite")

df = pd.read_sql_query("""
    SELECT *
    FROM game
    WHERE game_date >= '2010-01-01'
      AND season_type = 'Regular Season'
""", conn)

conn.close()

df["game_date"] = pd.to_datetime(df["game_date"])
df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
df = df[df["wl_home"].isin(["W", "L"])].copy()

# 1 = home team wins, 0 = away team wins
df["home_win"] = (df["wl_home"] == "W").astype(int)

print("Total games:", len(df))

# stats to use
stats = ["pts", "reb", "ast", "fg_pct"]

# Convert statistics to numbers
for stat in stats:
    for side in ["home", "away"]:
        column = f"{stat}_{side}"
        df[column] = pd.to_numeric(df[column], errors="coerce")


team_games = []

for side in ["home", "away"]:
    team_df = df[[
        "game_id",
        "game_date",
        "season_id",
        f"team_id_{side}",
        f"wl_{side}"
    ] + [f"{stat}_{side}" for stat in stats]].copy()

    team_df.columns = [
        "game_id",
        "game_date",
        "season_id",
        "team_id",
        "wl"
    ] + stats

    team_games.append(team_df)

team_games = pd.concat(team_games, ignore_index=True)   # concat the games for both home and away teams
team_games = team_games.sort_values(
    ["season_id", "team_id", "game_date", "game_id"]
)

# 1 = win, 0 = loss
team_games["win"] = (team_games["wl"] == "W").astype(int)


# averages with previous game of current season, with a minimum of 5 games played
for stat in stats + ["win"]:
    team_games[f"avg_{stat}"] = (
        team_games.groupby(["season_id", "team_id"])[stat]
        .transform(
            lambda x: x.shift(1).expanding(min_periods=5).mean()
        )
    )

# Rename average wins to win percentage
team_games = team_games.rename(
    columns={"avg_win": "win_pct"}
)

features = [
    "avg_pts",
    "avg_reb",
    "avg_ast",
    "avg_fg_pct",
    "win_pct"
]


# Merge previous averages back into each game
for side in ["home", "away"]:
    team_features = team_games[
        ["game_id", "team_id"] + features
    ].copy()

    team_features = team_features.rename(
        columns={
            "team_id": f"team_id_{side}",
            **{f: f"{side}_{f}" for f in features}
        }
    )

    df = df.merge(
        team_features,
        on=["game_id", f"team_id_{side}"],
        how="left"
    )


# Remove rows with missing historical averages
model_features = [
    f"{side}_{feature}"
    for side in ["home", "away"]
    for feature in features
]

df = df.dropna(subset=model_features)

processed_df = df[[
    "game_id",
    "game_date",
    "season_id",
    "team_abbreviation_home",
    "team_abbreviation_away"
] + model_features + ["home_win"]]


# Save processed data
processed_df.to_csv("csv/processed_games.csv", index=False)

print("Total games after preprocessing:", len(processed_df))
print(processed_df.head().to_string(index=False))