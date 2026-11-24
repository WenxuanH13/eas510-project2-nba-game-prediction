import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("csv/processed_games.csv")
df["game_date"] = pd.to_datetime(df["game_date"])
df = df.sort_values(["game_date", "game_id"])


features = [
    "home_avg_pts",
    "home_avg_reb",
    "home_avg_ast",
    "home_avg_fg_pct",
    "home_win_pct",
    "away_avg_pts",
    "away_avg_reb",
    "away_avg_ast",
    "away_avg_fg_pct",
    "away_win_pct"
]

X = df[features]
y = df["home_win"]

# 80% training, 20% testing
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training games:", len(X_train))
print("Testing games:", len(X_test))


models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print(f"\n{name}")
    print(f"Accuracy:  {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall:    {recall}")
    print(f"F1-score:  {f1}")
