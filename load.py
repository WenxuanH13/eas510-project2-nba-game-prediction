import sqlite3
import pandas as pd

conn = sqlite3.connect("data/nba.sqlite")

df = pd.read_sql_query("SELECT * FROM game LIMIT 5", conn)

print(df.to_string())
print("\nColumns:")
print(df.columns.tolist())

conn.close()