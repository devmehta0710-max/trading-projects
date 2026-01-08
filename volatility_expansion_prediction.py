"""
Project: Volatility Expansion Prediction

Objective:
This project predicts the probability of a large price move
and determines whether the next trading day is likely to be
a volatility expansion or consolidation day.

Approach:
- Historical price and volume data is used
- Features are derived from volatility, range, and returns
- A probabilistic model estimates the likelihood of expansion

Output:
- Probability of a big move
- Classification: Expansion / Non-Expansion

Use Case:
Helps traders anticipate breakout or trending days.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

symbol = "^GSPC"
data = yf.download(symbol, start="2010-01-01", progress=False)


if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data.dropna(inplace=True)


data["range"] = data["High"] - data["Low"]
data["return"] = data["Close"].pct_change()


data["tr1"] = data["High"] - data["Low"]
data["tr2"] = (data["High"] - data["Close"].shift(1)).abs()
data["tr3"] = (data["Low"] - data["Close"].shift(1)).abs()
data["true_range"] = data[["tr1", "tr2", "tr3"]].max(axis=1)
data["atr"] = data["true_range"].rolling(14).mean()


data["vol_10"] = data["return"].rolling(10).std()
data["vol_20"] = data["return"].rolling(20).std()


data["vol_ma_20"] = data["Volume"].rolling(20).mean()
data["volume_ratio"] = data["Volume"] / data["vol_ma_20"]


data["body"] = (data["Close"] - data["Open"]).abs()
data["upper_wick"] = data["High"] - data[["Close", "Open"]].max(axis=1)
data["lower_wick"] = data[["Close", "Open"]].min(axis=1) - data["Low"]

data["weekday"] = data.index.weekday
data["is_friday"] = (data["weekday"] == 4).astype(int)


data["range_ma_20"] = data["range"].rolling(20).mean()
data["tomorrow_range"] = data["range"].shift(-1)

data["big_move"] = (
    data["tomorrow_range"] > 1.25 * data["range_ma_20"]
).astype(int)


features = [
    "atr",
    "vol_10",
    "vol_20",
    "volume_ratio",
    "body",
    "upper_wick",
    "lower_wick",
    "is_friday"
]

df = data[features + ["big_move"]].dropna()

X = df[features]
y = df["big_move"]


split = int(0.8 * len(df))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)



latest_features = X.iloc[-1:]
probability = model.predict_proba(latest_features)[0][1]

print("\n==============================")
print(f"Latest data date: {data.index[-1].date()}")
print(f"Probability of BIG MOVE tomorrow: {probability:.2%}")

if probability < 0.4:
    print("Market Regime: CONSOLIDATION")
elif probability < 0.6:
    print("Market Regime: UNCERTAIN")
else:
    print("Market Regime: EXPANSION / TREND")
print("==============================")
