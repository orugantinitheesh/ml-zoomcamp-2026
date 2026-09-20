import pandas as pd
import numpy as np

df = pd.read_csv("homework-01/dataset.csv")

print(df.head())
print(df.shape)
print(pd.__version__)
print(df["fuel_type"].unique())
print(df.isna().any().sum())
print(df["fuel_efficiency_mpg"].max())
print(df["horsepower"].median())
median_hp = df["horsepower"].median()

df["horsepower"] = df["horsepower"].fillna(median_hp)

print(df["horsepower"].median())
print(df["vehicle_weight"].sum())

cars_from_asia = df[df["origin"] == "Asia"]
cars_from_asia = cars_from_asia[["vehicle_weight", "model_year"]]
cars_from_asia = cars_from_asia.iloc[:7, :]

X = cars_from_asia.to_numpy()

XTX = np.dot(X.T, X)
XTX_inv = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = np.dot(np.dot(XTX_inv, X.T), y)

print(w)
print(w.sum())