from paths import *
import pandas as pd
import os

sets = []

print(gcj_path)
for f in os.listdir(gcj_path):
    if f.endswith(".csv"):
        df = pd.read_csv(gcj_path + "/" + f)
        df_jo = df[df["file"].str.endswith(".java", na=False)]

        # filter data
        df_filtered = df_jo[df_jo["solution"] == 1].copy()
        df_filtered["linecount"] = df_filtered["flines"].str.len()
        df_filtered = df_filtered[df_filtered["linecount"] <= 100]
        sets.append(df_filtered)
        df_filtered.describe()

dataset = pd.concat(sets)
dataset.describe()
dataset
