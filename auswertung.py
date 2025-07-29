import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path("simulation_results/reduced")


def prepare():
    csv_files = list(BASE_PATH.rglob("*.csv"))
    dataframes = []

    for file in csv_files:
        df = pd.read_csv(file)
        parts = file.parts[-5:]

        match_id = parts[0]
        step = parts[1]
        player = parts[2]
        method = parts[3]
        parameters = file.stem.replace("_reducedDF", "")

        df["match_id"] = match_id
        df["step"] = step
        df["player"] = player
        df["method"] = method
        df["parameters"] = parameters

        dataframes.append(df)

    full_df = pd.concat(dataframes, ignore_index=True)
    full_df["DAS_potential_percentage"] = np.where(
        full_df["DAS_potential"] > 0,
        ((full_df["DAS"] - full_df["DAS_new"]) / full_df["DAS"]) * 100,
        0.0,
    )

    return full_df


def group(full_df):
    grouped_df = (
        full_df.groupby(["match_id", "player", "step", "method", "parameters"])[
            ["DAS_potential", "DAS_potential_percentage", "distance"]
        ]
        .mean()
        .reset_index()
    )
    return grouped_df


def plot_results(grouped_df):

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=grouped_df, x="method", y="DAS_potential_percentage")
    plt.title("Vergleich der Methoden (DAS-Potential %)")
    plt.ylabel("Verbesserung (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    full_df = prepare()
    grouped_df = group(full_df)
    plot_results(grouped_df)
