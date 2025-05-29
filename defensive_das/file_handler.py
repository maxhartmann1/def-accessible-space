from multiprocessing import Value
import pandas as pd
from sklearn.neighbors import VALID_METRICS
import streamlit as st
import os

CSV_PATH_DAS = "simulation_results/DAS/das.csv"


def _merge_rows_with_prefix(df, key_col="team_possession"):
    dfs = []
    for _, row in df.iterrows():
        prefix = row[key_col]
        row = row.drop(key_col)
        row.index = [f"{prefix}_{col}" for col in row.index]
        dfs.append(row)

    merged_row = pd.concat(dfs).to_frame().T
    return merged_row


def total_das_to_csv(df, frame_step_size, provider, game_id, frame_rate=25):
    df = df[["frame", "team_possession", "DAS"]]
    df = df.drop_duplicates(subset=["frame"], keep="first")
    df = df.groupby("team_possession").agg({"DAS": ["count", "sum", "mean", "max"]})
    df.columns = ["_".join(col).strip() for col in df.columns]
    df = df.reset_index()
    df_merged = _merge_rows_with_prefix(df)
    df_merged["frame_step_size"] = frame_step_size
    df_merged["provider"] = provider
    df_merged["game"] = game_id
    df_merged["intervall (sec)"] = frame_step_size / frame_rate
    df_merged = df_merged.rename(
        columns={
            "home_DAS_sum": "against_away_sum_DAS",
            "away_DAS_sum": "against_home_sum_DAS",
            "home_DAS_mean": "against_away_mean_DAS",
            "away_DAS_mean": "against_home_mean_DAS",
            "home_DAS_max": "against_away_max_DAS",
            "away_DAS_max": "against_home_max_DAS",
        }
    )
    df_merged = df_merged.reindex(
        [
            "provider",
            "game",
            "frame_step_size",
            "intervall (sec)",
            "home_DAS_count",
            "away_DAS_count",
            "against_home_sum_DAS",
            "against_away_sum_DAS",
            "against_home_mean_DAS",
            "against_away_mean_DAS",
            "against_home_max_DAS",
            "against_away_max_DAS",
        ],
        axis=1,
    )
    if len(df_merged) != 1:
        raise ValueError("df_merged muss genau 1 Zeile enthalten.")

    if os.path.exists(CSV_PATH_DAS):
        df_csv = pd.read_csv(CSV_PATH_DAS)
    else:
        folder = os.path.dirname(CSV_PATH_DAS)
        os.makedirs(folder, exist_ok=True)
        df_csv = pd.DataFrame()

    if not df_csv.empty:
        mask = df_csv["provider"] == df_merged.iloc[0]["provider"]
        for key_col in ["game", "frame_step_size"]:
            mask &= df_csv[key_col] == df_merged.iloc[0][key_col]
        if mask.any():
            df_csv.loc[mask, :] = df_merged.iloc[0].values
        else:
            df_csv = pd.concat([df_csv, df_merged], ignore_index=True)
    else:
        df_csv = df_merged.copy()

    df_csv = df_csv.sort_values(
        ["provider", "game", "frame_step_size"], ascending=[True, True, False]
    )
    df_csv.to_csv(CSV_PATH_DAS, index=False)
