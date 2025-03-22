from matplotlib import use
import streamlit as st
import numpy as np
import pandas as pd

from accessible_space.utility import progress_bar
from defensive_das import prep_match
import defensive_das


POSS_TO_DEF = {"home": "away", "away": "home"}
TEAM_COLORS = {"home": "blue", "away": "red"}
MAX_RADIUS = 5
STEP_SIZE = 1


@st.cache_data
def get_pre_frames(df, fps, frame=None, frame_list=[]):
    if frame:
        df_before = df[df["frame"] == frame - fps]
        return df_before
    pre_frame_list = frame_list - fps
    df_before = df[df["frame"].isin(pre_frame_list)]
    return df_before


@st.cache_data
def calculate_optimal_das_one_frame(
    pitch_result, df_tracking, df_pre_frame, match, index_das, frame
):
    st.write("# Optimierung Verteidigungsposition")

    # Verteidiger filtern
    def_team = POSS_TO_DEF[
        df_tracking[df_tracking["frame"] == frame]["ball_possession"].iloc[0]
    ]
    df_verteidiger = (
        df_tracking[
            (df_tracking["frame"] == frame) & (df_tracking["team_id"] == def_team)
        ]
        .dropna(subset=["player_x", "player_y"])
        .reset_index(drop=True)
    )

    # Index Match-Objekt
    idx_match = match.tracking_data.index[
        match.tracking_data["frame"] == frame
    ].tolist()[0]

    # Variablen vereinfachen
    dangerous_result, frame_index = (
        pitch_result.dangerous_result,
        pitch_result.frame_index,
    )
    x_grid, y_grid, attack_poss_density = (
        dangerous_result.x_grid[index_das],
        dangerous_result.y_grid[index_das],
        dangerous_result.attack_poss_density[index_das],
    )
    df_tracking_oneframe = df_tracking[df_tracking["frame"] == frame]
    das_start = pitch_result.das.loc[index_das]
    match_optimized = match.copy()
    pitch_result_optimized = pitch_result

    for i, row in df_verteidiger.iterrows():
        df_tracking_temp = df_tracking_oneframe.copy()
        player_id = row["player_id"]
        x_center, y_center = df_pre_frame.loc[
            df_pre_frame["player_id"] == player_id, ["player_x", "player_y"]
        ].values[0]

        distances = np.sqrt((x_grid - x_center) ** 2 + (y_grid - y_center) ** 2)
        mask = distances <= MAX_RADIUS
        max_value = np.max(attack_poss_density[mask])
        max_indices = np.where((attack_poss_density == max_value) & mask)
        max_x, max_y = x_grid[max_indices][0], y_grid[max_indices][0]

        df_tracking_temp.loc[
            df_tracking_temp["player_id"] == player_id, ["player_x", "player_y"]
        ] = (max_x, max_y)

        pitch_result_temp = prep_match.get_dangerous_accessible_space(
            df_tracking_temp, use_progress_bar=False
        )
        das_temp = pitch_result_temp.das.iloc[0]

        if das_temp < das_start:
            df_tracking_oneframe = df_tracking_temp
            das_start = das_temp
            match_optimized.tracking_data.loc[
                match_optimized.tracking_data["frame"] == frame,
                [f"{player_id}_x", f"{player_id}_y"],
            ] = (max_x, max_y)
            pitch_result_optimized = pitch_result_temp

    return match_optimized, df_tracking_oneframe, pitch_result_optimized


@st.cache_data
def calculate_optimal_das_all_frames(
    pitch_result, df_tracking, df_pre_frames, match, frame_list
):
    dangerous_result = pitch_result.dangerous_result
    df_das_changes = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_frames = len(frame_list)

    for idx, frame in enumerate(frame_list):
        progress_bar.progress((idx + 1) / total_frames)
        status_text.text(f"Verarbeite Frame {idx+1} von {total_frames}...")

        def_team = POSS_TO_DEF[
            df_tracking[df_tracking["frame"] == frame]["ball_possession"].iloc[0]
        ]
        df_verteidiger = (
            df_tracking[
                (df_tracking["frame"] == frame) & (df_tracking["team_id"] == def_team)
            ]
            .dropna(subset=["player_x", "player_y"])
            .reset_index(drop=True)
        )
        x_grid, y_grid, attack_poss_density = (
            dangerous_result.x_grid[idx],
            dangerous_result.y_grid[idx],
            dangerous_result.attack_poss_density[idx],
        )
        das_start = pitch_result.das.loc[idx]
        das_opt = das_start
        df_pre_one_frame = df_pre_frames[
            df_pre_frames["frame"] == frame - match.frame_rate
        ]
        df_one_frame = df_tracking[df_tracking["frame"] == frame]
        for i, row in df_verteidiger.iterrows():
            df_tracking_temp = df_one_frame.copy()
            player_id = row["player_id"]
            x_center, y_center = df_pre_one_frame.loc[
                df_pre_one_frame["player_id"] == player_id, ["player_x", "player_y"]
            ].values[0]

            distances = np.sqrt((x_grid - x_center) ** 2 + (y_grid - y_center) ** 2)
            mask = distances <= MAX_RADIUS
            max_value = np.max(attack_poss_density[mask])
            max_indices = np.where((attack_poss_density == max_value) & mask)
            max_x, max_y = x_grid[max_indices][0], y_grid[max_indices][0]

            df_tracking_temp.loc[
                df_tracking_temp["player_id"] == player_id, ["player_x", "player_y"]
            ] = (max_x, max_y)
            pitch_result_temp = prep_match.get_dangerous_accessible_space(
                df_tracking_temp, use_progress_bar=False
            )
            das_temp = pitch_result_temp.das.iloc[0]
            if das_temp < das_opt:
                das_opt = das_temp
                df_one_frame = df_tracking_temp

        df_tracking.loc[df_tracking["frame"] == frame] = df_one_frame
        df_das_changes.append(
            {
                "frame": frame,
                "def_team": def_team,
                "das_start": das_start,
                "das_opt": das_opt,
            }
        )

    df_das_changes = pd.DataFrame(df_das_changes)
    progress_bar.empty()
    status_text.empty()
    return df_das_changes, df_tracking


@st.cache_data
def single_player_das_optimization(
    df_frameified, pitch_result, df_pre_frames, match, frame_list, player
):
    # Initialisierung Ziel Dataframe
    df_frameified_simulated_position = df_frameified.head(0).copy()

    # Filtern auf ausgewählten Spieler
    player_team = df_frameified[df_frameified["player_id"] == player].iloc[0]["team_id"]
    df_verteidiger = (
        df_frameified[
            (df_frameified["ball_possession"] != player_team)
            & (df_frameified["player_id"] == player)
        ]
        .dropna(subset=["player_x", "player_y"])
        .reset_index(drop=True)
    )
    def_frame_list = np.intersect1d(frame_list, df_verteidiger["frame"].values)
    st.write(
        f"Spieler {player} in {len(def_frame_list)}/{len(frame_list)} als Verteidiger aktiv. Berechnung läuft..."
    )

    for idx, row in df_verteidiger.iterrows():
        frame = def_frame_list[idx]
        x_center, y_center = df_pre_frames.loc[
            df_pre_frames["frame"] == frame - 25,
            [f"{player}_x", f"{player}_y"],
        ].values[0]
        x_range = np.arange(-MAX_RADIUS, MAX_RADIUS + STEP_SIZE, STEP_SIZE)
        y_range = np.arange(-MAX_RADIUS, MAX_RADIUS + STEP_SIZE, STEP_SIZE)
        for dx in x_range:
            for dy in y_range:
                if np.sqrt(dx**2 + dy**2) > MAX_RADIUS:
                    continue
                df_simulated_temp = (
                    df_frameified[
                        (df_frameified["frame"] == frame)
                        & (df_frameified["player_id"] != player)
                    ]
                    .dropna(subset=["player_x", "player_y"])
                    .reset_index(drop=True)
                )
                new_x, new_y = x_center + dx, y_center + dy

                new_row = row.copy()
                new_row["player_x"] = new_x
                new_row["player_y"] = new_y

                df_simulated_temp = pd.concat(
                    [df_simulated_temp, pd.DataFrame([new_row])],
                    ignore_index=True,
                )
                df_simulated_temp["new_frame"] = (
                    df_simulated_temp["frame"].astype(str)
                    + "_"
                    + player
                    + "_"
                    + str(dx)
                    + str(dy)
                )
                df_simulated_temp["opt_player"] = player
                # st.write(f"Shape von simulate temp {df_simulated_temp.shape}")
                # st.dataframe(df_simulated_temp)
                df_frameified_simulated_position = pd.concat(
                    [df_frameified_simulated_position, df_simulated_temp],
                    ignore_index=True,
                )

    pitch_result_optimized = defensive_das.get_dangerous_accessible_space(
        df_frameified_simulated_position, frame_col="new_frame"
    )
    df_frameified_simulated_position["AS_new"] = pitch_result_optimized.acc_space
    df_frameified_simulated_position["DAS_new"] = pitch_result_optimized.das
    # st.dataframe(df_frameified)
    df_frameified_simulated_position = df_frameified_simulated_position.loc[
        df_frameified_simulated_position.groupby(["frame", "opt_player", "player_id"])[
            "DAS_new"
        ].idxmin()
    ]
    return df_frameified_simulated_position
