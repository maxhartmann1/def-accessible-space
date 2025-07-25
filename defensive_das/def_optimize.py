from matplotlib import use
import streamlit as st
import numpy as np
import pandas as pd
import time
from accessible_space.utility import progress_bar
from defensive_das import prep_game
import defensive_das
import random
from scipy.interpolate import LinearNDInterpolator


POSS_TO_DEF = {"home": "away", "away": "home"}
TEAM_COLORS = {"home": "blue", "away": "red"}
MAX_RADIUS = 5.0
STEP_SIZE = 1.0
MIN_TEAMMATE_DIST = 2.0
SAMPLE_N = 20


# @st.cache_data
# def get_pre_frames(df, fps, frame=None, frame_list=[]):
#     if frame:
#         df_before = df[df["frame"] == frame - fps]
#         return df_before
#     pre_frame_list = frame_list - fps
#     df_before = df[df["frame"].isin(pre_frame_list)]
#     return df_before


def get_default_parameters():
    return MAX_RADIUS, STEP_SIZE, MIN_TEAMMATE_DIST, SAMPLE_N


def generate_dx_dy_combinations(max_radius, step_size):
    x_range = np.arange(-max_radius, max_radius + step_size, step_size)
    y_range = np.arange(-max_radius, max_radius + step_size, step_size)
    return [
        (dx, dy)
        for dx in x_range
        for dy in y_range
        if np.sqrt(dx**2 + dy**2) <= max_radius
    ]


def is_far_from_teammates(x, y, teammates, min_dist=2.0):
    return all(np.linalg.norm([x - tx, y - ty]) >= min_dist for tx, ty in teammates)


def estimate_das_from_pitch_map(x, y, pitch_map_func):
    return pitch_map_func(y, x)


def build_pitch_map_from_pitch_result(pitch_result, frame_list):
    pitch_map_func_per_frame = {}

    frame_indices = pitch_result.frame_index
    dangerous_result = pitch_result.dangerous_result
    das_volume = dangerous_result.attack_poss_density
    x_grid = dangerous_result.x_grid
    y_grid = dangerous_result.y_grid

    for i, frame in enumerate(frame_list):
        X = np.array(x_grid[i])
        Y = np.array(y_grid[i])
        Z = np.array(das_volume[i])

        if X.ndim != 2 or Y.ndim != 2 or Z.ndim != 2:
            st.warning("Dimension passt nicht für alle!")
            continue

        points = np.stack([X.flatten(), Y.flatten()], axis=-1)
        values = Z.flatten()

        if (
            points.shape[0] < 4
            or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2
        ):
            contiued_frames += 1
            continue

        interpolator = LinearNDInterpolator(
            points,
            values,
            fill_value=np.max(values),
        )
        pitch_map_func_per_frame[int(frame)] = lambda x, y, f=interpolator: f(x, y)

    return pitch_map_func_per_frame


def create_valid_positions(
    df_frameified,
    frame,
    player,
    player_team,
    dx_dy_combinations,
    x_center,
    y_center,
    min_teammate_dist,
):
    teammates = (
        df_frameified[
            (df_frameified["frame"] == frame)
            & (df_frameified["player_id"] != player)
            & (df_frameified["team_id"] == player_team)
        ][["player_x", "player_y"]]
        .dropna()
        .values.tolist()
    )
    valid_positions = [
        (dx, dy)
        for dx, dy in dx_dy_combinations
        if is_far_from_teammates(
            x_center + dx, y_center + dy, teammates, min_teammate_dist
        )
    ]
    return valid_positions


def create_interpolate_sample(
    valid_positions, x_center, y_center, pitch_func, sample_n
):
    scored_positions = []
    for dx, dy in valid_positions:
        new_x = x_center + dx
        new_y = y_center + dy
        score = estimate_das_from_pitch_map(new_x, new_y, pitch_func)
        scored_positions.append((dx, dy, score))

    top_combinations = sorted(scored_positions, key=lambda t: t[2], reverse=True)[
        :sample_n
    ]
    sample = [(x, y) for x, y, _ in top_combinations]
    return sample


@st.cache_data
def optimize_player_position(
    df_frameified,
    df_pre_frames,
    frame_list,
    player,
    pitch_result,
    method,
    all_frames=True,
    min_teammate_dist=MIN_TEAMMATE_DIST,
    consider_teammate_dist=True,
    sample_n=SAMPLE_N,
    max_radius=MAX_RADIUS,
    step_size=STEP_SIZE,
):
    simulated_positions = []

    start_time = time.time()

    # df_verteidiger enthält nur die Zeilen mit dem ausgewählten Spieler, in denen dieser als Verteidiger aktive ist
    player_team = df_frameified[df_frameified["player_id"] == player].iloc[0]["team_id"]
    df_verteidiger = (
        df_frameified[
            (df_frameified["team_possession"] != player_team)
            & (df_frameified["player_id"] == player)
        ]
        .dropna(subset=["player_x", "player_y"])
        .reset_index(drop=True)
    )
    def_frame_list = np.intersect1d(frame_list, df_verteidiger["frame"].values)

    if not all_frames:
        max_row = df_verteidiger.loc[df_verteidiger["DAS"].idxmax()]
        frame_at_max_das = max_row["frame"]

    dx_dy_combinations = generate_dx_dy_combinations(max_radius, step_size)
    if method == "interpolate":
        pitch_map_func_per_frame = build_pitch_map_from_pitch_result(
            pitch_result, frame_list
        )
    st.write(
        f"Spieler {player} in {len(def_frame_list)}/{len(frame_list)} als Verteidiger aktiv. Berechnung läuft..."
    )

    for idx, row in df_verteidiger.iterrows():
        frame = def_frame_list[idx]
        if not all_frames and frame != frame_at_max_das:
            continue
        pre_frame_data = df_pre_frames[df_pre_frames["frame"] == frame - 25]
        if pre_frame_data.empty:
            continue

        try:
            x_center, y_center = pre_frame_data[[f"{player}_x", f"{player}_y"]].values[
                0
            ]
        except IndexError:
            continue

        valid_positions = (
            create_valid_positions(
                df_frameified,
                frame,
                player,
                player_team,
                dx_dy_combinations,
                x_center,
                y_center,
                min_teammate_dist,
            )
            if consider_teammate_dist
            else dx_dy_combinations
        )

        if method == "random":
            sample = random.sample(valid_positions, min(sample_n, len(valid_positions)))
        elif method == "interpolate":
            pitch_func = pitch_map_func_per_frame.get(frame)
            if pitch_func is None:
                continue
            sample = create_interpolate_sample(
                valid_positions, x_center, y_center, pitch_func, sample_n
            )
        else:
            sample = valid_positions

        for dx, dy in sample:
            new_x = x_center + dx
            new_y = y_center + dy

            df_simulated_temp = (
                df_frameified[
                    (df_frameified["frame"] == frame)
                    & (df_frameified["player_id"] != player)
                ]
                .dropna(subset=["player_x", "player_y"])
                .reset_index(drop=True)
            )

            new_row = row.copy()
            new_row["player_x"] = new_x
            new_row["player_y"] = new_y

            df_simulated_temp = pd.concat(
                [df_simulated_temp, pd.DataFrame([new_row])],
                ignore_index=True,
            )

            df_simulated_temp["new_frame"] = f"{frame}_{player}_{dx:.2f}_{dy:.2f}"
            df_simulated_temp["opt_player"] = player
            simulated_positions.append(df_simulated_temp)

    df_frameified_simulated_position = pd.concat(simulated_positions, ignore_index=True)
    st.write(f"Frames erstellen in {time.time() - start_time:.2f} Sekunden")

    start_time = time.time()
    pitch_result_optimized = defensive_das.get_dangerous_accessible_space(
        df_frameified_simulated_position, frame_col="new_frame"
    )
    df_frameified_simulated_position["AS_new"] = pitch_result_optimized.acc_space
    df_frameified_simulated_position["DAS_new"] = pitch_result_optimized.das

    df_frameified_simulated_position = df_frameified_simulated_position.loc[
        df_frameified_simulated_position.groupby(["frame", "opt_player", "player_id"])[
            "DAS_new"
        ].idxmin()
    ]
    new_frame = df_frameified_simulated_position.iloc[0]["new_frame"]
    das_idx = df_frameified_simulated_position.index[0]
    res_idx = pitch_result_optimized.frame_index.iloc[das_idx]
    st.write(f"DAS berechnen mit {method} in {time.time() - start_time:.2f} Sekunden")
    return (
        df_frameified_simulated_position,
        pitch_result_optimized,
        # new_frame,
        # das_idx,
        # res_idx,
    )


# @st.cache_data
# def optimize_all_pos_in_reach(df_frameified, df_pre_frames, frame_list, player):
#     simulated_positions = []

#     start_time = time.time()

#     player_team = df_frameified[df_frameified["player_id"] == player].iloc[0]["team_id"]
#     df_verteidiger = (
#         df_frameified[
#             (df_frameified["team_possession"] != player_team)
#             & (df_frameified["player_id"] == player)
#         ]
#         .dropna(subset=["player_x", "player_y"])
#         .reset_index(drop=True)
#     )
#     def_frame_list = np.intersect1d(frame_list, df_verteidiger["frame"].values)

#     dx_dy_combinations = generate_dx_dy_combinations(MAX_RADIUS, STEP_SIZE)
#     st.write(
#         f"Spieler {player} in {len(def_frame_list)}/{len(frame_list)} als Verteidiger aktiv. Berechnung läuft..."
#     )

#     for idx, row in df_verteidiger.iterrows():
#         frame = def_frame_list[idx]
#         pre_frame_data = df_pre_frames[df_pre_frames["frame"] == frame - 25]
#         if pre_frame_data.empty:
#             continue

#         try:
#             x_center, y_center = pre_frame_data[[f"{player}_x", f"{player}_y"]].values[
#                 0
#             ]
#         except IndexError:
#             continue

#         for dx, dy in dx_dy_combinations:
#             new_x = x_center + dx
#             new_y = y_center + dy

#             df_simulated_temp = (
#                 df_frameified[
#                     (df_frameified["frame"] == frame)
#                     & (df_frameified["player_id"] != player)
#                 ]
#                 .dropna(subset=["player_x", "player_y"])
#                 .reset_index(drop=True)
#             )

#             new_row = row.copy()
#             new_row["player_x"] = new_x
#             new_row["player_y"] = new_y

#             df_simulated_temp = pd.concat(
#                 [df_simulated_temp, pd.DataFrame([new_row])],
#                 ignore_index=True,
#             )
#             df_simulated_temp["new_frame"] = f"{frame}_{player}_{dx:.2f}_{dy:.2f}"
#             df_simulated_temp["opt_player"] = player
#             simulated_positions.append(df_simulated_temp)

#     df_frameified_simulated_position = pd.concat(simulated_positions, ignore_index=True)
#     st.write(f"Frames erstellen in {time.time() - start_time:.2f} Sekunden")

#     start_time = time.time()
#     pitch_result_optimized = defensive_das.get_dangerous_accessible_space(
#         df_frameified_simulated_position, frame_col="new_frame"
#     )
#     df_frameified_simulated_position["AS_new"] = pitch_result_optimized.acc_space
#     df_frameified_simulated_position["DAS_new"] = pitch_result_optimized.das

#     df_frameified_simulated_position = df_frameified_simulated_position.loc[
#         df_frameified_simulated_position.groupby(["frame", "opt_player", "player_id"])[
#             "DAS_new"
#         ].idxmin()
#     ]
#     st.write(f"DAS Heuristik berechnen in {time.time() - start_time:.2f} Sekunden")
#     return df_frameified_simulated_position


# @st.cache_data
# def optimize_random_pos(
#     df_frameified, df_pre_frames, frame_list, player, sample_n=20, min_teammate_dist=2.0
# ):
#     simulated_positions = []

#     start_time = time.time()
#     player_team = df_frameified[df_frameified["player_id"] == player].iloc[0]["team_id"]
#     df_verteidiger = (
#         df_frameified[
#             (df_frameified["team_possession"] != player_team)
#             & (df_frameified["player_id"] == player)
#         ]
#         .dropna(subset=["player_x", "player_y"])
#         .reset_index(drop=True)
#     )
#     max_row = df_verteidiger.loc[df_verteidiger["DAS"].idxmax()]
#     max_das = max_row["DAS"]
#     frame_at_max_das = max_row["frame"]
#     def_frame_list = np.intersect1d(frame_list, df_verteidiger["frame"].values)
#     dx_dy_combinations = generate_dx_dy_combinations(MAX_RADIUS, STEP_SIZE)
#     st.write(
#         f"Spieler {player} in {len(def_frame_list)}/{len(frame_list)} als Verteidiger aktiv. Berechnung läuft..."
#     )
#     for idx, row in df_verteidiger.iterrows():
#         frame = def_frame_list[idx]
#         if frame != frame_at_max_das:
#             continue
#         pre_frame_data = df_pre_frames[df_pre_frames["frame"] == frame - 25]
#         if pre_frame_data.empty:
#             continue

#         try:
#             x_center, y_center = pre_frame_data[[f"{player}_x", f"{player}_y"]].values[
#                 0
#             ]
#         except IndexError:
#             continue

#         teammates = (
#             df_frameified[
#                 (df_frameified["frame"] == frame)
#                 & (df_frameified["player_id"] != player)
#                 & (df_frameified["team_id"] == player_team)
#             ][["player_x", "player_y"]]
#             .dropna()
#             .values.tolist()
#         )

#         valid_positions = [
#             (dx, dy)
#             for dx, dy in dx_dy_combinations
#             if is_far_from_teammates(
#                 x_center + dx, y_center + dy, teammates, min_teammate_dist
#             )
#         ]

#         sampled = random.sample(valid_positions, min(sample_n, len(valid_positions)))

#         for dx, dy in sampled:
#             new_x = x_center + dx
#             new_y = y_center + dy

#             df_simulated_temp = (
#                 df_frameified[
#                     (df_frameified["frame"] == frame)
#                     & (df_frameified["player_id"] != player)
#                 ]
#                 .dropna(subset=["player_x", "player_y"])
#                 .reset_index(drop=True)
#             )

#             new_row = row.copy()
#             new_row["player_x"] = new_x
#             new_row["player_y"] = new_y

#             df_simulated_temp = pd.concat(
#                 [df_simulated_temp, pd.DataFrame([new_row])],
#                 ignore_index=True,
#             )
#             df_simulated_temp["new_frame"] = f"{frame}_{player}_{dx:.2f}_{dy:.2f}"
#             df_simulated_temp["opt_player"] = player
#             simulated_positions.append(df_simulated_temp)

#     df_frameified_simulated_position = pd.concat(simulated_positions, ignore_index=True)
#     st.write(f"Frames erstellen in {time.time() - start_time:.2f} Sekunden")

#     start_time = time.time()
#     pitch_result_optimized = defensive_das.get_dangerous_accessible_space(
#         df_frameified_simulated_position, frame_col="new_frame"
#     )
#     df_frameified_simulated_position["AS_new"] = pitch_result_optimized.acc_space
#     df_frameified_simulated_position["DAS_new"] = pitch_result_optimized.das

#     df_frameified_simulated_position = df_frameified_simulated_position.loc[
#         df_frameified_simulated_position.groupby(["frame", "opt_player", "player_id"])[
#             "DAS_new"
#         ].idxmin()
#     ]
#     new_frame = df_frameified_simulated_position.iloc[0]["new_frame"]
#     das_idx = df_frameified_simulated_position.index[0]
#     res_idx = pitch_result_optimized.frame_index.iloc[das_idx]

#     st.write(f"DAS Random berechnen in {time.time() - start_time:.2f} Sekunden")
#     return (
#         df_frameified_simulated_position,
#         pitch_result_optimized,
#         new_frame,
#         das_idx,
#         res_idx,
#     )


# @st.cache_resource
# def optimize_topn_pitchdas(
#     df_frameified,
#     df_pre_frames,
#     frame_list,
#     player,
#     pitch_result,
#     top_n=20,
#     min_teammate_dist=2.0,
# ):
#     simulated_positions = []
#     pitch_map_func_per_frame = build_pitch_map_from_pitch_result(
#         pitch_result, frame_list
#     )

#     start_time = time.time()
#     player_team = df_frameified[df_frameified["player_id"] == player].iloc[0]["team_id"]
#     df_verteidiger = (
#         df_frameified[
#             (df_frameified["team_possession"] != player_team)
#             & (df_frameified["player_id"] == player)
#         ]
#         .dropna(subset=["player_x", "player_y"])
#         .reset_index(drop=True)
#     )
#     max_row = df_verteidiger.loc[df_verteidiger["DAS"].idxmax()]
#     max_das = max_row["DAS"]
#     frame_at_max_das = max_row["frame"]
#     def_frame_list = np.intersect1d(frame_list, df_verteidiger["frame"].values)
#     dx_dy_combinations = generate_dx_dy_combinations(MAX_RADIUS, STEP_SIZE)
#     st.write(
#         f"Spieler {player} in {len(def_frame_list)}/{len(frame_list)} als Verteidiger aktiv. Berechnung läuft..."
#     )
#     for idx, row in df_verteidiger.iterrows():
#         frame = def_frame_list[idx]
#         if frame != frame_at_max_das:
#             continue
#         pre_frame_data = df_pre_frames[df_pre_frames["frame"] == frame - 25]
#         if pre_frame_data.empty:
#             continue

#         try:
#             x_center, y_center = pre_frame_data[[f"{player}_x", f"{player}_y"]].values[
#                 0
#             ]
#         except IndexError:
#             continue

#         teammates = (
#             df_frameified[
#                 (df_frameified["frame"] == frame)
#                 & (df_frameified["player_id"] != player)
#                 & (df_frameified["team_id"] == player_team)
#             ][["player_x", "player_y"]]
#             .dropna()
#             .values.tolist()
#         )

#         pitch_func = pitch_map_func_per_frame.get(frame)
#         if pitch_func is None:
#             continue

#         valid_positions = [
#             (dx, dy)
#             for dx, dy in dx_dy_combinations
#             if is_far_from_teammates(
#                 x_center + dx, y_center + dy, teammates, min_teammate_dist
#             )
#         ]

#         scored_positions = []
#         for dx, dy in valid_positions:
#             new_x = x_center + dx
#             new_y = y_center + dy
#             score = estimate_das_from_pitch_map(new_x, new_y, pitch_func)
#             scored_positions.append((dx, dy, score))

#         top_combinations = sorted(scored_positions, key=lambda t: t[2], reverse=True)[
#             :top_n
#         ]

#         for dx, dy, _ in top_combinations:
#             new_x = x_center + dx
#             new_y = y_center + dy
#             df_simulated_temp = (
#                 df_frameified[
#                     (df_frameified["frame"] == frame)
#                     & (df_frameified["player_id"] != player)
#                 ]
#                 .dropna(subset=["player_x", "player_y"])
#                 .reset_index(drop=True)
#             )

#             new_row = row.copy()
#             new_row["player_x"] = new_x
#             new_row["player_y"] = new_y

#             df_simulated_temp = pd.concat(
#                 [df_simulated_temp, pd.DataFrame([new_row])],
#                 ignore_index=True,
#             )
#             df_simulated_temp["new_frame"] = f"{frame}_{player}_{dx:.2f}_{dy:.2f}"
#             df_simulated_temp["opt_player"] = player
#             simulated_positions.append(df_simulated_temp)

#     df_frameified_simulated_position = pd.concat(simulated_positions, ignore_index=True)
#     st.write(f"Frames erstellen in {time.time() - start_time:.2f} Sekunden")

#     start_time = time.time()
#     pitch_result_optimized = defensive_das.get_dangerous_accessible_space(
#         df_frameified_simulated_position, frame_col="new_frame"
#     )
#     df_frameified_simulated_position["AS_new"] = pitch_result_optimized.acc_space
#     df_frameified_simulated_position["DAS_new"] = pitch_result_optimized.das

#     df_frameified_simulated_position = df_frameified_simulated_position.loc[
#         df_frameified_simulated_position.groupby(["frame", "opt_player", "player_id"])[
#             "DAS_new"
#         ].idxmin()
#     ]
#     new_frame = df_frameified_simulated_position.iloc[0]["new_frame"]
#     das_idx = df_frameified_simulated_position.index[0]
#     res_idx = pitch_result_optimized.frame_index.iloc[das_idx]

#     st.write(f"DAS Interpolation berechnen in {time.time() - start_time:.2f} Sekunden")

#     return (
#         df_frameified_simulated_position,
#         pitch_result_optimized,
#         new_frame,
#         das_idx,
#         res_idx,
#     )
