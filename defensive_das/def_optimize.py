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

        pitch_result_temp = prep_game.get_dangerous_accessible_space(
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
            pitch_result_temp = prep_game.get_dangerous_accessible_space(
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
def single_player_das_optimization(df_frameified, df_pre_frames, frame_list, player):
    # Initialisierung Ziel Dataframe
    df_frameified_simulated_position = df_frameified.head(0).copy()

    start_time = time.time()
    # Filtern auf ausgewählten Spieler
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
    st.write(f"DAS Neu berechnen in {time.time() - start_time:.2f} Sekunden")
    return df_frameified_simulated_position


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


@st.cache_data
def optimize_all_pos_in_reach(df_frameified, df_pre_frames, frame_list, player):
    simulated_positions = []

    start_time = time.time()
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
    dx_dy_combinations = generate_dx_dy_combinations(MAX_RADIUS, STEP_SIZE)
    st.write(
        f"Spieler {player} in {len(def_frame_list)}/{len(frame_list)} als Verteidiger aktiv. Berechnung läuft..."
    )

    for idx, row in df_verteidiger.iterrows():
        frame = def_frame_list[idx]
        pre_frame_data = df_pre_frames[df_pre_frames["frame"] == frame - 25]
        if pre_frame_data.empty:
            continue

        try:
            x_center, y_center = pre_frame_data[[f"{player}_x", f"{player}_y"]].values[
                0
            ]
        except IndexError:
            continue

        for dx, dy in dx_dy_combinations:
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
    st.write(f"DAS Heuristik berechnen in {time.time() - start_time:.2f} Sekunden")
    return df_frameified_simulated_position


@st.cache_data
def optimize_random_pos(
    df_frameified, df_pre_frames, frame_list, player, sample_n=20, min_teammate_dist=2.0
):
    simulated_positions = []

    start_time = time.time()
    player_team = df_frameified[df_frameified["player_id"] == player].iloc[0]["team_id"]
    df_verteidiger = (
        df_frameified[
            (df_frameified["team_possession"] != player_team)
            & (df_frameified["player_id"] == player)
        ]
        .dropna(subset=["player_x", "player_y"])
        .reset_index(drop=True)
    )
    max_row = df_verteidiger.loc[df_verteidiger["DAS"].idxmax()]
    max_das = max_row["DAS"]
    frame_at_max_das = max_row["frame"]
    def_frame_list = np.intersect1d(frame_list, df_verteidiger["frame"].values)
    dx_dy_combinations = generate_dx_dy_combinations(MAX_RADIUS, STEP_SIZE)
    st.write(
        f"Spieler {player} in {len(def_frame_list)}/{len(frame_list)} als Verteidiger aktiv. Berechnung läuft..."
    )
    for idx, row in df_verteidiger.iterrows():
        frame = def_frame_list[idx]
        if frame != frame_at_max_das:
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

        sampled = random.sample(valid_positions, min(sample_n, len(valid_positions)))

        for dx, dy in sampled:
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

    st.write(f"DAS Random berechnen in {time.time() - start_time:.2f} Sekunden")
    return (
        df_frameified_simulated_position,
        pitch_result_optimized,
        new_frame,
        das_idx,
        res_idx,
    )


@st.cache_resource
def optimize_topn_pitchdas(
    df_frameified,
    df_pre_frames,
    frame_list,
    player,
    pitch_result,
    top_n=20,
    min_teammate_dist=2.0,
):
    simulated_positions = []
    pitch_map_func_per_frame = build_pitch_map_from_pitch_result(
        pitch_result, frame_list
    )

    start_time = time.time()
    player_team = df_frameified[df_frameified["player_id"] == player].iloc[0]["team_id"]
    df_verteidiger = (
        df_frameified[
            (df_frameified["team_possession"] != player_team)
            & (df_frameified["player_id"] == player)
        ]
        .dropna(subset=["player_x", "player_y"])
        .reset_index(drop=True)
    )
    max_row = df_verteidiger.loc[df_verteidiger["DAS"].idxmax()]
    max_das = max_row["DAS"]
    frame_at_max_das = max_row["frame"]
    def_frame_list = np.intersect1d(frame_list, df_verteidiger["frame"].values)
    dx_dy_combinations = generate_dx_dy_combinations(MAX_RADIUS, STEP_SIZE)
    st.write(
        f"Spieler {player} in {len(def_frame_list)}/{len(frame_list)} als Verteidiger aktiv. Berechnung läuft..."
    )
    for idx, row in df_verteidiger.iterrows():
        frame = def_frame_list[idx]
        if frame != frame_at_max_das:
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

        teammates = (
            df_frameified[
                (df_frameified["frame"] == frame)
                & (df_frameified["player_id"] != player)
                & (df_frameified["team_id"] == player_team)
            ][["player_x", "player_y"]]
            .dropna()
            .values.tolist()
        )

        pitch_func = pitch_map_func_per_frame.get(frame)
        if pitch_func is None:
            continue

        valid_positions = [
            (dx, dy)
            for dx, dy in dx_dy_combinations
            if is_far_from_teammates(
                x_center + dx, y_center + dy, teammates, min_teammate_dist
            )
        ]

        scored_positions = []
        for dx, dy in valid_positions:
            new_x = x_center + dx
            new_y = y_center + dy
            score = estimate_das_from_pitch_map(new_x, new_y, pitch_func)
            scored_positions.append((dx, dy, score))

        top_combinations = sorted(scored_positions, key=lambda t: t[2], reverse=True)[
            :top_n
        ]

        for dx, dy, _ in top_combinations:
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

    st.write(f"DAS Interpolation berechnen in {time.time() - start_time:.2f} Sekunden")

    return (
        df_frameified_simulated_position,
        pitch_result_optimized,
        new_frame,
        das_idx,
        res_idx,
    )
