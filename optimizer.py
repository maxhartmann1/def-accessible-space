from time import time
from altair import SampleTransform
from scipy import linalg
import numpy as np
import pandas as pd
import random
import accessible_space
import logging

logging.basicConfig(
    filename="berechnung.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def optimize_player_position(
    df_frameified,
    player,
    frame_list,
    params,
    df_pre_frames,
    method,
    game_id,
    frame_step_size,
):
    simulated_positions = []
    start_time = time()

    player_team = df_frameified[df_frameified["player_id"] == player].iloc[0]["team_id"]
    df_def_frames = (
        df_frameified[
            (df_frameified["team_possession"] != player_team)
            & (df_frameified["player_id"] == player)
        ]
        .dropna(subset=["player_x", "player_y"])
        .reset_index(drop=True)
    )
    def_frame_list = np.intersect1d(frame_list, df_def_frames["frame"].values)

    dx_dy_combinations = generate_dx_dy_combinations(params)

    print(
        f"Spieler {player} in {len(def_frame_list)}/{len(frame_list)} als Verteidiger aktiv. Optimierung startet..."
    )
    # --------------------------------------------------------------------------

    for idx, row in df_def_frames.iterrows():
        frame = def_frame_list[idx]
        pre_frame_data = df_pre_frames[df_pre_frames["frame"] == frame - 25]
        if pre_frame_data.empty:
            print(
                f"Für Frame {frame} konnten keine Pre Frame Daten ergmittel werden. Aus Berechnung gestrichen."
            )
            continue

        x_center, y_center = pre_frame_data[[f"{player}_x", f"{player}_y"]].values[0]

        valid_positions = create_valid_positions(
            df_frameified,
            frame,
            player,
            player_team,
            dx_dy_combinations,
            x_center,
            y_center,
            params["min_dist"],
        )
        if method == "random":
            sample = random.sample(
                valid_positions, min(params["n"], len(valid_positions))
            )
        else:
            sample = valid_positions
            logging.info(f"Sample-Größe bei all_positions für {player}: {len(sample)}.")
        for dx, dy in sample:
            new_x, new_y = x_center + dx, y_center + dy

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
                [df_simulated_temp, pd.DataFrame([new_row])], ignore_index=True
            )
            df_simulated_temp["new_frame"] = f"{frame}_{player}_{dx:.2f}_{dy:.2f}"
            df_simulated_temp["opt_player"] = player
            simulated_positions.append(df_simulated_temp)

    df_frameified_simulated_position = pd.concat(simulated_positions, ignore_index=True)
    optimized_frame_list = df_frameified_simulated_position["new_frame"]

    duration = time() - start_time
    print(f"Frames erstellen in  {duration:.4f} Sekunden")
    logging.info(
        f"Frames erstellen für {player} mit {len(def_frame_list)} Frames bei step size {frame_step_size} für {game_id} mit {method} in {duration:.4f} Sekunden"
    )
    start_time = time()

    pitch_result_optimized = accessible_space.get_dangerous_accessible_space(
        df_frameified_simulated_position,
        frame_col="new_frame",
        player_col="player_id",
        ball_player_id="ball",
        team_col="team_id",
        x_col="player_x",
        y_col="player_y",
        vx_col="player_vx",
        vy_col="player_vy",
        period_col="period_id",
        team_in_possession_col="team_possession",
        attacking_direction_col=None,
        infer_attacking_direction=True,
        additional_fields_to_return=None,
        use_progress_bar=True,
        respect_offside=True,
        player_in_possession_col="player_possession",
    )
    df_frameified_simulated_position["AS_new"] = pitch_result_optimized.acc_space
    df_frameified_simulated_position["DAS_new"] = pitch_result_optimized.das

    df_frameified_optimized_position = df_frameified_simulated_position.loc[
        df_frameified_simulated_position.groupby(["frame", "opt_player", "player_id"])[
            "DAS_new"
        ].idxmin()
    ]

    duration = time() - start_time
    print(f"Pitch Result optimieren in  {time() - start_time:.2f} Sekunden")
    logging.info(
        f"Pitch Result berechnen für {player} mit {len(def_frame_list)} Frames für {game_id} in {duration:.4f} Sekunden"
    )
    logging.info(f"Params des oben genannten Runs: {params}")

    return (
        df_frameified_simulated_position,
        df_frameified_optimized_position,
        pitch_result_optimized,
        optimized_frame_list,
    )


def generate_dx_dy_combinations(params):
    max_radius = params["max_radius"]
    step_size = params["opt_step_size"]
    x_range = np.arange(-max_radius, max_radius + step_size, step_size)
    y_range = np.arange(-max_radius, max_radius + step_size, step_size)
    return [
        (dx, dy)
        for dx in x_range
        for dy in y_range
        if np.sqrt(dx**2 + dy**2) <= max_radius
    ]


def create_valid_positions(
    df_frameified,
    frame,
    player,
    player_team,
    dx_dy_combinations,
    x_center,
    y_center,
    min_dist,
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
        if is_far_from_teammates(x_center + dx, y_center + dy, teammates, min_dist)
    ]
    return valid_positions


def is_far_from_teammates(x, y, teammates, min_dist):
    return all(np.linalg.norm([x - tx, y - ty]) >= min_dist for tx, ty in teammates)
