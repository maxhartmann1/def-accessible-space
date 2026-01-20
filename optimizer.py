from time import time
from scipy import linalg
import numpy as np
import pandas as pd
import random
import accessible_space
import logging
import sys


logging.basicConfig(
    filename="berechnung.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def generate_dx_dy_combinations(params, fine_radius=None, fine_step=None):
    max_radius = fine_radius if fine_radius else params["max_radius"]
    step_size = fine_step if fine_step else params["opt_step_size"]
    x_range = np.arange(-max_radius, max_radius + 1e-12, step_size)
    y_range = np.arange(-max_radius, max_radius + 1e-12, step_size)
    return [
        (float(dx), float(dy))
        for dx in x_range
        for dy in y_range
        if np.sqrt(dx**2 + dy**2) <= max_radius
    ]


def _create_valid_positions(
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
        if _is_far_from_teammates(x_center + dx, y_center + dy, teammates, min_dist)
    ]
    return valid_positions


def _is_far_from_teammates(x, y, teammates, min_dist):
    return all(np.linalg.norm([x - tx, y - ty]) >= min_dist for tx, ty in teammates)


def _optimize_random(
    df_frameified,
    is_random,
    valid_positions,
    n,
    x_center,
    y_center,
    frame,
    player,
    row,
    simulated_positions,
):
    if is_random:
        sample = random.sample(valid_positions, min(n, len(valid_positions)))
    else:
        sample = valid_positions

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

    return simulated_positions, len(sample)


def _optimize_grid_search(
    df_frameified,
    df_def_frames,
    def_frame_list,
    df_pre_frames,
    params,
    player,
    player_team,
    simulated_positions,
    fine_search=True,
):
    coarse_step = params.get("opt_step_size", 1.5)
    coarse_radius = params.get("max_radius", 6.0)
    coarse_top_k = params.get("n", 5)
    min_dist = params.get("min_dist", 2)
    fine_radius = 0.7 * coarse_step
    fine_step = 0.25 * fine_radius

    frame_to_row = {}
    frames_with_predata = []

    for idx, row in df_def_frames.iterrows():
        frame = def_frame_list[idx]
        pre_frame_data = df_pre_frames[df_pre_frames["frame"] == frame - 25]
        if pre_frame_data.empty:
            continue

        x_center, y_center = pre_frame_data[[f"{player}_x", f"{player}_y"]].values[0]
        frame_to_row[frame] = row.copy()
        frames_with_predata.append((frame, float(x_center), float(y_center)))

    if not frames_with_predata:
        print("Keine Frames mit Pre-Frame Data für Grid gefunden.")
        sys.exit(0)

    coarse_dxdy = generate_dx_dy_combinations(params)
    coarse_simulated_all = []

    for frame, x_center, y_center in frames_with_predata:
        valid_positions = _create_valid_positions(
            df_frameified,
            frame,
            player,
            player_team,
            coarse_dxdy,
            x_center,
            y_center,
            min_dist,
        )
        base_frame_df = (
            df_frameified[
                (df_frameified["frame"] == frame)
                & (df_frameified["player_id"] != player)
            ]
            .dropna(subset=["player_x", "player_y"])
            .reset_index(drop=True)
        )

        for dx, dy in valid_positions:
            new_x, new_y = x_center + dx, y_center + dy
            new_row = frame_to_row[frame].copy()
            new_row["player_x"] = new_x
            new_row["player_y"] = new_y
            df_sim_temp = pd.concat(
                [base_frame_df, pd.DataFrame([new_row])], ignore_index=True
            )
            df_sim_temp["new_frame"] = f"{frame}_{player}_{dx:.2f}_{dy:.2f}"
            df_sim_temp["opt_player"] = player
            coarse_simulated_all.append(df_sim_temp)

    if not fine_search or not coarse_simulated_all:
        return coarse_simulated_all, len(valid_positions)

    df_coarse = pd.concat(coarse_simulated_all, ignore_index=True)
    pitch_coarse = accessible_space.get_dangerous_accessible_space(
        df_coarse,
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
    df_coarse["AS_new"] = pitch_coarse.acc_space
    df_coarse["DAS_new"] = pitch_coarse.das

    # idx_best = df_coarse.groupby("new_frame")["DAS_new"].idxmax()
    df_coarse_player = df_coarse[df_coarse["player_id"] == player]
    frame_to_top_centers = {}
    for f, g in df_coarse_player.groupby("frame"):
        topk = g.sort_values("DAS_new", ascending=True).head(coarse_top_k)
        frame_to_top_centers[f] = topk[["player_x", "player_y"]].values.tolist()

    fine_dxdy = generate_dx_dy_combinations(
        params, fine_radius=fine_radius, fine_step=fine_step
    )

    for frame, x_center, y_center in frames_with_predata:
        top_centers = frame_to_top_centers.get(frame, [])
        if not top_centers:
            continue
        base_frame_df = (
            df_frameified[
                (df_frameified["frame"] == frame)
                & (df_frameified["player_id"] != player)
            ]
            .dropna(subset=["player_x", "player_y"])
            .reset_index(drop=True)
        )
        for cx, cy in top_centers:
            if (float(0), float(0)) not in fine_dxdy:
                fine_dxdy.append((float(0), float(0)))
            fine_valid_positions = _create_valid_positions(
                df_frameified, frame, player, player_team, fine_dxdy, cx, cy, min_dist
            )

            for dx, dy in fine_valid_positions:
                new_x, new_y = cx + dx, cy + dy
                new_row = frame_to_row[frame].copy()
                new_row["player_x"] = new_x
                new_row["player_y"] = new_y
                df_sim_temp = pd.concat(
                    [base_frame_df, pd.DataFrame([new_row])], ignore_index=True
                )
                df_sim_temp["new_frame"] = f"{frame}_{player}_{dx:2f}_{dy:.2f}"
                df_sim_temp["opt_player"] = player
                simulated_positions.append(df_sim_temp)
    return simulated_positions, (
        len(valid_positions),
        coarse_top_k,
        len(fine_valid_positions),
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
    # -------------------------------------------------------------
    # Methode Random
    if "random" in method or "all_positions" in method:
        for idx, row in df_def_frames.iterrows():
            frame = def_frame_list[idx]
            pre_frame_data = df_pre_frames[df_pre_frames["frame"] == frame - 25]

            if pre_frame_data.empty:
                print(
                    f"Für Frame {frame} konnten keine Pre-Frame Daten ermittelt werden. Aus Berechnung gestrichen."
                )
                continue

            x_center, y_center = pre_frame_data[[f"{player}_x", f"{player}_y"]].values[
                0
            ]
            valid_positions = _create_valid_positions(
                df_frameified,
                frame,
                player,
                player_team,
                dx_dy_combinations,
                x_center,
                y_center,
                params["min_dist"],
            )
            is_random = True if "random" in method else False
            simulated_positions, sample_size = _optimize_random(
                df_frameified,
                is_random,
                valid_positions,
                params["n"],
                x_center,
                y_center,
                frame,
                player,
                row,
                simulated_positions,
            )

    elif "grid" in method:
        simulated_positions, sample_size = _optimize_grid_search(
            df_frameified,
            df_def_frames,
            def_frame_list,
            df_pre_frames,
            params,
            player,
            player_team,
            simulated_positions,
        )
        # print(len(simulated_positions))
        # print(sample_size)
        # sys.exit(0)

    # Methode unbekannt
    else:
        print(f"ACHTUNG: Methode {method} nicht implementiert")
        sys.exit(0)

    df_frameified_simulated_position = pd.concat(simulated_positions, ignore_index=True)
    optimized_frame_list = df_frameified_simulated_position["new_frame"]

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
    df_frameified_optimized_position["comp_time"] = duration
    print(f"Pitch Result optimieren in  {time() - start_time:.2f} Sekunden")
    logging.info(
        f"Pitch Result Spieler {player}: frame step: {frame_step_size} | game: {game_id} | zeit: {duration:.4f} | anzahl frame: {len(def_frame_list)} | params: {params} | method: {method} | sample: {sample_size}"
    )
    return (
        df_frameified_simulated_position,
        df_frameified_optimized_position,
        pitch_result_optimized,
        optimized_frame_list,
    )
