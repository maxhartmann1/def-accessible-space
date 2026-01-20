import numpy as np
import pandas as pd
from accessible_space import get_dangerous_accessible_space
import sys  # löschen, nur debug


def compute_optimization(
    df_tracking, df_pre_frames, players, frame_list, config, column_schema
):
    for player in players:
        subset = df_tracking[df_tracking["player_id"] == player]
        if subset.empty:
            print(f"Keine Tracking Daten für Spieler {player}")
            continue

        pitch_result_optimized, frame_list_optimized, df_tracking_optimized = (
            _find_optimal_position(
                df_tracking, df_pre_frames, player, frame_list, config, column_schema
            )
        )
    return pitch_result_optimized, frame_list_optimized, df_tracking_optimized


def reduce_df_optimization(df, column_schema):
    frame_col = column_schema.frame_column

    df_reduced = df[df["opt_player"] == df["player_id"]][
        ["player_id", frame_col, "DAS", "DAS_new", "new_frame"]
    ]
    df_reduced["PDD_Absolute"] = df_reduced["DAS"] - df_reduced["DAS_new"]
    df_reduced["PDD"] = (df_reduced["DAS"] - df_reduced["DAS_new"]) / df_reduced["DAS"]
    return df_reduced


# Interne Funktionen
def _find_optimal_position(
    df_tracking, df_pre_frames, player, frame_list, config, column_schema
):
    frame_col = column_schema.frame_column
    team_col = column_schema.team_column
    period_col = column_schema.period_column
    team_possession_col = column_schema.team_possession_col
    player_in_possession_col = column_schema.player_in_possession_col
    frame_rate = config.frame_rate
    min_dist = config.min_distance_to_teammates
    simulated_positions = []

    player_team = df_tracking[df_tracking["player_id"] == player].iloc[0][team_col]
    df_defensive_frames = (
        df_tracking[
            (df_tracking[team_possession_col] != player_team)
            & (df_tracking["player_id"] == player)
        ]
        .dropna(subset=["player_x", "player_y"])
        .reset_index(drop=True)
    )
    def_frame_list = np.intersect1d(frame_list, df_defensive_frames[frame_col].values)

    dx_dy_combinations = _generate_dx_dy_combinations(config)

    for idx, row in df_defensive_frames.iterrows():
        frame = def_frame_list[idx]
        pre_frame_data = df_pre_frames[df_pre_frames[frame_col] == frame - frame_rate]
        if pre_frame_data.empty:
            print(
                f"Für Frame {frame} konnten keine Pre-Frame Daten ermittelt werden. Aus Berechnung gestrichen"
            )
            continue

        x_center, y_center = pre_frame_data[[f"{player}_x", f"{player}_y"]].values[0]
        valid_positions = _create_valid_positions(
            df_tracking,
            frame,
            player,
            player_team,
            dx_dy_combinations,
            x_center,
            y_center,
            min_dist,
            frame_col,
            team_col,
        )
        if (float(0), float(0)) not in valid_positions:
            valid_positions.append((float(0), float(0)))

        for dx, dy in valid_positions:
            new_x, new_y = x_center + dx, y_center + dy
            df_simulated_temp = (
                df_tracking[
                    (df_tracking[frame_col] == frame)
                    & (df_tracking["player_id"] != player)
                ]
                .dropna(subset=["player_x", "player_y"])
                .reset_index(drop=True)
            )
            new_row = row.copy()
            new_row["player_x"], new_row["player_y"] = new_x, new_y
            df_simulated_temp = pd.concat(
                [df_simulated_temp, pd.DataFrame([new_row])], ignore_index=True
            )
            df_simulated_temp["new_frame"] = f"{frame}_{player}_{dx:.2f}_{dy:.2f}"
            df_simulated_temp["opt_player"] = player
            simulated_positions.append(df_simulated_temp)

    df_tracking_simulated_positions = pd.concat(simulated_positions, ignore_index=True)
    optimized_frame_list = df_tracking_simulated_positions["new_frame"]

    pitch_result_optimized = get_dangerous_accessible_space(
        df_tracking_simulated_positions,
        frame_col="new_frame",
        player_col="player_id",
        ball_player_id="ball",
        x_col="player_x",
        y_col="player_y",
        vx_col="player_vx",
        vy_col="player_vy",
        team_col=team_col,
        period_col=period_col,
        team_in_possession_col=team_possession_col,
        player_in_possession_col=player_in_possession_col,
        attacking_direction_col=None,
        infer_attacking_direction=True,
        respect_offside=True,
        use_progress_bar=True,
    )
    df_tracking_simulated_positions["AS_new"] = pitch_result_optimized.acc_space
    df_tracking_simulated_positions["DAS_new"] = pitch_result_optimized.das

    df_tracking_optimized_positions = df_tracking_simulated_positions.loc[
        df_tracking_simulated_positions.groupby([frame_col, "opt_player", "player_id"])[
            "DAS_new"
        ].idxmin()
    ]

    return pitch_result_optimized, optimized_frame_list, df_tracking_optimized_positions


def _generate_dx_dy_combinations(config):
    max_radius = config.max_radius
    discretization_step = config.discretization_step
    x_range = np.arange(-max_radius, max_radius + 1e-12, discretization_step)
    y_range = np.arange(-max_radius, max_radius + 1e-12, discretization_step)
    return [
        (float(dx), float(dy))
        for dx in x_range
        for dy in y_range
        if np.sqrt(dx**2 + dy**2) <= max_radius
    ]


def _create_valid_positions(
    df_tracking,
    frame,
    player,
    player_team,
    dx_dy_combinations,
    x_center,
    y_center,
    min_dist,
    frame_col,
    team_col,
):
    teammates = (
        df_tracking[
            (df_tracking[frame_col] == frame)
            & (df_tracking["player_id"] != player)
            & (df_tracking[team_col] == player_team)
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
