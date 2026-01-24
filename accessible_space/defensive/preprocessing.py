import pandas as pd
from accessible_space import (
    per_object_frameify_tracking_data,
    get_dangerous_accessible_space,
)
from collections import defaultdict
from .config import TrackingColumnSchema, PDDInputError
import sys  # Debug to delete


def compute_preprocessing(tracking_data, config, column_schema):
    df = pd.DataFrame(tracking_data.copy())

    # Filter nach Frame-Step, Ballbesitz
    df_tracking_filtered = _filter_tracking_data(
        df, config.frame_rate, config.frame_filter_rate
    )
    # Frameify
    df_tracking_frameified = _frameify_tracking_data(
        df_tracking_filtered, column_schema
    )
    # DAS-Berechnung Origin
    pitch_result, df_tracking_frameified = _calculate_pitch_result(
        df_tracking_frameified, column_schema
    )
    df_tracking = _filter_tracking_data_post_pitch_result(
        df_tracking_frameified, config.das_threshold
    )

    return df_tracking


def compute_preprocessing_long(tracking_data, config, column_schema):
    df = pd.DataFrame(tracking_data.copy())

    # Check ob benötigte Spalten vorhanden
    _check_columns(df, column_schema)

    # Filter Frames
    df_tracking_filtered = _filter_tracking_data_long(
        df,
        column_schema.frame_column,
        config.frame_rate,
        config.frame_filter_rate,
        config.das_threshold,
    )
    return df_tracking_filtered


def compute_preframes(df_tracking, frame_rate, frame_list, frame_col):
    pre_frame_list = frame_list - frame_rate
    df_pre_frame_tracking = df_tracking[df_tracking[frame_col].isin(pre_frame_list)]
    return df_pre_frame_tracking


# Interne Funktionen
def _filter_tracking_data(df, frame_rate, frame_filter_rate):
    df_tracking_filtered = df[df["player_possession"].notna()]
    df_tracking_filtered = df_tracking_filtered[frame_rate::frame_filter_rate]
    return df_tracking_filtered


def _frameify_tracking_data(df_tracking_filtered, column_schema):
    coordinate_cols, player_to_team, players = _extract_player_structures(
        df_tracking_filtered, column_schema
    )
    df_tracking = per_object_frameify_tracking_data(
        df_tracking_filtered,
        column_schema.frame_column,
        coordinate_cols,
        players,
        player_to_team,
        new_coordinate_cols=("player_x", "player_y", "player_vx", "player_vy"),
    )
    df_tracking = df_tracking[df_tracking["player_x"].notna()]
    return df_tracking


def _extract_player_structures(df, schema: TrackingColumnSchema):
    sep = schema.separator
    attrs = {schema.x, schema.y, schema.vx, schema.vy}

    players = set()
    player_to_team = {}
    coord_map = defaultdict(dict)

    for col in df.columns:
        parts = col.split(sep)
        if len(parts) < 2:
            continue

        attr = parts[-1]
        if attr not in attrs:
            continue

        player_id = sep.join(parts[:-1])
        players.add(player_id)

        team = "unkown"
        for part in parts:
            if part == "ball":
                team = "ball"
                break
            if part in [schema.home_team, schema.away_team]:
                team = part
                break

        player_to_team[player_id] = team

        coord_map[player_id][attr] = col

    coordinate_cols = []
    for player in sorted(players):
        cols = coord_map[player]
        coordinate_cols.append(
            [
                cols.get(schema.x),
                cols.get(schema.y),
                cols.get(schema.vx),
                cols.get(schema.vy),
            ]
        )
    return coordinate_cols, player_to_team, sorted(players)


def _calculate_pitch_result(df_tracking, schema):

    pitch_result = get_dangerous_accessible_space(
        df_tracking,
        frame_col=schema.frame_column,
        player_col="player_id",
        ball_player_id="ball",
        x_col="player_x",
        y_col="player_y",
        vx_col="player_vx",
        vy_col="player_vy",
        team_col=schema.team_column,
        period_col=schema.period_column,
        team_in_possession_col=schema.team_possession_col,
        player_in_possession_col=schema.player_in_possession_col,
        attacking_direction_col=None,
        infer_attacking_direction=True,
        respect_offside=True,
        use_progress_bar=True,
    )

    df_tracking["AS"] = pitch_result.acc_space
    df_tracking["DAS"] = pitch_result.das

    return pitch_result, df_tracking


def _filter_tracking_data_post_pitch_result(df_tracking, das_threshold):
    return df_tracking[df_tracking["DAS"] >= das_threshold]


def _check_columns(df, column_schema):
    required_columns = set(
        [
            "player_id",
            "player_x",
            "player_y",
            column_schema.frame_column,
            column_schema.team_column,
            column_schema.period_column,
            column_schema.team_possession_col,
            column_schema.player_in_possession_col,
        ]
    )
    missing = required_columns - set(df.columns)
    if missing:
        raise PDDInputError(
            f"Tracking data is missing required columns: {sorted(missing)}"
        )
    return


def _filter_tracking_data_long(
    df, frame_col, frame_rate, frame_filter_rate, das_threshold
):
    df_tracking_filtered = df[df["player_possession"].notna()]
    frame_list = df_tracking_filtered[frame_col].drop_duplicates()
    frame_list = frame_list[frame_rate::frame_filter_rate].tolist()
    df_tracking_filtered = df_tracking_filtered[
        df_tracking_filtered[frame_col].isin(frame_list)
    ]
    df_tracking_filtered = df_tracking_filtered[
        df_tracking_filtered["DAS"] >= das_threshold
    ]
    return df_tracking_filtered
