import pandas as pd
from .config import PDDConfig, TrackingColumnSchema, PDDInputError
from .preprocessing import (
    compute_preprocessing,
    compute_preframes,
    compute_preprocessing_long,
)
from .optimization import compute_optimization, reduce_df_optimization
from .aggregation import compute_player_level_pdd
import sys  # Debug to delete


def compute_pdd(
    tracking_data: pd.DataFrame,
    config: PDDConfig = PDDConfig(),
    column_schema: TrackingColumnSchema = TrackingColumnSchema(),
    player_id: str | list[str] = None,
    wide_format: bool = True,
):
    """
    Compute Player PDD.

    Parameters
    ----------
    :param tracking_data: Tracking Data in Wide Format,
    :type tracking_data: pd.DataFrame {x/y, vx/vy, team_possession, player_possession}
    :param config: Parameterset der Optimierung
    :type config: PDDConfig
    :param defending_team: Spieler zu optimieren
    :type defending_team: str

    Returns
    --------
    We will see
    """
    # If Wide Format -> Preprocessing, else directly to DAS
    if wide_format:
        post_preprocessing_tracking_data = compute_preprocessing(
            tracking_data, config, column_schema
        )

    else:
        try:
            post_preprocessing_tracking_data = compute_preprocessing_long(
                tracking_data, config, column_schema
            )
        except PDDInputError as e:
            raise RuntimeError(f"PDD computation failed: {e}") from None

    frame_list = post_preprocessing_tracking_data[column_schema.frame_column].unique()

    # Pre Frames auslesen
    pre_frame_tracking_data = compute_preframes(
        tracking_data, config.frame_rate, frame_list, column_schema.frame_column
    )
    if player_id is None:
        player_id = post_preprocessing_tracking_data["player_id"].unique().tolist()
    elif isinstance(player_id, str):
        player_id = [player_id]

    # Optimierung
    pitch_result_optimized, frame_list_optimized, df_tracking_optimized = (
        compute_optimization(
            post_preprocessing_tracking_data,
            pre_frame_tracking_data,
            player_id,
            frame_list,
            config,
            column_schema,
            wide_format,
        )
    )
    pdd_frame_level = reduce_df_optimization(df_tracking_optimized, column_schema)

    # Aggregierung auf Spieler-Level
    pdd_player_level = compute_player_level_pdd(pdd_frame_level)

    return pdd_frame_level, pdd_player_level
