from calendar import c
import streamlit as st

from databallpy import get_open_match
from databallpy.features.team_possession import add_team_possession
from databallpy.features import get_individual_player_possession
from databallpy.features.differentiate import add_velocity

import accessible_space


@st.cache_resource
def load_match_data(source):
    if source is None:
        return get_open_match()
    else:
        return get_open_match(source)


@st.cache_data
def prep_match_data(match):
    match_copy = match.copy()
    match_copy.synchronise_tracking_and_event_data()
    add_team_possession(
        match_copy.tracking_data,
        match_copy.event_data,
        match_copy.home_team_id,
        inplace=True,
    )
    get_individual_player_possession(match_copy.tracking_data, inplace=True)
    return match_copy


@st.cache_data
def prep_tracking_frame(
    match, frame_col, coordinate_cols, players, player_to_team, column_ids
):
    df_tracking = accessible_space.per_object_frameify_tracking_data(
        match.tracking_data,
        frame_col,
        coordinate_cols,
        players,
        player_to_team,
        new_coordinate_cols=("player_x", "player_y"),
    )
    df_tracking = add_velocity(
        df_tracking,
        column_ids=column_ids,
        frame_rate=match.frame_rate,
        max_velocity=15.0,
    )
    return df_tracking


@st.cache_data
def match_id_cols(df_tracking, passes):
    pass


@st.cache_data
def get_expected_pass_completion(passes, df_tracking):
    pass


@st.cache_data
def get_dangerous_accessible_space(
    df_tracking,
    frame_col="frame",
    player_col="player_id",
    ball_player_id="ball",
    team_col="team_id",
    x_col="player_x",
    y_col="player_y",
    vx_col="player_vx",
    vy_col="player_vy",
    period_col="period_id",
    team_in_possession_col="ball_possession",
    attacking_direction_col=None,
    infer_attacking_direction=True,
):
    pitch_result = accessible_space.get_dangerous_accessible_space(
        df_tracking,
        frame_col=frame_col,
        player_col=player_col,
        ball_player_id=ball_player_id,
        team_col=team_col,
        x_col=x_col,
        y_col=y_col,
        vx_col=vx_col,
        vy_col=vy_col,
        period_col=period_col,
        team_in_possession_col=team_in_possession_col,
        attacking_direction_col=attacking_direction_col,
        infer_attacking_direction=infer_attacking_direction,
    )
    return pitch_result
