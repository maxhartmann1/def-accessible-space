import accessible_space
from accessible_space.interface import ReturnValueDAS
import streamlit as st
from databallpy import get_open_game


@st.cache_resource
def load_game_data(source, game_id):
    return get_open_game(provider=source, game_id=game_id)


@st.cache_resource
def prep_game_data(game_name, _game, max_velocity=25):
    game_copy = _game.copy()
    col_ids = _game.get_column_ids() + ["ball"]

    game_copy.tracking_data.add_velocity(column_ids=col_ids, max_velocity=max_velocity)

    game_copy.synchronise_tracking_and_event_data()

    game_copy.tracking_data.add_team_possession(
        game_copy.event_data,
        game_copy.home_team_id,
    )
    game_copy.tracking_data.add_individual_player_possession()
    return game_copy


@st.cache_data
def frameify_tracking_data(
    df_tracking,
    coordinate_cols,
    players,
    player_to_team,
    frame_col="frame",
):
    df_tracking_data = accessible_space.per_object_frameify_tracking_data(
        df_tracking,
        frame_col,
        coordinate_cols,
        players,
        player_to_team,
        new_coordinate_cols=("player_x", "player_y", "player_vx", "player_vy"),
    )
    return df_tracking_data


@st.cache_resource
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
    team_in_possession_col="team_possession",
    attacking_direction_col=None,
    infer_attacking_direction=True,
    additional_fields_to_return=None,
    use_progress_bar=True,
):
    st.write(f"Formate des zu optimierenden DF: {df_tracking.shape}")
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
        additional_fields_to_return=additional_fields_to_return,
        use_progress_bar=use_progress_bar,
    )
    return pitch_result


@st.cache_data
def get_pre_frames(df, fps, frame=None, frame_list=[]):
    if frame:
        df_before = df[df["frame"] == frame - fps]
        return df_before
    pre_frame_list = frame_list - fps
    df_before = df[df["frame"].isin(pre_frame_list)]
