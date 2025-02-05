import accessible_space
import prep_das_def
import accessible_space.apps.readme
from accessible_space.tests.resources import df_passes, df_tracking
import streamlit as st
from databallpy import get_match, get_open_match
from databallpy.visualize import plot_soccer_pitch, plot_events, plot_tracking_data
from databallpy.features.filters import filter_tracking_data
from databallpy.features.differentiate import add_velocity, add_acceleration
from databallpy.features.team_possession import add_team_possession
from databallpy.features import get_individual_player_possession
import numpy as np
import pandas as pd
import datetime as dt
import time

CORDINATE_COLS = []
TRACKING_PLAYERS = []
TRACKING_PLAYER_TO_TEAM = {}

TEAM_MAPPING = {"FIFATMA": "home", "FIFATMB": "away"}


@st.cache_data
def _load_match_data():
    return get_open_match()


@st.cache_data
def _load_match_data_spec(metrica):
    match = get_open_match(metrica)
    match.synchronise_tracking_and_event_data()
    add_team_possession(
        match.tracking_data, match.event_data, match.home_team_id, inplace=True
    )
    get_individual_player_possession(match.tracking_data, inplace=True)

    return match


@st.cache_data
def _per_object_frameify_tracking_data(
    df_tracking, frame_col, coordinate_cols, players, player_to_team
):
    return accessible_space.per_object_frameify_tracking_data(
        df_tracking,
        frame_col,
        coordinate_cols,
        players,
        player_to_team,
        new_coordinate_cols=("player_x", "player_y"),
    )


@st.cache_data
def _match_id_cols(df_tracking_match, passes):
    df_tracking_match["event_player_id"] = (
        "Player " + df_tracking_match["player_id"].str.split("_").str[1]
    )
    df_tracking_match["event_player_id"] = df_tracking_match["event_player_id"].where(
        df_tracking_match["player_id"] != "ball", "ball"
    )
    passes.loc[:, "team_id"] = passes["team_id"].map(TEAM_MAPPING)
    return df_tracking_match, passes


@st.cache_data
def _get_expected_pass_completion(passes, df_tracking_match):
    return accessible_space.get_expected_pass_completion(
        passes,
        df_tracking_match,
        event_frame_col="tracking_frame",
        event_player_col="player_name",
        event_team_col="team_id",
        event_start_x_col="start_x",
        event_start_y_col="start_y",
        event_end_x_col="end_x",
        event_end_y_col="end_y",
        tracking_frame_col="frame",
        tracking_player_col="event_player_id",
        tracking_team_col="team_id",
        tracking_x_col="player_x",
        tracking_y_col="player_y",
        tracking_vx_col="player_vx",
        tracking_vy_col="player_vy",
        ball_tracking_player_id="ball",
        additional_fields_to_return=["defense_cum_prob"],
    )


def _fill_coordinates(match):
    cols = match.get_column_ids()
    cols.append("ball")
    TRACKING_PLAYERS.extend(cols)
    for player in cols:
        CORDINATE_COLS.append([f"{player}_x", f"{player}_y"])
        TRACKING_PLAYER_TO_TEAM[str(player)] = player.split("_")[0]
    return


def format_matchtime(row):
    minute = int(row["minutes"])
    second = int(row["seconds"])
    period = row["period_id"]

    if period == 1 and minute > 45:
        extra_min = minute - 45
        return f"45:00+{extra_min}:{str(second).zfill(2)}"
    elif period == 2 and minute > 90:
        extra_min = minute - 90
        return f"90:00+{extra_min}:{str(second).zfill(2)}"
    else:
        return f"{str(minute).zfill(2)}:{str(second).zfill(2)}"


def run():
    # Example 1
    st.title("Expected Pass Completion Analysis")
    st.dataframe(df_passes)
    st.dataframe(df_tracking)
    # accessible_space.apps.readme.main()
    pass_result = accessible_space.get_expected_pass_completion(
        df_passes,
        df_tracking,
        event_frame_col="frame_id",
        event_player_col="player_id",
        event_team_col="team_id",
        event_start_x_col="x",
        event_start_y_col="y",
        event_end_x_col="x_target",
        event_end_y_col="y_target",
        tracking_frame_col="frame_id",
        tracking_player_col="player_id",
        tracking_team_col="team_id",
        tracking_team_in_possession_col="team_in_possession",
        tracking_x_col="x",
        tracking_y_col="y",
        tracking_vx_col="vx",
        tracking_vy_col="vy",
        ball_tracking_player_id="ball",
    )
    df_passes["xC"] = pass_result.xc  # Expected pass completion rate

    st.subheader("Expected Pass Completion Rates")
    st.dataframe(df_passes[["event_string", "xC"]])

    # Example 2
    das_gained_result = accessible_space.get_das_gained(
        df_passes,
        df_tracking,
        event_frame_col="frame_id",
        event_success_col="pass_outcome",
        event_target_frame_col="target_frame_id",
        tracking_frame_col="frame_id",
        tracking_period_col="period_id",
        tracking_player_col="player_id",
        tracking_team_col="team_id",
        tracking_x_col="x",
        tracking_y_col="y",
        tracking_vx_col="vx",
        tracking_vy_col="vy",
        tracking_team_in_possession_col="team_in_possession",
        x_pitch_min=-52.5,
        x_pitch_max=52.5,
        y_pitch_min=-34,
        y_pitch_max=34,
    )
    df_passes["DAS_Gained"] = das_gained_result.das_gained
    df_passes["AS_Gained"] = das_gained_result.as_gained
    st.dataframe(df_passes[["event_string", "DAS_Gained", "AS_Gained"]])


def run_match():
    match = _load_match_data()
    # Zugriff auf die Ereignisdaten
    event_data = match.event_data

    # Filtern der Passereignisse
    passes = event_data[event_data["databallpy_event"] == "pass"]

    # Anzeigen der ersten Zeilen der Passdaten
    print(passes.head())

    # Title
    st.title(
        f"{match.home_team_name} - {match.away_team_name}  {match.home_score}:{match.away_score}"
    )

    # Preprocess Tracking Data
    unfiltered_data = match.tracking_data
    filtered_data = filter_tracking_data(
        match.tracking_data,
        column_ids="ball",
        filter_type="savitzky_golay",
        window_length=25,
        polyorder=2,
        inplace=False,
    )

    df_tracking_match = accessible_space.per_object_frameify_tracking_data(
        filtered_data,
        "frame",
        SPORTEC_TRACKING_PLAYER_COORDINATES,
        SPORTEC_TRACKING_PLAYERS,
        SPORTEC_TRACKING_TEAM,
        new_coordinate_cols=("player_x", "player_y"),
    )
    df_tracking_match = add_velocity(
        df_tracking_match, frame_rate=match.frame_rate, column_ids=["player", "ball"]
    )
    match.synchronise_tracking_and_event_data()

    # Preprocess Event Data

    df_passes_match = match.event_data[match.event_data["databallpy_event"] == "pass"]

    # df_passes_match["matchtime_ed"] = df_passes_match.apply(format_matchtime, axis=1)

    # df_passes_match["frame_id"] = df_passes_match["matchtime_ed"].map(
    #     match.tracking_data.drop_duplicates(subset="matchtime_td").set_index(
    #         "matchtime_td"
    #     )["frame"]
    # )

    st.header("Tracking Frame")
    st.dataframe(df_tracking_match[df_tracking_match["frame"] == 10_051])
    st.header("Passes Frame")
    st.dataframe(df_passes_match)

    pass_result = accessible_space.get_expected_pass_completion(
        df_passes_match,
        df_tracking_match,
        event_frame_col="frame_id",
        event_player_col="player_id",
        event_team_col="team_id",
        event_start_x_col="start_x",
        event_start_y_col="start_y",
        tracking_frame_col="frame",
        tracking_player_col="player_id",
        tracking_team_col="team_id",
        tracking_x_col="player_x",
        tracking_y_col="player_y",
        tracking_vx_col="player_vx",
        tracking_vy_col="player_vy",
    )


def run_metrica_match():
    start_time = time.time()
    match = _load_match_data_spec("metrica")
    # Title
    st.title(match.name)
    st.write(f"Loading match data: {time.time() - start_time:.2f} seconds")

    # Filtern Pass Events
    start_time = time.time()
    event_data = match.event_data
    passes = event_data[event_data["databallpy_event"] == "pass"]
    goals = event_data.query('databallpy_event == "shot" and outcome == 1')
    st.write(f"Filtering events: {time.time() - start_time:.2f} seconds")

    # Preprocessing Tracking Data

    # Frameify Tracking Data
    start_time = time.time()
    _fill_coordinates(match)
    df_tracking_match = _per_object_frameify_tracking_data(
        match.tracking_data,
        "frame",
        CORDINATE_COLS,
        TRACKING_PLAYERS,
        TRACKING_PLAYER_TO_TEAM,
    )
    st.write(f"Frameify: {time.time() - start_time:.2f} seconds")

    # Add velocity
    start_time = time.time()
    df_tracking_match = add_velocity(
        df_tracking_match,
        column_ids=["player"],
        frame_rate=match.frame_rate,
        max_velocity=13.0,
    )
    st.write(f"Add Velocity: {time.time() - start_time:.2f} seconds")

    # Match id columns
    start_time = time.time()
    df_tracking_match, passes = _match_id_cols(df_tracking_match, passes)
    st.write(f"Match Datasets: {time.time() - start_time:.2f} seconds")

    # Expected Pass Completion

    start_time = time.time()
    st.header("Expected Pass Completion")
    pass_result = _get_expected_pass_completion(passes, df_tracking_match)
    st.write(f"Pass completion calculation: {time.time() - start_time:.2f} seconds")

    passes["xC"] = pass_result.xc

    # Dangerous Accessbile Space

    # DAS gained for passes
    # start_time = time.time()
    # das_gained_result = accessible_space.get_das_gained(
    #     passes,
    #     df_tracking_match,
    #     event_frame_col="tracking_frame",
    #     event_target_frame_col="tracking_frame",
    #     event_success_col="outcome",
    #     event_start_x_col="start_x",
    #     event_start_y_col="start_y",
    #     event_target_x_col="end_x",
    #     event_target_y_col="end_y",
    #     event_team_col="team_id",
    #     event_receiver_team_col="team_id",
    #     tracking_frame_col="frame",
    #     tracking_player_col="player_id",
    #     tracking_team_col="team_id",
    #     tracking_x_col="player_x",
    #     tracking_y_col="player_y",
    #     tracking_vx_col="player_vx",
    #     tracking_vy_col="player_vy",
    #     tracking_team_in_possession_col="ball_possession",
    #     ball_tracking_player_id="ball",
    #     tracking_period_col="period_id",
    # )
    # passes["DAS_Gained"] = das_gained_result.das_gained
    # passes["AS_Gained"] = das_gained_result.as_gained
    # st.write(f"DAS gained for passes: {time.time() - start_time:.2f} seconds")

    # DAS for tracking
    start_time = time.time()
    das_df_tracking_match = df_tracking_match[df_tracking_match["frame"] == 3040]
    pitch_result = accessible_space.get_dangerous_accessible_space(
        das_df_tracking_match,
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
    )
    das_df_tracking_match["AS"] = pitch_result.acc_space
    das_df_tracking_match["DAS"] = pitch_result.das
    st.write(f"DAS for tracking: {time.time() - start_time:.2f} seconds")

    # Print Section
    st.title("Print Section")
    st.dataframe(passes)
    st.dataframe(das_df_tracking_match)

    st.write("#### Example 4.2: Plot accessible space and dangerous accessible space")
    idx = das_df_tracking_match["frame"].iloc[0] - 1

    fig, ax = plot_soccer_pitch(field_dimen=match.pitch_dimensions)
    fig, ax = plot_tracking_data(
        match,
        idx,
        fig=fig,
        ax=ax,
        team_colors=["blue", "red"],
        events=["pass"],
        add_player_possession=True,
    )
    accessible_space.plot_expected_completion_surface(
        pitch_result.dangerous_result, frame_index=0
    )
    st.pyplot(fig)


if __name__ == "__main__":
    print(f"Restart: {dt.datetime.time((dt.datetime.now()))}")
    # run()
    # run_match()
    run_metrica_match()
    print(f"Finished: {dt.datetime.time((dt.datetime.now()))}")
