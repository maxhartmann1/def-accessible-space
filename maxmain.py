import accessible_space
import greedy_opt_das
import prep_das_def
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

METRICA_MAPPING = {"FIFATMA": "home", "FIFATMB": "away"}


def _get_frameify_args(match):
    coordinate_cols = []
    player_to_team = {}
    players = match.get_column_ids()
    players.append("ball")
    for player in players:
        coordinate_cols.append([f"{player}_x", f"{player}_y"])
        player_to_team[str(player)] = player.split("_")[0]
    return (
        coordinate_cols,
        players,
        player_to_team,
    )


def run_match(source):
    # Load Match Data
    st.write("Cached Run")
    start_time = time.time()
    match = prep_das_def.load_match_data("metrica")
    st.title(match.name)
    st.write(f"Loading match data: {time.time() - start_time:.2f} seconds")

    # Prep Match Data
    start_time = time.time()
    prep_match = prep_das_def.prep_match_data(match)
    st.write(f"Prep Match Data: {time.time() - start_time:.2f} seconds")

    # Prep Framify Args
    start_time = time.time()
    frameify_coordinate_cols, frameify_players, frameify_player_to_team = (
        _get_frameify_args(prep_match)
    )
    st.write(f"Get Frameify Args: {time.time() - start_time:.2f} seconds")

    # Prep Tracking Frame
    start_time = time.time()
    df_tracking = prep_das_def.prep_tracking_frame(
        prep_match,
        "frame",
        frameify_coordinate_cols,
        frameify_players,
        frameify_player_to_team,
        "player",
    )
    st.write(f"Prep Tracking Frame: {time.time() - start_time:.2f} seconds")

    # DAS for tracking
    start_time = time.time()
    df_tracking = df_tracking[df_tracking["databallpy_event"] == "pass"]

    pitch_result = prep_das_def.get_dangerous_accessible_space(df_tracking)
    df_tracking["AS"] = pitch_result.acc_space
    df_tracking["DAS"] = pitch_result.das

    st.write(f"DAS for tracking: {time.time() - start_time:.2f} seconds")
    df_tracking = df_tracking[df_tracking["DAS"] >= 20]
    idx_dbpy = df_tracking["frame"].iloc[0] - 1
    idx_das = pitch_result.frame_index.loc[idx_dbpy]

    fig, ax = plot_soccer_pitch(field_dimen=match.pitch_dimensions, pitch_color="white")
    fig, ax = plot_tracking_data(
        prep_match,
        idx_dbpy,
        fig=fig,
        ax=ax,
        team_colors=["blue", "red"],
        events=["pass"],
        add_player_possession=True,
    )
    accessible_space.plot_expected_completion_surface(
        pitch_result.dangerous_result, frame_index=idx_das
    )
    st.pyplot(fig)

    # Optimieung DEF Positioning
    das_map = np.array(pitch_result.dangerous_result.attack_poss_density[idx_das])
    df_tracking_optframe = df_tracking[df_tracking["frame"] == idx_dbpy + 1]
    df_defenders = df_tracking_optframe.loc[
        df_tracking_optframe["team_id"].ne(df_tracking_optframe["ball_possession"])
    ]
    df_defenders = df_defenders[df_defenders["team_id"] != "ball"]
    df_defenders = df_defenders[df_defenders["player_x"].notna()]
    defenders = greedy_opt_das.extract_defender_positions(df_defenders)
    optimized_defenders = pd.DataFrame(
        greedy_opt_das.optimize_defensive_positions(das_map, defenders),
        columns=["player_id", "player_x", "player_y"],
    )
    optimized_defenders["player_id"] = optimized_defenders["player_id"].astype(
        df_tracking_optframe["player_id"].dtype
    )

    df_tracking_optframe.loc[
        df_tracking_optframe["player_id"].isin(optimized_defenders["player_id"]),
        ["player_x", "player_y"],
    ] = (
        df_tracking_optframe.loc[
            df_tracking_optframe["player_id"].isin(optimized_defenders["player_id"])
        ]
        .merge(optimized_defenders, on="player_id", suffixes=("", "_new"))[
            ["player_x_new", "player_y_new"]
        ]
        .values
    )
    for player_id, x, y in zip(
        optimized_defenders["player_id"],
        optimized_defenders["player_x"],
        optimized_defenders["player_y"],
    ):
        prep_match.tracking_data[f"{player_id}_x"] = x
        prep_match.tracking_data[f"{player_id}_y"] = y
    df_tracking_optframe["direction"] = -1
    opt_pitch_result = prep_das_def.get_dangerous_accessible_space(
        df_tracking_optframe,
        attacking_direction_col="direction",
        infer_attacking_direction=False,
    )
    df_tracking_optframe["AS"] = opt_pitch_result.acc_space
    df_tracking_optframe["DAS"] = opt_pitch_result.das
    st.dataframe(df_tracking[df_tracking["frame"] == idx_dbpy + 1])
    st.dataframe(
        prep_match.tracking_data[prep_match.tracking_data["frame"] == idx_dbpy + 1]
    )
    st.dataframe(df_tracking_optframe)

    fig, ax = plot_soccer_pitch(field_dimen=match.pitch_dimensions, pitch_color="white")
    fig, ax = plot_tracking_data(
        prep_match,
        idx_dbpy,
        fig=fig,
        ax=ax,
        team_colors=["blue", "red"],
        events=["pass"],
        add_player_possession=True,
    )
    st.write(type(opt_pitch_result.dangerous_result))
    st.write(opt_pitch_result.simulation_result)
    accessible_space.plot_expected_completion_surface(
        opt_pitch_result.dangerous_result, frame_index=0, color="red"
    )
    st.pyplot(fig)

    fig, ax = plot_soccer_pitch(field_dimen=match.pitch_dimensions, pitch_color="white")
    fig, ax = plot_tracking_data(
        prep_match,
        idx_dbpy,
        fig=fig,
        ax=ax,
        team_colors=["blue", "red"],
        events=["pass"],
        add_player_possession=True,
    )
    accessible_space.plot_expected_completion_surface(
        opt_pitch_result.simulation_result, frame_index=0
    )
    st.pyplot(fig)


if __name__ == "__main__":
    print(f"Restart: {dt.datetime.time((dt.datetime.now()))}")
    run_match("metrica")
    print(f"Finished: {dt.datetime.time((dt.datetime.now()))}")
