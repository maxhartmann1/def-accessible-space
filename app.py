from tkinter import EXCEPTION
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import accessible_space
import optimize_def, resources, prep_das_def
from databallpy.visualize import plot_soccer_pitch, plot_tracking_data


def load_match():
    start_time = time.time()
    match = prep_das_def.load_match_data("metrica")
    prep_match = prep_das_def.prep_match_data(match)
    st.write(f"Match-Daten geladen in {time.time() - start_time:.2f} Sekunden")
    return prep_match


def frameify_tracking(match, df_tracking_reduced):
    coordinate_cols = []
    player_to_team = {}
    players = match.get_column_ids()
    players.append("ball")
    for player in players:
        coordinate_cols.append([f"{player}_x", f"{player}_y"])
        player_to_team[str(player)] = player.split("_")[0]

    df_tracking = prep_das_def.prep_tracking_frame_by_frame(
        df_tracking_reduced, coordinate_cols, players, player_to_team
    )
    return df_tracking


def get_frames_per_second(match):
    df = match.tracking_data.copy()
    df["timestamp_sec"] = df["datetime"].dt.floor("s")
    grouped_df = df.groupby("timestamp_sec").size().reset_index(name="Anzahl")
    return grouped_df["Anzahl"].iloc[0]


def filter_tracking_data_pre_frameify(df, n, x):
    return df.iloc[n::x]


def filter_tracking_data_post_frameify(df, n, x):
    return df[df["frame"].isin(range(n, df["frame"].max() + 1, x))]


def plot_frame(match, frame, pitch_result, das_frame_index, df_pre_frame):
    idx = match.tracking_data.index[match.tracking_data["frame"] == frame].tolist()[0]
    idx_das = pitch_result.frame_index[das_frame_index]
    fig, ax = plt.subplots(figsize=(10, 6))
    fig, ax = plot_soccer_pitch(
        fig=fig, ax=ax, field_dimen=match.pitch_dimensions, pitch_color="white"
    )
    fig, ax = plot_tracking_data(
        match,
        idx,
        fig=fig,
        ax=ax,
        team_colors=["blue", "red"],
        variable_of_interest=round(float(pitch_result.das.iloc[idx_das]), 2),
        add_player_possession=True,
    )
    try:
        accessible_space.plot_expected_completion_surface(
            pitch_result.dangerous_result, frame_index=idx_das
        )
    except:
        st.warning(f"Fehler beim Plotten von DAS")

    for _, row in df_pre_frame.iterrows():
        if row["team_id"] != row["ball_possession"] and row["team_id"] != "ball":
            player_x, player_y = row["player_x"], row["player_y"]
            circle = plt.Circle(
                (player_x, player_y),
                color=resources.get_team_colors()[row["team_id"]],
                alpha=0.3,
                fill=True,
            )
            ax.add_patch(circle)
    st.pyplot(fig)


def show_tracking_data(match, frame):
    if "show_tracking_data" in st.session_state and st.session_state.show_tracking_data:
        st.write("### Tracking-Daten für den aktuellen Frame")
        st.dataframe(match.tracking_data.iloc[frame])


def show_event_data(df_passes):
    if "show_event_data" in st.session_state and st.session_state.show_event_data:
        st.write("### Event Daten für Match")
        st.dataframe(df_passes)


def main():
    st.title("Interaktive Fussball-Analyse")
    match = load_match()
    fps = get_frames_per_second(match)
    df_passes, kick_off_frame = prep_das_def.event_data_prep(match)
    df_tracking_reduced = filter_tracking_data_pre_frameify(
        match.tracking_data, kick_off_frame, fps
    )
    df_frameified = frameify_tracking(match, df_tracking_reduced)
    df_tracking = filter_tracking_data_post_frameify(
        df_frameified, kick_off_frame + 1, fps * 60
    )
    df_tracking_ball = df_tracking[df_tracking["player_id"] == "ball"].reset_index(
        drop=True
    )

    start_time = time.time()
    pitch_result = prep_das_def.get_dangerous_accessible_space(df_tracking)
    df_tracking["AS"] = pitch_result.acc_space
    df_tracking["DAS"] = pitch_result.das
    st.write(
        f"Dangerous Accessible Space Calculation in {time.time() - start_time:.2f} Sekunden"
    )

    # Slider und Frame Auswahl
    col1, col2 = st.columns([4, 1])
    with col1:
        index = st.slider(
            "Wähle Frame-Index", 0, len(df_tracking_ball) - 1, 0, key="index_slider"
        )
    with col2:
        index_input = st.number_input(
            "Direkte Index-Eingabe",
            0,
            len(df_tracking_ball) - 1,
            value=index,
            step=1,
            key="frame_input",
        )

    frame = df_tracking_ball["frame"].iloc[index]
    das_frame_index = index * 60

    if index_input != index:
        index = index_input
    if "last_index" not in st.session_state or st.session_state.last_index != index:
        frame = df_tracking_ball["frame"].iloc[index]
        das_frame_index = index * 60
        st.session_state.show_tracking_data = False
        st.session_state.last_index = index

    st.write(f"Aktuell gewählter Frame: {frame}")

    # Framefiy - Dangerous Accessible acc_space
    df_pre_frame = optimize_def.get_recheable_area(
        df_frameified, das_frame_index, frame, fps, kick_off_frame + 1
    )

    # Match Plotting
    plot_frame(match, frame, pitch_result, das_frame_index, df_pre_frame)

    optimize_def.get_highest_dangerous_values(
        pitch_result, df_pre_frame, df_tracking, frame, das_frame_index, match
    )

    # optimize_def.get_highest_dangerous_values_manually(
    #     pitch_result, df_pre_frame, df_tracking, frame, das_frame_index, match
    # )


if __name__ == "__main__":
    main()
