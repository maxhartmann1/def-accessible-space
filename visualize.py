import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path


def visualize():
    st.markdown(
        "<h1 style='text-align: center;'>DAS Potential Ansicht</h1>",
        unsafe_allow_html=True,
    )
    basic_path = Path("./simulation_results/reduced")

    match_list = [x.name for x in basic_path.iterdir() if x.is_dir()]
    match = st.selectbox(
        "Match wählen", match_list, format_func=lambda x: x.capitalize()
    )
    match_path = basic_path / match

    frame_step_list = [x.name for x in match_path.iterdir() if x.is_dir()]
    frame_step = st.selectbox(
        "Frame Step wählen",
        frame_step_list,
        format_func=lambda x: x.lstrip("step"),
    )
    frame_step_path = match_path / frame_step

    player_list = [x.name for x in frame_step_path.iterdir() if x.is_dir()]
    player = st.selectbox(
        "Spieler wählen",
        player_list,
        format_func=lambda x: x.capitalize(),
    )
    player_path = frame_step_path / player

    method_list = [x.name for x in player_path.iterdir() if x.is_dir()]
    files_list = []
    parameter_list = []
    for method in method_list:
        method_path = player_path / method
        files_list = files_list + list(method_path.glob("*.csv"))
        parameter_list = parameter_list + [
            method.stem for method in method_path.glob("*csv")
        ]
    parameter_list = list(set(parameter_list))
    df_random_results = pd.concat(
        (pd.read_csv(f) for f in files_list if "random" in str(f)),
        ignore_index=True,
    )
    df_random_results
    df_random_results["method"] = "random"
    df_all_players_interpolate = pd.concat(
        (pd.read_csv(f) for f in files_list if "interpolate" in str(f)),
        ignore_index=True,
    )
    df_all_players_interpolate["method"] = "interpolate"
    df_all_players_grouped = df_random_results.groupby("player_id").agg(
        {"DAS_potential": ["mean", "min", "max", "std", "median"]}
    )

    # Möge die Visualisierung beginnen
    selected_player = st.selectbox("Spieler auswählen", player_list)
    df_filtered = pd.concat(
        [
            df_random_results[df_random_results["player_id"] == selected_player],
            df_all_players_interpolate[
                df_all_players_interpolate["player_id"] == selected_player
            ],
        ]
    )

    chart = (
        alt.Chart(df_filtered)
        .mark_line()
        .encode(
            x=alt.X("frame:Q", title="Frame"),
            y=alt.Y("DAS_potential:Q", title="DAS Potential"),
            color=alt.Color("method:N", title="Methode"),
        )
        .properties(
            width=800, height=400, title="DAS Potential pro Methode über Zeitverlauf"
        )
    )
    chart_total = (
        alt.Chart(df_filtered)
        .mark_line()
        .encode(
            x=alt.X("frame:Q", title="Frame"),
            y=alt.Y("DAS_new:Q", title="DAS_new"),
            color=alt.Color("method:N", title="Methode"),
        )
        .properties(
            width=800, height=400, title="DAS Potential pro Methode über Zeitverlauf"
        )
    )

    st.altair_chart(chart, use_container_width=True)
    st.altair_chart(chart_total, use_container_width=True)


if __name__ == "__main__":
    visualize()
