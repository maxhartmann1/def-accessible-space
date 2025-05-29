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
    basic_path = Path("./simulation_results")

    provider_list = [x.name for x in basic_path.iterdir() if x.is_dir()]
    provider = st.selectbox(
        "Provider wählen", provider_list, format_func=lambda x: x.capitalize()
    )
    provider_path = basic_path / provider

    match_list = [x.name for x in provider_path.iterdir() if x.is_dir()]
    match = st.selectbox(
        "Match wählen", match_list, format_func=lambda x: x.capitalize()
    )
    match_path = provider_path / match

    frame_frequence_list = [x.name for x in match_path.iterdir() if x.is_dir()]
    frame_frequence = st.selectbox(
        "Frame Frequenz wählen",
        frame_frequence_list,
        format_func=lambda x: x.capitalize(),
    )
    frame_frequence_path = match_path / frame_frequence

    method_list = [x.name for x in frame_frequence_path.iterdir() if x.is_dir()]
    player_files_list = []
    player_list = []
    for method in method_list:
        method_path = frame_frequence_path / method
        player_files_list = player_files_list + list(method_path.glob("*.csv"))
        player_list = player_list + [player.stem for player in method_path.glob("*csv")]
    player_list = list(set(player_list))
    df_all_players_random = pd.concat(
        (pd.read_csv(f) for f in player_files_list if "random" in str(f)),
        ignore_index=True,
    )
    df_all_players_random["method"] = "random"
    df_all_players_interpolate = pd.concat(
        (pd.read_csv(f) for f in player_files_list if "interpolate" in str(f)),
        ignore_index=True,
    )
    df_all_players_interpolate["method"] = "interpolate"
    df_all_players_grouped = df_all_players_random.groupby("player_id").agg(
        {"DAS_potential": ["mean", "min", "max", "std", "median"]}
    )

    # Möge die Visualisierung beginnen
    selected_player = st.selectbox("Spieler auswählen", player_list)
    df_filtered = pd.concat(
        [
            df_all_players_random[
                df_all_players_random["player_id"] == selected_player
            ],
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
