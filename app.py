from turtle import home
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import defensive_das
from databallpy.visualize import plot_soccer_pitch, plot_tracking_data
from databallpy.utils.constants import OPEN_MATCH_IDS_DFL


def load_match(provider, match_id):
    match = defensive_das.load_match_data(provider, match_id)
    prep_match = defensive_das.prep_match_data(match)
    return prep_match


def get_open_source_matches():
    matches = OPEN_MATCH_IDS_DFL
    matches["metrica"] = "Metrica Anonymisiertes Spiel"
    return matches


def frameify_tracking_data(df_tracking, match):
    coordinate_cols = []
    player_to_team = {}
    players = match.get_column_ids()
    players.append("ball")
    for player in players:
        coordinate_cols.append([f"{player}_x", f"{player}_y"])
        player_to_team[str(player)] = player.split("_")[0]

    df_tracking = defensive_das.frameify_tracking_data(
        df_tracking, match.frame_rate, coordinate_cols, players, player_to_team
    )
    return df_tracking


def filter_tracking_df_for_das(
    df, minute_frame_rate, frame_rate, method="frame_rate", dead_ball=True
):
    if dead_ball:
        df = df[df["ball_status"] == "alive"]
    if method == "frame_rate":
        filter_step = int((frame_rate * 60) / minute_frame_rate)
        return df[frame_rate::filter_step]
    else:
        return df


def calculate_dangerous_accessible_space(df_frameified):
    return defensive_das.get_dangerous_accessible_space(df_frameified)


def one_frame_optimization(
    frame_amount, df_tracking_ball, match, pitch_result, df_frameified
):
    # Frame Auswahl
    input_index = st.number_input(
        "Index für Frame zu printen",
        0,
        frame_amount - 1,
        value=st.session_state.input_index,
        step=1,
        key="frame_input",
    )

    frame = df_tracking_ball["frame"].iloc[input_index]
    if (
        "last_index" not in st.session_state
        or st.session_state.last_index != input_index
    ):
        frame = df_tracking_ball["frame"].iloc[input_index]
        st.session_state.last_index = input_index

    st.write(f"Aktuell gewählter Frame: {frame}")

    df_pre_frame = defensive_das.get_pre_frames(
        match.tracking_data, match.frame_rate, frame
    )

    df_pre_frame = frameify_tracking_data(df_pre_frame, match)
    match_optimized, df_optimized, pitch_result_optimized = (
        defensive_das.calculate_optimal_das_one_frame(
            pitch_result, df_frameified, df_pre_frame, match, input_index, frame
        )
    )
    return (
        frame,
        input_index,
        df_pre_frame,
        match_optimized,
        df_optimized,
        pitch_result_optimized,
    )


def all_frame_optimization(match, frame_list, pitch_result, df_frameified):
    df_pre_frames = defensive_das.get_pre_frames(
        match.tracking_data, match.frame_rate, frame_list=frame_list
    )
    df_pre_frames = frameify_tracking_data(df_pre_frames, match)
    df_das_changes, df_tracking_opt = defensive_das.calculate_optimal_das_all_frames(
        pitch_result, df_frameified, df_pre_frames, match, frame_list
    )
    return df_das_changes, df_tracking_opt


def single_player_optimize(
    match, frame_list, df_frameified, pitch_result, player_column_id
):
    df_pre_frames = defensive_das.get_pre_frames(
        match.tracking_data, match.frame_rate, frame_list=frame_list
    )
    # df_pre_frames = frameify_tracking_data(df_pre_frames, match)
    df_frameified_simulations = defensive_das.single_player_das_optimization(
        df_frameified, pitch_result, df_pre_frames, match, frame_list, player_column_id
    )
    return df_frameified_simulations


def reduce_df_simulations(df_frameified_simulations):
    df_frameified_simulations = df_frameified_simulations[
        df_frameified_simulations["opt_player"]
        == df_frameified_simulations["player_id"]
    ]


def click_button(button):
    if button not in st.session_state:
        st.session_state[button] = True
    else:
        st.session_state[button] = not st.session_state[button]


def reset_possesion():
    st.session_state.possesion_value_home = 0


# -------------------------------------------------------------


def main():
    # if "minute_frame_rate" not in st.session_state:
    #     st.session_state.minute_frame_rate = 1
    # if "input_index" not in st.session_state:
    #     st.session_state.input_index = 0
    # if "one_frame_clicked" not in st.session_state:
    #     st.session_state.one_frame_clicked = False
    # if "all_frame_clicked" not in st.session_state:
    #     st.session_state.all_frame_clicked = False

    st.write(st.session_state)
    # Überschrift Streamlit Page
    st.markdown(
        "<h1 style='text-align: center;'>Defensive Metric based on DAS</h1>",
        unsafe_allow_html=True,
    )
    # Verfügbare Spiele laden
    col1, col2 = st.columns(2)
    matches = get_open_source_matches()
    with col1:
        st.write("### Match auswählen")
        match_id = st.selectbox(
            "Open Source Match auswählen",
            matches,
            index=7,
            format_func=lambda x: matches[x],
            on_change=reset_possesion,
        )
    provider = "metrica" if match_id == "metrica" else "dfl"

    # Ausgewähltes Spiel laden
    start_time = time.time()
    match = load_match(provider, match_id)
    with col1:
        st.write(
            f"{match.home_team_name} {match.home_score} - {match.away_score} {match.away_team_name} provided by {match.tracking_data_provider} mit Frame Rate von {match.frame_rate} Frames / Sekunde."
        )

    # Anzahl zu verarbeitender Frames wählen
    with col2:
        st.write("### Frame-Rate reduzieren")
        minute_frame_rate = st.number_input(
            "Frames pro Minute",
            min_value=1,
            max_value=match.frame_rate * 60,
            key="minute_frame_rate",
        )
        st.number_input(
            "Ballbesitz Heimteam in %",
            min_value=0,
            max_value=99,
            key="possesion_value_home",
        )
    df_tracking_filtered = filter_tracking_df_for_das(
        match.tracking_data, minute_frame_rate, match.frame_rate
    )
    df_frameified = frameify_tracking_data(df_tracking_filtered, match)
    st.write(
        f"**Preprocessing für DAS Calculation in {time.time() - start_time:.2f} Sekunden**"
    )
    st.divider()
    # ---------------------------------------------------------

    st.write("### DAS Berechnung durchführen")
    df_tracking_ball = df_frameified[df_frameified["player_id"] == "ball"].reset_index(
        drop=True
    )
    start_time = time.time()
    pitch_result = calculate_dangerous_accessible_space(df_frameified)
    df_frameified["AS"] = pitch_result.acc_space
    df_frameified["DAS"] = pitch_result.das
    frame_amount = df_frameified["frame"].nunique()
    frame_list = df_frameified["frame"].unique()
    st.write(
        f"**Dangerous Accessible Space Calculation in {time.time() - start_time:.2f} Sekunden** mit {frame_amount} Frames"
    )
    defensive_das.plot_total_das(df_frameified)
    st.divider()
    # ---------------------------------------------------------

    # Optimierung von 1 Frame für Entwicklungszwecke / Testen, für tatsächliche Auswertungen wertlos
    # >>> st.button("Optimierung Ein Frame", on_click=click_button, args=["one_frame"])
    # if st.session_state.one_frame_clicked:
    #     (
    #         frame,
    #         input_index,
    #         df_pre_frame,
    #         match_optimized,
    #         df_optimized,
    #         pitch_result_optimized,
    #     ) = one_frame_optimization(
    #         frame_amount, df_tracking_ball, match, pitch_result, df_frameified
    #     )
    #     fig, ax = defensive_das.plot_frame_origin(
    #         match, frame, pitch_result, input_index, df_pre_frame
    #     )
    #     fig, ax = defensive_das.plot_optimal_positions(
    #         fig, ax, match_optimized, frame
    #     )
    #     if st.toggle("Plot Field"):
    #         st.pyplot(fig)
    home_players = match.home_players[["id", "full_name", "position"]].copy()
    home_players["team_id"] = "Home"
    away_players = match.away_players[["id", "full_name", "position"]].copy()
    away_players["team_id"] = "Away"
    players = pd.concat([home_players, away_players], ignore_index=True)
    player_id = st.selectbox(
        "Spieler auswählen",
        players,
        format_func=lambda x: f"{players.loc[players['id'] == x, 'full_name'].values[0]}: {players.loc[players['id'] == x, 'position'].values[0]} ({players.loc[players['id'] == x, 'team_id'].values[0]} Team)",
    )
    player_column_id = match.player_id_to_column_id(player_id)
    st.button(
        "DAS allowed berechnen für Spieler",
        on_click=click_button,
        args=["single_player"],
    )

    # with col2:
    #     st.button("Optimierung Gesamtteam", on_click=click_button, args=["team"])

    if "single_player" in st.session_state and st.session_state.single_player:
        start_time = time.time()
        df_frameified_simulations = single_player_optimize(
            match, frame_list, df_frameified, pitch_result, player_column_id
        )
        st.write(
            f"Optimierung auf Spielerlevel in {time.time() - start_time:.2f} Sekunden"
        )

        st.dataframe(df_frameified_simulations)

    # if "all_frame_clicked" in st.session_state and st.session_state.all_frame_clicked:
    #     start_time = time.time()
    #     df_das_changes, df_tracking_opt = all_frame_optimization(
    #         match, frame_list, pitch_result, df_frameified
    #     )
    #     df_das_changes["difference"] = (
    #         df_das_changes["das_start"] - df_das_changes["das_opt"]
    #     )
    #     das_home = np.mean(
    #         df_das_changes[df_das_changes["def_team"] == "home"]["difference"]
    #     )
    #     das_away = np.mean(
    #         df_das_changes[df_das_changes["def_team"] == "away"]["difference"]
    #     )
    #     st.dataframe(df_das_changes)
    #     st.write(f"DAS Mean Heim Team = {das_home} | DAS Mean Away Team = {das_away}")
    #     st.write(
    #         f"Optimierung von {len(frame_list)} in {time.time() - start_time:.2f} Sekunden"
    #     )
    # -----------------------------------------------------------
    # Match Plotting

    # fig, ax = defensive_das.plot_frame(
    #     match_optimized, frame, pitch_result_optimized, input_index, df_pre_frame
    # )
    # st.pyplot(fig)
    # st.dataframe(match.tracking_data[match.tracking_data["frame"] == 105059])
    # fig, ax = plt.subplots(figsize=(10, 6))
    # fig, ax = plot_soccer_pitch(fig=fig, ax=ax, field_dimen=match.pitch_dimensions)
    # fig, ax = plot_tracking_data(
    #     match,
    #     105059,
    #     fig=fig,
    #     ax=ax,
    #     team_colors=["blue", "red"],
    # )
    # st.pyplot(fig)


if __name__ == "__main__":
    main()
