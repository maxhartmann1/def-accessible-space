import streamlit as st
import time
import pandas as pd
import numpy as np
from pathlib import Path
import defensive_das
import joblib

# from services import defensive_das
from databallpy.utils.constants import OPEN_GAME_IDS_DFL


def render():
    st.write(st.session_state)
    st.markdown(
        "<h1 style='text-align: center;'>Defensvie Metric based on DAS</h1>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    # Spielauswahl
    games = get_open_source_games()
    provider, game_id = render_game_selector(games, col1)
    start_time = time.time()
    game = load_game(provider, game_id)
    st.write(
        f"**Preprocessing für DAS Calculation in {time.time() - start_time:.2f} Sekunden**"
    )

    render_game_info(game, provider, col1)

    # Frame-Auswahl und Preprocessing
    frame_step_size = render_frame_step_selector(game, col2)

    df_tracking_filtered = filter_tracking_data(game, frame_step_size)
    df_frameified = frameify_tracking_data(df_tracking_filtered, game)

    st.divider()

    # DAS-Berechnung
    pitch_result = calculate_pitch_result(
        df_frameified, provider, game_id, frame_step_size
    )


# Hilfsfunktionen


def get_open_source_games():
    games = OPEN_GAME_IDS_DFL
    games["metrica"] = "Metrica Anonymisiertes Spiel"
    return games


def load_game(provider, game_id):
    game = defensive_das.load_game_data(provider, game_id)
    prep_game = defensive_das.prep_game_data(game.name, game)
    return prep_game


def render_game_selector(games, col):
    with col:
        game_id = st.selectbox(
            "Open Source Match auswählen",
            games,
            index=7,
            format_func=lambda x: games[x],
        )
    provider = "metrica" if game_id == "metrica" else "dfl"
    return provider, game_id


def render_game_info(game, provider, col):
    with col:
        st.write(
            f"{game.home_team_name} {game.home_score} - {game.away_score} {game.away_team_name} provided by {provider} mit Frame Rate von {game.tracking_data.frame_rate} Frames / Sekunde."
        )
        st.number_input(
            "Ballbesitz Heimteam in %",
            min_value=0,
            max_value=99,
            key="possesion_value_home",
        )


def render_frame_step_selector(game, col):
    with col:
        if "frame_filter_step" not in st.session_state:
            st.session_state.frame_filter_step = game.tracking_data.frame_rate * 60
        frame_filter_step = st.number_input(
            "Step size für Frames",
            min_value=1,
            max_value=game.tracking_data.frame_rate * 60,
            key="frame_filter_step",
        )
    return frame_filter_step


def filter_tracking_data(game, step_size):
    df = pd.DataFrame(game.tracking_data.copy())
    df = df[df["ball_status"] == "alive"]
    return df[game.tracking_data.frame_rate :: step_size]


def frameify_tracking_data(df_tracking, game):
    coordinate_cols = []
    player_to_team = {}
    players = game.get_column_ids()
    players.append("ball")
    for player in players:
        coordinate_cols.append(
            [f"{player}_x", f"{player}_y", f"{player}_vx", f"{player}_vy"]
        )
        player_to_team[str(player)] = player.split("_")[0]

    df_tracking = defensive_das.frameify_tracking_data(
        df_tracking, coordinate_cols, players, player_to_team
    )
    return df_tracking


def calculate_pitch_result(df_frameified, provider, game_id, frame_step_size):
    pitch_result_path = (
        Path("cache") / f"pitch_result_{provider}_{game_id}_step{frame_step_size}.pkl"
    )
    pitch_result_path.parent.mkdir(parents=True, exist_ok=True)

    if pitch_result_path.exists():
        pitch_result = joblib.load(pitch_result_path)
        st.success("Pitch Result geladen aus Cache.")
    else:
        pitch_result = defensive_das.get_dangerous_accessible_space(df_frameified)
        joblib.dump(pitch_result, pitch_result_path)
        st.success("Pitch Result berechnet und gespeichert.")
    return pitch_result
