import streamlit as st
import time
import pandas as pd
import numpy as np
from pathlib import Path
import defensive_das
import joblib
from accessible_space.interface import ReturnValueDAS

# from services import defensive_das
from databallpy.utils.constants import OPEN_GAME_IDS_DFL

PLOT_TOTAL_DAS_SWITCH = False


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
    df_frameified, frame_amount, frame_list = attach_das(df_frameified, pitch_result)
    handle_das_result(df_frameified, frame_step_size, provider, game_id)
    st.divider()

    # Optimierung Spielerposition
    render_player_optimization(
        game,
        df_frameified,
        pitch_result,
        frame_step_size,
        provider,
        game_id,
        frame_list,
    )


# Hilfsfunktionen ----------------------------------------------------------------------------------------------------------------


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


def attach_das(df_frameified, pitch_result):
    df_frameified["AS"] = pitch_result.acc_space
    df_frameified["DAS"] = pitch_result.das
    frame_amount = df_frameified["frame"].nunique()
    frame_list = df_frameified["frame"].unique()
    return df_frameified, frame_amount, frame_list


def handle_das_result(df_frameified, frame_step_size, provider, game_id):
    defensive_das.total_das_to_csv(df_frameified, frame_step_size, provider, game_id)
    if PLOT_TOTAL_DAS_SWITCH:
        defensive_das.plot_total_das(df_frameified)


def render_player_optimization(
    game, df_frameified, pitch_result, step_size, provider, game_id, frame_list
):
    st.write("## Optimale Position für Verteidiger")

    home_players = game.home_players[["id", "full_name", "position"]].copy()
    home_players["team_id"] = "Home"
    away_players = game.away_players[["id", "full_name", "position"]].copy()
    away_players["team_id"] = "Away"
    players = pd.concat([home_players, away_players], ignore_index=True)
    players["player_column_id"] = players["id"].apply(game.player_id_to_column_id)
    player = st.selectbox(
        "Spieler auswählen",
        players,
        format_func=lambda x: (
            f"{players.loc[players['id'] == x, 'full_name'].values[0]} "
            f"({players.loc[players['id'] == x, 'position'].values[0]}) von "
            f"{players.loc[players['id'] == x, 'team_id'].values[0]} Team | "
            f"{players.loc[players['id'] == x, 'player_column_id'].values[0]}"
        ),
    )
    player_column_id = players.loc[players["id"] == player, "player_column_id"].values[
        0
    ]
    methods = st.multiselect(
        "Optimierungsmethode wählen", ["all_positions", "random", "interpolate"]
    )

    if st.button("Optimierung starten"):
        st.session_state.run_optimization = True

    if st.session_state.get("run_optimization"):
        df_pre_frames = prepare_optimization(game, frame_list)

        if "all_positions" in methods:
            st.write("Run all_positions")

        if "random" in methods:
            st.write("Run random")

        if "interpolate" in methods:
            st.write("Run interpolate")

    st.session_state.run_optimization = False


def prepare_optimization(game, frame_list):
    df = pd.DataFrame(game.tracking_data.copy())
    df_pre_frames = defensive_das.get_pre_frames(
        df, game.tracking_data.frame_rate, frame_list=frame_list
    )
