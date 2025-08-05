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

    frame_list_path = (
        Path("cache") / f"frame_list_{provider}_{game_id}_step{frame_step_size}.csv"
    )
    frame_list_path.parent.mkdir(parents=True, exist_ok=True)
    if not frame_list_path.exists():
        np.savetxt(frame_list_path, frame_list, delimiter=",", fmt="%.0f")

    handle_das_result(df_frameified, frame_step_size, provider, game_id)
    st.divider()

    # Optimierung Spielerposition
    sim_result = render_player_optimization(
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
            format_func=lambda x: games[x] + f": {x}",
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
            st.session_state.frame_filter_step = game.tracking_data.frame_rate * 5
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
    game, df_frameified, pitch_result, frame_step_size, provider, game_id, frame_list
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
    with st.expander("Parameter setzen"):
        max_radius, opt_step_size, min_teammate_dist, sample_n = (
            get_default_parameters()
        )
        render_parameter_selection(
            max_radius, opt_step_size, min_teammate_dist, sample_n
        )
        if st.button("Optimierung mit individuellen Parametern starten"):
            st.session_state.run_parameter_optimization = True

    if st.button("Optimierung starten"):
        st.session_state.run_optimization = True

    sim_result = {}
    if st.session_state.get("run_optimization"):
        df_pre_frames = prepare_optimization(game, frame_list)

        for method in methods:
            simulation_path = get_simulation_path(
                game_id, frame_step_size, player_column_id, method
            )

            df_path = simulation_path / "df_sim.csv"
            pitch_result_path = simulation_path / "pitch_res.pkl"
            if df_path.exists() and pitch_result_path.exists():
                df_frameified_simulations = pd.read_csv(df_path)
                pitch_result_optimized = joblib.load(pitch_result_path)
                st.success("Pitch Result geladen aus Cache.")
            else:
                df_frameified_simulations, pitch_result_optimized, new_frame_list = (
                    defensive_das.optimize_player_position(
                        df_frameified,
                        df_pre_frames,
                        frame_list,
                        player_column_id,
                        pitch_result,
                        method,
                    )
                )
                df_frameified_simulations.to_csv(df_path, index=False)
                joblib.dump(pitch_result_optimized, pitch_result_path)
                save_new_frame_list(
                    new_frame_list, method, game_id, frame_step_size, player_column_id
                )
                st.success("Pitch Result berechnet und gespeichert.")

            df_frameified_simulations = reduce_df_simulations(df_frameified_simulations)
            sim_result[method] = df_frameified_simulations

        # if "all_positions" in methods:
        #     df_frameified_all_positions = defensive_das.optimize_player_position(
        #         df_frameified,
        #         df_pre_frames,
        #         frame_list,
        #         player_column_id,
        #         pitch_result,
        #         method="all_positions",
        #     )

        # if "random" in methods:
        #     df_frameified_random, pitch_result_random = (
        #         defensive_das.optimize_player_position(
        #             df_frameified,
        #             df_pre_frames,
        #             frame_list,
        #             player_column_id,
        #             pitch_result,
        #             method="random",
        #         )
        #     )
        #     st.write(df_frameified_random)
        #     st.write(pitch_result_random)

        # if "interpolate" in methods:
        #     df_frameified_interpolate = defensive_das.optimize_player_position(
        #         df_frameified,
        #         df_pre_frames,
        #         frame_list,
        #         player_column_id,
        #         pitch_result,
        #         method="interpolate",
        #     )

        # st.write(sim_result)

    if st.session_state.get("run_parameter_optimization"):
        df_pre_frames = prepare_optimization(game, frame_list)

        max_radius = st.session_state.get("max_radius")
        opt_step_size = st.session_state.get("opt_step_size")
        min_teammate_dist = st.session_state.get("min_teammate_dist")
        consider_teammates = st.session_state.get("consider_teammates")
        sample_n = st.session_state.get("sample_n")
        file_prefix = f"{max_radius}_{opt_step_size}_{min_teammate_dist}_{sample_n}_"

        for method in methods:
            simulation_path = get_simulation_path(
                game_id, frame_step_size, player_column_id, method
            )

            df_path = simulation_path / f"{file_prefix}df_sim.csv"
            pitch_result_path = simulation_path / f"{file_prefix}pitch_res.pkl"
            if df_path.exists() and pitch_result_path.exists():
                df_frameified_simulations = pd.read_csv(df_path)
                pitch_result_optimized = joblib.load(pitch_result_path)
                st.success("Pitch Result geladen aus Cache.")
            else:
                df_frameified_simulations, pitch_result_optimized, new_frame_list = (
                    defensive_das.optimize_player_position(
                        df_frameified,
                        df_pre_frames,
                        frame_list,
                        player_column_id,
                        pitch_result,
                        method,
                        min_teammate_dist=min_teammate_dist,
                        sample_n=sample_n,
                        max_radius=max_radius,
                        step_size=opt_step_size,
                        consider_teammate_dist=consider_teammates,
                    )
                )
                df_frameified_simulations.to_csv(df_path, index=False)
                joblib.dump(pitch_result_optimized, pitch_result_path)
                save_new_frame_list(
                    new_frame_list, method, game_id, frame_step_size, player_column_id
                )
                st.success("Pitch Result berechnet und gespeichert.")

            df_frameified_simulations = reduce_df_simulations(df_frameified_simulations)
            sim_result[method] = df_frameified_simulations

    st.session_state.run_optimization = False
    st.session_state.run_parameter_optimization = False

    save_sim_result(sim_result, game_id, frame_step_size, player_column_id)

    # if st.button("Neue Frame Liste erstellen"):
    #     df_pre_frames = prepare_optimization(game, frame_list)
    #     for method in methods:
    #         frame_list_path = (
    #             Path("cache")
    #             / f"simulations/{game_id}/step{frame_step_size}/{player_column_id}/{method}/frame_list_new.csv"
    #         )
    #         frame_list_path.parent.mkdir(parents=True, exist_ok=True)
    #         if not frame_list_path.exists():
    #             new_frame_list = defensive_das.get_new_frame_list(
    #                 df_frameified,
    #                 df_pre_frames,
    #                 frame_list,
    #                 player_column_id,
    #                 pitch_result,
    #                 method,
    #                 step_size=opt_step_size,
    #             )
    #             np.savetxt(frame_list_path, new_frame_list, delimiter=",", fmt="%s")
    return sim_result


def prepare_optimization(game, frame_list):
    df = pd.DataFrame(game.tracking_data.copy())
    df_pre_frames = defensive_das.get_pre_frames(
        df, game.tracking_data.frame_rate, frame_list=frame_list
    )
    return df_pre_frames


def get_simulation_path(game_id, frame_step_size, player_id, method):
    simulation_path = (
        Path("cache/simulations")
        / f"{game_id}/step{frame_step_size}/{player_id}/{method}"
    )
    simulation_path.mkdir(parents=True, exist_ok=True)
    return simulation_path


def reduce_df_simulations(df_frameified_simulations):
    df_frameified_simulations = df_frameified_simulations[
        df_frameified_simulations["opt_player"]
        == df_frameified_simulations["player_id"]
    ][["player_id", "frame", "DAS", "DAS_new", "new_frame"]]
    df_frameified_simulations["DAS_potential"] = (
        df_frameified_simulations["DAS"] - df_frameified_simulations["DAS_new"]
    ).clip(lower=0)
    split_cols = df_frameified_simulations["new_frame"].str.split("_", expand=True)
    df_frameified_simulations["move_x"] = split_cols.iloc[:, -2].astype(float)
    df_frameified_simulations["move_y"] = split_cols.iloc[:, -1].astype(float)
    df_frameified_simulations["distance"] = np.sqrt(
        np.square(df_frameified_simulations["move_x"])
        + np.square(df_frameified_simulations["move_y"])
    )
    return df_frameified_simulations


def get_default_parameters():
    return defensive_das.get_default_parameters()


def render_parameter_selection(max_radius, opt_step_size, min_teammate_dist, sample_n):
    st.number_input("Maximaler Radius", 0.0, 15.0, max_radius, 0.1, key="max_radius")
    st.number_input("Step Size", 0.0, 2.0, opt_step_size, 0.01, key="opt_step_size")
    st.toggle("Consider Teammate Distance", True, key="consider_teammates")
    if st.session_state.get("consider_teammates"):
        st.number_input(
            "Min Distance to Teammates",
            0.1,
            5.0,
            min_teammate_dist,
            0.1,
            key="min_teammate_dist",
        )
    else:
        st.session_state.min_teammate_dist = 0
    st.number_input("Sample Size", 1, 100, sample_n, 1, key="sample_n")


def save_sim_result(sim_result, game_id, frame_step_size, player_id):

    for key in sim_result:
        path = (
            Path("simulation_results/reduced/")
            / f"{game_id}/step{frame_step_size}/{player_id}/{key}"
        )
        path.mkdir(parents=True, exist_ok=True)
        max_radius = st.session_state.get("max_radius")
        opt_step_size = st.session_state.get("opt_step_size")
        min_teammate_dist = st.session_state.get("min_teammate_dist")
        sample_n = st.session_state.get("sample_n")
        file_name = (
            f"{max_radius}_{opt_step_size}_{min_teammate_dist}_{sample_n}_reducedDF.csv"
        )

        df_path = path / file_name

        sim_result[key].to_csv(df_path, index=False)

        st.success("Reduzierter Frame gespeichert.")


def save_new_frame_list(
    new_frame_list, method, game_id, frame_step_size, player_column_id
):
    frame_list_path = (
        Path("cache")
        / f"simulations/{game_id}/step{frame_step_size}/{player_column_id}/{method}/frame_list_new.csv"
    )
    frame_list_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(frame_list_path, new_frame_list, delimiter=",", fmt="%s")
