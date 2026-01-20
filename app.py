import argparse
from rich.console import Console

from data_quality import possession_spread_report


# def load_game(provider, game_id):
#     game = defensive_das.load_game_data(provider, game_id)
#     prep_game = defensive_das.prep_game_data(game.name, game)
#     return prep_game


# def get_open_source_games():
#     games = OPEN_GAME_IDS_DFL
#     games["metrica"] = "Metrica Anonymisiertes Spiel"
#     return games


# def frameify_tracking_data(df_tracking, game):
#     coordinate_cols = []
#     player_to_team = {}
#     players = game.get_column_ids()
#     players.append("ball")
#     for player in players:
#         coordinate_cols.append(
#             [f"{player}_x", f"{player}_y", f"{player}_vx", f"{player}_vy"]
#         )
#         player_to_team[str(player)] = player.split("_")[0]

#     df_tracking = defensive_das.frameify_tracking_data(
#         df_tracking, coordinate_cols, players, player_to_team
#     )
#     return df_tracking


# def filter_tracking_df_for_das(
#     tracking_data, frame_step_size, frame_rate, method="frame_rate", dead_ball=True
# ):
#     df = pd.DataFrame(tracking_data.copy())
#     if dead_ball:
#         df = df[df["ball_status"] == "alive"]

#     if method == "frame_rate":
#         return df[frame_rate::frame_step_size]

#     else:
#         return df


# def calculate_dangerous_accessible_space(df_frameified):
#     return defensive_das.get_dangerous_accessible_space(df_frameified)


# def single_player_heuristic(game, frame_list, df_frameified, player_column_id):
#     df_tracking = pd.DataFrame(game.tracking_data.copy())
#     df_pre_frames = defensive_das.get_pre_frames(
#         df_tracking, game.tracking_data.frame_rate, frame_list=frame_list
#     )
#     df_frameified_simulations = defensive_das.optimize_all_pos_in_reach(
#         df_frameified, df_pre_frames, frame_list, player_column_id
#     )
#     return df_frameified_simulations


# def random_heuristic(game, frame_list, df_frameified, player_column_id):
#     df_tracking = pd.DataFrame(game.tracking_data.copy())
#     df_pre_frames = defensive_das.get_pre_frames(
#         df_tracking, game.tracking_data.frame_rate, frame_list=frame_list
#     )
#     df_frameified_simulations = defensive_das.optimize_random_pos(
#         df_frameified, df_pre_frames, frame_list, player_column_id
#     )
#     return df_frameified_simulations


# def interpolate_heuristic(
#     game, frame_list, df_frameified, player_column_id, pitch_result
# ):
#     df_tracking = pd.DataFrame(game.tracking_data.copy())
#     df_pre_frames = defensive_das.get_pre_frames(
#         df_tracking, game.tracking_data.frame_rate, frame_list=frame_list
#     )
#     df_frameified_simulations = defensive_das.optimize_topn_pitchdas(
#         df_frameified, df_pre_frames, frame_list, player_column_id, pitch_result
#     )
#     return df_frameified_simulations


# def reduce_df_simulations(df_frameified_simulations):
#     df_frameified_simulations = df_frameified_simulations[
#         df_frameified_simulations["opt_player"]
#         == df_frameified_simulations["player_id"]
#     ][["player_id", "frame", "DAS", "DAS_new"]]
#     df_frameified_simulations["DAS_potential"] = (
#         df_frameified_simulations["DAS"] - df_frameified_simulations["DAS_new"]
#     ).clip(lower=0)
#     return df_frameified_simulations


# def result_to_file(match, spieler, df, frame_frequenz, optimierung):
#     filepath = Path(
#         f"simulation_results/{match[0]}/{match[1]}/{frame_frequenz}/{optimierung}/{spieler}.csv"
#     )
#     filepath.parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(filepath, index=False)


# def click_button(button):
#     if button not in st.session_state:
#         st.session_state[button] = True
#     else:
#         st.session_state[button] = not st.session_state[button]


# def reset_possesion():
#     st.session_state.possesion_value_home = 0


# # -------------------------------------------------------------


# def main():

#     st.write(st.session_state)
#     st.markdown(
#         "<h1 style='text-align: center;'>Defensive Metric based on DAS</h1>",
#         unsafe_allow_html=True,
#     )

#     # Verfügbare Spiele laden
#     col1, col2 = st.columns(2)
#     games = get_open_source_games()
#     with col1:
#         st.write("### Spiel auswählen")
#         game_id = st.selectbox(
#             "Open Source Match auswählen",
#             games,
#             index=7,
#             format_func=lambda x: games[x],
#             on_change=reset_possesion,
#         )
#     provider = "metrica" if game_id == "metrica" else "dfl"

#     # Ausgewähltes Spiel laden
#     start_time = time.time()
#     game = load_game(provider, game_id)
#     with col1:
#         st.write(
#             f"{game.home_team_name} {game.home_score} - {game.away_score} {game.away_team_name} provided by {game.tracking_data.provider} mit Frame Rate von {game.tracking_data.frame_rate} Frames / Sekunde."
#         )
#         st.number_input(
#             "Ballbesitz Heimteam in %",
#             min_value=0,
#             max_value=99,
#             key="possesion_value_home",
#         )

#     # Anzahl zu verarbeitender Frames wählen
#     with col2:
#         if "minute_frame_rate" not in st.session_state:
#             st.session_state.minute_frame_rate = game.tracking_data.frame_rate * 60
#         st.write("### Frame-Rate reduzieren")
#         st.write(
#             f"Step size eingeben für Frame Filter. **{game.tracking_data.frame_rate}** entspricht 1 Frame pro Sekunde."
#         )
#         frame_step_size = st.number_input(
#             "Step Size für Frames",
#             min_value=1,
#             max_value=game.tracking_data.frame_rate * 60,
#             key="minute_frame_rate",
#         )

#     df_tracking_filtered = filter_tracking_df_for_das(
#         game.tracking_data, frame_step_size, game.tracking_data.frame_rate
#     )
#     df_frameified = frameify_tracking_data(df_tracking_filtered, game)
#     st.write(
#         f"**Preprocessing für DAS Calculation in {time.time() - start_time:.2f} Sekunden**"
#     )
#     st.divider()
#     # ---------------------------------------------------------

#     st.write("### DAS Berechnung durchführen")

#     start_time = time.time()
#     pitch_result = calculate_dangerous_accessible_space(df_frameified)

#     df_frameified["AS"] = pitch_result.acc_space
#     df_frameified["DAS"] = pitch_result.das
#     frame_amount = df_frameified["frame"].nunique()
#     frame_list = df_frameified["frame"].unique()
#     st.write(
#         f"**Dangerous Accessible Space Calculation in {time.time() - start_time:.2f} Sekunden** mit {frame_amount} Frames"
#     )
#     defensive_das.total_das_to_csv(df_frameified, frame_step_size, provider, game_id)
#     defensive_das.plot_total_das(df_frameified)
#     st.divider()
#     # ---------------------------------------------------------

#     home_players = game.home_players[["id", "full_name", "position"]].copy()
#     home_players["team_id"] = "Home"
#     away_players = game.away_players[["id", "full_name", "position"]].copy()
#     away_players["team_id"] = "Away"
#     players = pd.concat([home_players, away_players], ignore_index=True)
#     player_id = st.selectbox(
#         "Spieler auswählen",
#         players,
#         format_func=lambda x: f"{players.loc[players['id'] == x, 'full_name'].values[0]}: {players.loc[players['id'] == x, 'position'].values[0]} ({players.loc[players['id'] == x, 'team_id'].values[0]} Team)",
#     )
#     player_column_id = game.player_id_to_column_id(player_id)
#     # st.button(
#     #     "DAS allowed berechnen für Spieler",
#     #     on_click=click_button,
#     #     args=["single_player"],
#     # )
#     st.button(
#         "DAS Optimize All Positions",
#         on_click=click_button,
#         args=["heuristic"],
#     )
#     st.button(
#         "DAS Random",
#         on_click=click_button,
#         args=["random"],
#     )
#     st.button(
#         "DAS Interpolate",
#         on_click=click_button,
#         args=["interpolate"],
#     )

#     # if "single_player" in st.session_state and st.session_state.single_player:
#     #     start_time = time.time()
#     #     df_frameified_simulations = single_player_optimize(
#     #         game, frame_list, df_frameified, pitch_result, player_column_id
#     #     )
#     #     st.write(
#     #         f"Optimierung auf Spielerlevel in {time.time() - start_time:.2f} Sekunden"
#     #     )
#     #     start_time = time.time()
#     #     # st.dataframe(
#     #     #     df_frameified_simulations[df_frameified_simulations["frame"] == 45080]
#     #     # )
#     #     df_frameified_simulations = reduce_df_simulations(df_frameified_simulations)
#     #     st.dataframe(df_frameified_simulations)

#     if "heuristic" in st.session_state and st.session_state.heuristic:
#         start_time = time.time()
#         df_frameified_simulations = single_player_heuristic(
#             game, frame_list, df_frameified, player_column_id
#         )
#         st.write(
#             f"Heuristik auf Spielerlevel in {time.time() - start_time:.2f} Sekunden"
#         )
#         start_time = time.time()

#         df_frameified_simulations = reduce_df_simulations(df_frameified_simulations)
#         st.dataframe(df_frameified_simulations)

#     if "random" in st.session_state and st.session_state.random:
#         start_time = time.time()
#         (
#             df_frameified_random,
#             pitch_result_random,
#             new_frame_random,
#             das_idx_rd,
#             res_idx_rd,
#         ) = random_heuristic(game, frame_list, df_frameified, player_column_id)
#         st.write(f"Random auf Spielerlevel in {time.time() - start_time:.2f} Sekunden")
#         start_time = time.time()

#         df_frameified_random = reduce_df_simulations(df_frameified_random)
#         st.dataframe(df_frameified_random)

#     if "interpolate" in st.session_state and st.session_state.interpolate:
#         start_time = time.time()
#         (
#             df_frameified_interpolate,
#             pitch_result_interpolate,
#             new_frame_inter,
#             das_idx_inter,
#             res_idx_inter,
#         ) = interpolate_heuristic(
#             game, frame_list, df_frameified, player_column_id, pitch_result
#         )
#         st.write(
#             f"Interpolation auf Spielerlevel in {time.time() - start_time:.2f} Sekunden"
#         )
#         start_time = time.time()

#         df_frameified_interpolate = reduce_df_simulations(df_frameified_interpolate)
#         st.dataframe(df_frameified_interpolate)

#     if st.button("Ergebnis in CSV speichern"):
#         if "heuristic" in st.session_state and st.session_state.heuristic:
#             if not df_frameified_simulations.empty:
#                 result_to_file(
#                     (provider, game_id),
#                     player_column_id,
#                     df_frameified_simulations,
#                     frame_step_size,
#                     "all_positions",
#                 )
#         if "random" in st.session_state and st.session_state.random:
#             if not df_frameified_random.empty:
#                 result_to_file(
#                     (provider, game_id),
#                     player_column_id,
#                     df_frameified_random,
#                     frame_step_size,
#                     "random",
#                 )
#         if "interpolate" in st.session_state and st.session_state.interpolate:
#             if not df_frameified_interpolate.empty:
#                 result_to_file(
#                     (provider, game_id),
#                     player_column_id,
#                     df_frameified_interpolate,
#                     frame_step_size,
#                     "interpolate",
#                 )

#     # Match Plotting

#     frame = df_frameified_interpolate.iloc[0]["frame"]
#     game_idx = game.tracking_data[game.tracking_data["frame"] == frame].index[0]
#     pitch_result_idx = np.where(frame_list == frame)[0][0]
#     fig, ax = defensive_das.plot_frame_origin(
#         game, game_idx, pitch_result, pitch_result_idx, player_column_id
#     )
#     st.pyplot(fig)

#     game_random = game.copy()
#     fig, ax = defensive_das.plot_frame_random(
#         fig,
#         ax,
#         game_random,
#         game_idx,
#         pitch_result_random,
#         das_idx_rd,
#         res_idx_rd,
#         player_column_id,
#         new_frame_random,
#     )
#     st.pyplot(fig)

#     game_inter = game.copy()
#     fig, ax = defensive_das.plot_frame_random(
#         fig,
#         ax,
#         game_inter,
#         game_idx,
#         pitch_result_interpolate,
#         das_idx_inter,
#         res_idx_inter,
#         player_column_id,
#         new_frame_inter,
#     )
#     st.pyplot(fig)


def main_new():
    st.set_page_config(layout="wide", page_title="DAS Optimizer")
    dashbord.render()


def main_ohne_streamlit(args):
    cli_logic.calculate(args)


def is_streamlit_env():
    import streamlit.runtime as rt

    if hasattr(rt, "exists") and rt.exists():
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["auto", "streamlit", "cli"],
        default="auto",
        help="auto erkennt Streamlit-Laufzeit; oder explizit streamlit/cli",
    )
    parser.add_argument("--game_id", default="J03WQQ")
    parser.add_argument("--step_size", type=int, default=125)
    parser.add_argument("--player", nargs="+", default=["home_25"])
    parser.add_argument(
        "--method",
        nargs="+",
        choices=["random", "grid_search", "all_positions"],
        default=["random"],
    )
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--opt_step_size", type=float, default=1)
    parser.add_argument("--min_dist", type=float, default=2)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--cut", type=float, default=0.0)
    parser.add_argument("--fine_step", type=float)
    parser.add_argument("--fine_radius", type=float)
    args = parser.parse_args()
    if args.opt_step_size % 1 == 0:
        args.opt_step_size = int(args.opt_step_size)

    if args.quality:
        possession_spread_report()
    elif args.mode == "streamlit" or (args.mode == "auto" and is_streamlit_env()):
        import streamlit as st
        import dashbord

        main_new()
    else:
        console = Console()
        console.rule("[bold cyan]Defensive DAS[/bold cyan]", style="cyan")
        console.print("🚀 Programm startet...", style="bold green")
        import cli_logic

        main_ohne_streamlit(args)
