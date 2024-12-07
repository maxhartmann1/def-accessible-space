import matplotlib.pyplot as plt
import pandas as pd

import accessible_space
import numpy as np
# import streamlit as st


def readme():
    import accessible_space
    from accessible_space.tests.resources import df_passes, df_tracking  # Example data
    import matplotlib.pyplot as plt

    ### 1. Add expected completion to passes
    pass_result = accessible_space.get_expected_pass_completion(df_passes, df_tracking, event_frame_col="frame_id", event_player_col="player_id", event_team_col="team_id", event_start_x_col="x", event_start_y_col="y", event_end_x_col="x_target", event_end_y_col="y_target", tracking_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id", tracking_ball_possession_col="team_in_possession", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", ball_tracking_player_id="ball")
    df_passes["xC"] = pass_result.xc  # Expected pass completion rate
    print(df_passes[["event_string", "xC"]])

    ### 2. Add Dangerous Accessible Space to tracking frames
    pitch_result = accessible_space.get_dangerous_accessible_space(df_tracking, frame_col="frame_id", period_col="period_id", player_col="player_id", team_col="team_id", x_col="x", y_col="y", vx_col="vx", vy_col="vy", possession_team_col="team_in_possession")
    df_tracking["AS"] = pitch_result.acc_space  # Accessible space
    df_tracking["DAS"] = pitch_result.das  # Dangerous accessible space
    print(df_tracking[["frame_id", "team_in_possession", "AS", "DAS"]].drop_duplicates())

    ### 3. Access raw simulation results
    # Example 3.1: Expected interception rate = last value of the cumulative interception probability of the defending team
    pass_result = accessible_space.get_expected_pass_completion(df_passes, df_tracking)
    pass_frame = 0  # We consider the pass at frame 0
    df_passes["frame_index"] = pass_result.event_frame_index  # frame_index implements a mapping from original frame number to indexes of the numpy arrays in the raw simulation_result.
    df_pass = df_passes[df_passes["frame_id"] == pass_frame]  # Consider the pass at frame 0
    frame_index = int(df_pass["frame_index"].iloc[0])
    expected_interception_rate = pass_result.simulation_result.defense_cum_prob[frame_index, 0, -1]  # Frame x Angle x Distance
    print(f"Expected interception rate: {expected_interception_rate:.1%}")

    # Example 3.2: Plot accessible space and dangerous accessible space
    df_tracking["frame_index"] = pitch_result.frame_index

    def plot_constellation(df_tracking_frame):
        plt.figure()
        plt.xlim([-52.5, 52.5])
        plt.ylim([-34, 34])
        plt.scatter(df_tracking_frame["x"], df_tracking_frame["y"], c=df_tracking_frame["team_id"].map({"Home": "red", "Away": "blue"}).fillna("black"), marker="o")
        for _, row in df_tracking_frame.iterrows():
            plt.text(row["x"], row["y"], row["player_id"] if row["player_id"] != "ball" else "")
        plt.gca().set_aspect('equal', adjustable='box')

    df_tracking_frame = df_tracking[df_tracking["frame_id"] == 0]  # Plot frame 0
    frame_index = df_tracking_frame["frame_index"].iloc[0]

    plot_constellation(df_tracking_frame)
    accessible_space.plot_expected_completion_surface(pitch_result.simulation_result, frame_index=frame_index)
    plt.title(f"Accessible space: {df_tracking_frame['AS'].iloc[0]:.0f} m²")

    plot_constellation(df_tracking_frame)
    accessible_space.plot_expected_completion_surface(pitch_result.dangerous_result, frame_index=frame_index, color="red")
    plt.title(f"Dangerous accessible space: {df_tracking_frame['DAS'].iloc[0]:.2f} m²")
    plt.show()

    # Example 3.3: Get (dangerous) accessible space of individual players
    df_tracking["player_index"] = pitch_result.player_index  # Mapping from player to index in simulation_result
    areas = accessible_space.integrate_surfaces(pitch_result.simulation_result)  # Calculate surface integrals
    dangerous_areas = accessible_space.integrate_surfaces(pitch_result.dangerous_result)
    for _, row in df_tracking[(df_tracking["frame_id"] == 0) & (df_tracking["player_id"] != "ball")].iterrows():  # Consider frame 0
        is_attacker = row["team_id"] == row["team_in_possession"]
        acc_space = areas.player_poss[int(frame_index), int(row["player_index"])]
        das = dangerous_areas.player_poss[int(frame_index), int(row["player_index"])]

        plot_constellation(df_tracking_frame)
        accessible_space.plot_expected_completion_surface(pitch_result.simulation_result, "player_poss_density", frame_index=frame_index, player_index=int(row["player_index"]))
        accessible_space.plot_expected_completion_surface(pitch_result.dangerous_result, "player_poss_density", frame_index=frame_index, player_index=int(row["player_index"]), color="red")
        plt.title(f"{row['player_id']} ({'attacker' if is_attacker else 'defender'}) {acc_space:.0f}m² AS and {das:.2f} m² DAS.")
        plt.show()
        # Note: Individual space is not exclusive within a team. This is intentional because your team mates do not take away space from you in the competitive way that your opponents do.  TODO add this to paper.
        print(f"Player {row['player_id']} ({'attacker' if is_attacker else 'defender'}) controls {acc_space:.0f}m² AS and {das:.2f} m² DAS.")


if __name__ == '__main__':
    import importlib
    import accessible_space.tests.test_model_plausibility
    import accessible_space.interface
    import accessible_space.core
    import accessible_space.validation

    importlib.reload(accessible_space.tests.test_model_plausibility)
    importlib.reload(accessible_space.interface)
    importlib.reload(accessible_space.core)
    importlib.reload(accessible_space.validation)

    accessible_space.validation_dashboard()

    import streamlit as st
    st.stop()
    exit(123)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)
    import accessible_space.tests.resources as res
    importlib.reload(res)
    df_passes, df_tracking = res.df_passes, res.df_tracking

    for _, p4ss in df_passes.iterrows():
        df_tracking_frame = df_tracking[df_tracking["frame_id"] == p4ss["frame_id"]]
        plt.figure()
        for team, df_tracking_frame_team in df_tracking_frame.groupby("team_id"):
            plt.scatter(df_tracking_frame_team["x"], df_tracking_frame_team["y"], color="red" if team == "Home" else ("blue" if team == "Away" else "black"))

        df_tracking_frame_ball = df_tracking_frame[df_tracking_frame["player_id"] == "ball"]
        plt.scatter(df_tracking_frame_ball["x"], df_tracking_frame_ball["y"], color="black", marker="x", s=100)

        plt.arrow(p4ss["x"], p4ss["y"], p4ss["x_target"]-p4ss["x"], p4ss["y_target"]-p4ss["y"])

        plt.show()
        st.write(plt.gcf())

        break

    df_passes = df_passes.iloc[:1].copy()

    result = accessible_space.get_expected_pass_completion(df_passes, df_tracking, tracking_frame_col="frame_id", event_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id", ball_tracking_player_id="ball", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", event_start_x_col="x", event_start_y_col="y", event_end_x_col="x_target", event_end_y_col="y_target", event_team_col="team_id", event_player_col="player_id")
    df_passes["xC"], df_passes["frame_index"], simulation_result = result.xc, result.event_frame_index, result.simulation_result  # TODO remove warning

    st.write("df_passes")
    st.write(df_passes)
    st.write(simulation_result.attack_poss_density.shape)
    st.write(simulation_result.attack_cum_prob[int(df_passes["frame_index"].iloc[0]), 0, -1])

    st.write("simulation_result.attack_prob_density.shape")
    st.write(simulation_result.attack_prob_density.shape)
    st.write(simulation_result.player_cum_prob[0, :, 0, :])
    st.write(simulation_result.player_cum_poss[0, :, 0, :])

    st.stop()

    # accessible_space.validation_dashboard()

    # readme()
    import streamlit as st
    st.stop()


    exit(123123)

    # readme()


    # accessible_space.tests.test_model_plausibility._get_double_butterfly_data()
    # accessible_space.tests.test_model_plausibility.test_butterfly_xc()
    # accessible_space.tests.test_model_plausibility.test_double_butterfly_das()
    # accessible_space.tests.test_model_plausibility.test_butterfly_cum_prob_sum_is_1()
    # accessible_space.tests.test_model_plausibility.test_simulation_result_dimensions()
    accessible_space.tests.test_model_plausibility.test_poss_never_below_prob()

    # res = accessible_space.simulate_passes(np.array([[[0, 0, 0, 0], [50, 0, 0, 0]]]), np.array([[0, 0]]), np.array([[0]]), np.array([[10]]), np.array([0]), np.array([0, 1]), players=np.array(["A", "B"]), passers_to_exclude=np.array(["A"]),
    #                                        radial_gridsize=12.3)
    # print(res.r_grid)
    # exit(123)


# import matplotlib.pyplot as plt
# import numpy as np
# import streamlit as st
#
# import accessible_space
# import accessible_space.utility
# import databallpy
# # import databallpy.features
# import databallpy.visualize
#
#
# @st.cache_resource
# def get_data():
#     match = databallpy.get_open_match()
#     return match
#
#
# @st.cache_resource
# def get_preprocessed_data():
#     match = get_data()
#     match.synchronise_tracking_and_event_data()
#     databallpy.add_team_possession(match.tracking_data, match.event_data, match.home_team_id, inplace=True)
#
#     databallpy.add_velocity(
#         match.tracking_data, inplace=True, column_ids=match.home_players_column_ids() + match.away_players_column_ids(),
#         frame_rate=match.frame_rate
#     )
#
#     return match
#
#
# @st.cache_resource
# def _get_preprocessed_data():
#     match = get_preprocessed_data()
#
#     players = match.home_players_column_ids() + match.away_players_column_ids() + ["ball"]
#     frame_col = "frame"
#
#     player_to_team = {}
#     for player in players:
#         if player in match.home_players_column_ids():
#             player_to_team[player] = match.home_team_id
#         elif player in match.away_players_column_ids():
#             player_to_team[player] = match.away_team_id
#         else:
#             player_to_team[player] = None
#
#     coordinate_columns = [[f"{player}_{coord}" for coord in ["x", "y", "vx", "vy", "velocity"]] for player in players]
#     df_tracking = dangerous_accessible_space.per_object_frameify_tracking_data(match.tracking_data, frame_col, coordinate_columns, players, player_to_team, new_coordinate_cols=["x", "y", "vx", "vy", "v"])
#     df_tracking["team_in_possession"] = df_tracking["team_in_possession"].map({"home": match.home_team_id, "away": match.away_team_id})
#
#     df_events = match.event_data
#     df_events["tracking_player_id"] = df_events["player_id"].map(match.player_id_to_column_id)
#
#     return match, df_tracking, df_events
#
#
# def das_vs_xnorm(df_tracking, df_event):
#     df_tracking["attacking_direction"] = dangerous_accessible_space.infer_playing_direction(df_tracking)
#     df_event["attacking_direction"] = df_event["frame_id"].map(df_tracking.set_index("frame_id")["attacking_direction"].to_dict())
#
#     df_passes = df_event[(df_event["is_pass"]) & (~df_event["is_high"])]
#     df_tracking = df_tracking[df_tracking["frame_id"].isin(df_passes["frame_id"])]
#     df_tracking["AS"], df_tracking["DAS"], df_tracking["result_index"], _, _ = dangerous_accessible_space.get_dangerous_accessible_space(
#         df_tracking, tracking_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id",
#     )
#     df_passes["AS"] = df_passes["frame_id"].map(df_tracking.set_index("frame_id")["AS"].to_dict())
#     df_passes["DAS"] = df_passes["frame_id"].map(df_tracking.set_index("frame_id")["DAS"].to_dict())
#     df_passes["result_index"] = df_passes["frame_id"].map(df_tracking.set_index("frame_id")["result_index"].to_dict())
#
#     # correlate x_norm and DAS
#     df_passes["x_norm"] = df_passes["coordinates_x"] * df_passes["attacking_direction"]
#     df_passes["y_norm"] = df_passes["coordinates_y"] * df_passes["attacking_direction"]
#     corr = df_passes[["x_norm", "DAS"]].corr().iloc[0, 1]
#     st.write("Correlation between x_norm and DAS", corr)
#
#     # plot it
#     fig, ax = plt.subplots()
#     ax.scatter(df_passes["x_norm"], df_passes["DAS"])
#     ax.set_xlabel("x_norm")
#     ax.set_ylabel("DAS")
#     st.pyplot(fig)
#
#
# def demo_dashboard():
#     match, df_tracking, df_event = _get_preprocessed_data()
#
#     df_passes = df_event[df_event["databallpy_event"] == "pass"].reset_index()
#
#     df_passes["xc"] = np.nan
#     df_passes["AS"] = np.nan
#
#     # df_passes["index"] = df_passes.index
#
#     ### xC
#     df_passes["xc"], _, _ = dangerous_accessible_space.get_expected_pass_completion(
#         df_passes, df_tracking, event_frame_col="td_frame", tracking_frame_col="frame", event_start_x_col="start_x",
#         event_start_y_col="start_y", event_end_x_col="end_x", event_end_y_col="end_y",
#         event_player_col="tracking_player_id",
#     )
#
#     df_passes = df_passes.iloc[6:30]
#
#     ### AS
#     st.write("df_tracking")
#     st.write(df_tracking.head(50))
#     df_tracking = df_tracking[df_tracking["frame"].isin(df_passes["td_frame"])]
#     df_tracking["AS"], df_tracking["DAS"], df_tracking["result_index"], simulation_result, dangerous_result = dangerous_accessible_space.get_dangerous_accessible_space(
#         df_tracking, infer_attacking_direction=True, tracking_frame_col="frame", tracking_player_col="player_id",
#         tracking_team_col="team_id",
#     )
#     df_passes["AS"] = df_passes["td_frame"].map(df_tracking.set_index("frame")["AS"].to_dict())
#     df_passes["DAS"] = df_passes["td_frame"].map(df_tracking.set_index("frame")["DAS"].to_dict())
#     df_passes["result_index"] = df_passes["td_frame"].map(df_tracking.set_index("frame")["result_index"].to_dict())
#
#     # df_passes = df_passes.sort_values("xc", ascending=True)
#
#     for i, (frame, row) in enumerate(df_passes.iterrows()):
#         plt.figure()
#         fig, ax = databallpy.visualize.plot_soccer_pitch(pitch_color="white")
#         databallpy.visualize.plot_tracking_data(
#             match,
#             row["td_frame"],
#             team_colors=["blue", "red"],
#             title=f"Pass completion: {row['outcome']}",
#             add_velocities=True,
#             variable_of_interest=f"AS={row['AS']:.0f} m^2, xC={row['xc']:.1%}, DAS={row['DAS']:.2f} m^2",
#             # variable_of_interest=f"AS={row['AS']:.0f} m^2",
#             ax=ax,
#         )
#         team_color = "blue" if row["team_id"] == match.home_team_id else "red"
#         def_team_color = "red" if row["team_id"] == match.home_team_id else "blue"
#         plt.arrow(
#             row["start_x"], row["start_y"], row["end_x"] - row["start_x"], row["end_y"] - row["start_y"], head_width=1,
#             head_length=1, fc=team_color, ec=team_color
#         )
#
#         try:
#             fig = dangerous_accessible_space.plot_expected_completion_surface(
#                 dangerous_result, row["result_index"], plot_type_off="poss",
#                 # plot_type_def="poss",
#                 color_off=team_color, color_def=def_team_color, plot_gridpoints=True,
#             )
#         except NameError as e:
#             pass
#
#         st.write(fig)
#         plt.close(fig)
#
#         if i > 30:
#             break
#
#     st.write(fig)
#
#     # profiler.stop()
#
#
# if __name__ == '__main__':
#     # demo_dashboard()
#
#     import accessible_space.tests.resources
#     df_passes = accessible_space.tests.resources.df_passes
#     df_tracking = accessible_space.tests.resources.df_tracking
#
#     df_passes["xc"], df_passes["frame_index"], simulation_result_xc = accessible_space.get_expected_pass_completion(df_passes, df_tracking)
#     print(df_passes)
#
#     df_tracking["AS"], df_tracking["DAS"], df_tracking["frame_index"], simulation_result_as, simulation_result_das = accessible_space.get_dangerous_accessible_space(df_tracking)
#     print(df_tracking)
