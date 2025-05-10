import importlib

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import requests

import accessible_space.tests.test_model

import accessible_space.core
import accessible_space.interface
importlib.reload(accessible_space.core)
importlib.reload(accessible_space.utility)
importlib.reload(accessible_space.interface)

from accessible_space.tests.test_model import _get_butterfly_data
import accessible_space.tests.test_real_world_data


def eval_benchmark():
    owner = "EAISI"
    repo = "OJN-EPV-benchmark"
    path = "OJN-Pass-EPV-benchmark"  # Root directory; you can specify any subdirectory path here
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

    response = requests.get(url)
    files = response.json()
    import accessible_space

    # Iterate over the files and print file names
    data = []
    for dir in accessible_space.progress_bar(files, total=len(files), desc="Loading files"):
        dir = dir["name"]
        datapoint = {}
        file_mod = f"https://raw.githubusercontent.com/EAISI/OJN-EPV-benchmark/refs/heads/main/OJN-Pass-EPV-benchmark/{dir}/modification.csv"
        df_mod = pd.read_csv(file_mod, encoding="utf-8")
        higher_state = df_mod["higher_state_id"].iloc[0]
        datapoint["higher_state"] = df_mod["higher_state_id"].iloc[0]
        cols = st.columns(2)

        index2df = {}

        for index in [1, 2]:
            file_gamestate = f"https://raw.githubusercontent.com/EAISI/OJN-EPV-benchmark/refs/heads/main/OJN-Pass-EPV-benchmark/{dir}/game_state_{index}.csv"
            df = pd.read_csv(file_gamestate, encoding="utf-8")

            player2team = df[['player', 'team']].drop_duplicates().set_index('player').to_dict()['team']
            assert len(set(df["event_player"])) == 1
            event_player = df["event_player"].iloc[0]
            event_team = player2team[event_player]
            df["team_in_possession"] = event_team
            df["frame_id"] = 0
            df["playing_direction_event"] = df["playing_direction_event"].map({True: 1, False: -1})

            # set ball coordinates to passer coordinates
            passer_x = df[df["player"] == event_player]["pos_x"].iloc[0]
            passer_y = df[df["player"] == event_player]["pos_y"].iloc[0]
            # df["pos_x"] = df["pos_x"].where(df["player"] != 0, passer_x)
            # df["pos_y"] = df["pos_y"].where(df["player"] != 0, passer_y)

            ret = accessible_space.get_dangerous_accessible_space(
                df, frame_col='frame_id', player_col='player', team_col='team', x_col='pos_x', y_col='pos_y',
                vx_col='smooth_x_speed', vy_col='smooth_y_speed', team_in_possession_col="team_in_possession",
                period_col=None, ball_player_id=0, attacking_direction_col="playing_direction_event",
                infer_attacking_direction=False, passer_col="event_player",
            )
            das = ret.das.iloc[0]
            acc_space = ret.acc_space.iloc[0]

            datapoint[f"as_{index}"] = acc_space
            datapoint[f"das_{index}"] = das

            plt.figure()
            df["team_is_in_possession"] = df["team"] == df["team_in_possession"]
            df["team_is_ball"] = df["player"] == 0
            df["team_is_defending"] = (df["team"] != df["team_in_possession"]) & ~df["team_is_ball"]
            df["color"] = df["team_is_in_possession"].map({True: "red", False: "blue"})
            df["color"] = df["color"].where(~df["team_is_ball"], "black")
            plt.scatter(df["pos_x"], df["pos_y"], c=df["color"], cmap="viridis", alpha=1)

            # plot the passer extra
            plt.scatter(passer_x, passer_y, c="orange", marker="x", s=10, label="Passer")

            # plot velocities
            for i, row in df.iterrows():
                plt.arrow(row["pos_x"], row["pos_y"], row["smooth_x_speed"], row["smooth_y_speed"],
                          head_width=0.5, head_length=0.5, fc='black', ec='black', alpha=1)

            plt.xlim(-52.5, 52.5)
            plt.plot([0, 0], [-34, 34], color="black", linewidth=2)
            plt.ylim(-34, 34)
            # accessible_space.plot_expected_completion_surface(ret.simulation_result, 0, color="blue")
            accessible_space.plot_expected_completion_surface(ret.dangerous_result, 0, color="red")
            cols[index-1].write(f"{index}, {df['playing_direction_event'].iloc[0]}, {higher_state=}, {das=}, {acc_space=}")
            cols[index-1].write(plt.gcf())
            # cols[index-1].write(df)

            index2df[index] = df

        # check if dfs only differ by z coordinate of the ball
        index2df[1]["pos_z"] = None
        index2df[2]["pos_z"] = None

        # check if the two dataframes are equal
        if index2df[1].equals(index2df[2]):
            datapoint["differs_only_by_z"] = True
        else:
            datapoint["differs_only_by_z"] = False

        st.write(f"{higher_state=}")
        st.write("-----")

        # st.stop()
        data.append(datapoint)



    df = pd.DataFrame(data)

    def is_correct(row):
        # if row["das_1"] == row["das_2"]:
        #     raise ValueError(f"Das values are equal: {row['das_1']} and {row['das_2']}")
        if row["higher_state"] == 1:
            return row["das_1"] > row["das_2"]
        elif row["higher_state"] == 2:
            return row["das_1"] < row["das_2"]
        else:
            raise ValueError(f"Unknown higher state: {row['higher_state']}")

    df["correct"] = df.apply(lambda row: is_correct(row), axis=1)
    st.write("df")
    st.write(df)

    mean_correct = df["correct"].mean()

    mean_correct_not_only_differs_by_z = df[df["differs_only_by_z"] == False]["correct"].mean()

    st.write("mean_correct")
    st.write(mean_correct)

    st.write("mean_correct_not_only_differs_by_z")
    st.write(mean_correct_not_only_differs_by_z)

    # import streamlit as st
    #
    # # print all members of accessible_space.interface
    # import accessible_space.interface
    # st.write(dir(accessible_space.interface))
    #
    # accessible_space.tests.test_real_world_data.test_real_world_data(1)
    #


if __name__ == '__main__':
    eval_benchmark()
