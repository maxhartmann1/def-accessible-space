from turtle import width
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from databallpy.visualize import plot_soccer_pitch, plot_tracking_data
import pandas as pd

import accessible_space

TEAM_COLORS = {"home": "blue", "away": "red"}
POSS_TO_DEF = {"home": "away", "away": "home"}


def plot_total_das(df):
    color_scale = alt.Scale(domain=["home", "away"], range=["blue", "red"])
    df = df[["frame", "team_possession", "AS", "DAS"]]
    df = df.drop_duplicates(subset=["frame"], keep="first")
    df = df.groupby("team_possession").agg(
        {"AS": ["count", "sum", "mean", "max"], "DAS": ["count", "sum", "mean", "max"]}
    )
    df_das = df[["DAS"]]
    df_das.columns = ["_".join(col) for col in df_das.columns]

    # DAS Count darstellen
    df_das_count = df_das[["DAS_count"]].reset_index()

    count_chart = (
        alt.Chart(df_das_count)
        .mark_bar()
        .encode(
            x=alt.X(
                "DAS_count:Q",
                stack="normalize",
                title="Anteil (relativ)",
            ),
            color=alt.Color(
                "team_possession:N",
                scale=color_scale,
            ),
            order=alt.Order("team_possession:N", sort="descending"),
            tooltip=["team_possession", "DAS_count"],
        )
        .properties(title="DAS Count", width=600)
    )
    count_text = (
        alt.Chart(df_das_count)
        .mark_text(align="center", baseline="middle", dx=-100, color="white")
        .encode(
            x=alt.X(
                "DAS_count:Q",
                stack="normalize",
                title="Anteil (relativ)",
            ),
            order=alt.Order("team_possession:N", sort="descending"),
            text=alt.Text("DAS_count:Q", format=".0f"),
        )
    )

    # Ballbesitz berücksichtigen, wenn ergänzt
    if st.session_state.possesion_value_home > 0:
        df_poss = df_das_count.rename(columns={"DAS_count": "Possession"})
        df_poss.loc[df_poss["team_possession"] == "home", "Possession"] = (
            st.session_state.possesion_value_home
        )
        df_poss.loc[df_poss["team_possession"] == "away", "Possession"] = (
            100 - st.session_state.possesion_value_home
        )
        poss_chart = (
            alt.Chart(df_poss)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Possession:Q",
                    stack="normalize",
                    title="Gemessener Ballbesitz (Quelle Sofascore)",
                ),
                color=alt.Color(
                    "team_possession:N",
                    scale=color_scale,
                    legend=None,
                ),
                order=alt.Order("team_possession:N", sort="descending"),
                tooltip=["team_possession", "Possession"],
            )
            .properties(title="Echter Ballbesitz", width=600)
        )

    df_das = df_das.drop("DAS_count", axis=1)
    metriks = df_das.columns.tolist()
    df_das = df_das.reset_index()
    df_das["defending_team"] = df_das["team_possession"].map(POSS_TO_DEF)
    df_das_melted = df_das.melt(
        id_vars=["team_possession", "defending_team"],
        var_name="Metrik",
        value_name="Wert",
    )
    charts = []
    for metrik in metriks:
        chart = (
            alt.Chart(df_das_melted[df_das_melted["Metrik"] == metrik])
            .mark_bar()
            .encode(
                x=alt.X(
                    "defending_team:N",
                    title="Verteidigende Mannschaft",
                    sort=["home", "away"],
                ),
                y=alt.Y("Wert:Q", title=metrik),
                color=alt.Color("team_possession:N", scale=color_scale),
            )
            .properties(title=metrik, width=150, height=250)
        )
        text = (
            alt.Chart(df_das_melted[df_das_melted["Metrik"] == metrik])
            .mark_text(align="center", baseline="middle", dy=-10)
            .encode(
                x=alt.X(
                    "defending_team:N",
                    sort=["home", "away"],
                ),
                y="Wert:Q",
                text=alt.Text("Wert:Q", format=".1f"),  # Zahlenformat anpassen
            )
        )
        charts.append(chart + text)

    (
        st.altair_chart(
            alt.vconcat((count_chart + count_text), poss_chart, alt.hconcat(*charts))
        )
        if st.session_state.possesion_value_home > 0
        else st.altair_chart(
            alt.vconcat((count_chart + count_text), alt.hconcat(*charts))
        )
    )


def plot_frame_origin(game, game_idx, pitch_result, pitch_result_idx, player_column_id):
    # fig, ax = plt.subplots(figsize=(12, 8))
    fig, ax = plot_soccer_pitch(field_dimen=game.pitch_dimensions, pitch_color="white")
    fig, ax = plot_tracking_data(
        game,
        game_idx,
        fig=fig,
        ax=ax,
        team_colors=["red", "blue"],
        add_player_possession=True,
        variable_of_interest=round(float(pitch_result.das.iloc[pitch_result_idx]), 2),
    )

    try:
        accessible_space.plot_expected_completion_surface(
            pitch_result.dangerous_result, frame_index=pitch_result_idx
        )
    except:
        st.warning(f"Fehler beim Plotten von DAS")

    player_x, player_y = game.tracking_data.iloc[game_idx][
        [f"{player_column_id}_x", f"{player_column_id}_y"]
    ]
    circle = plt.Circle(
        (player_x, player_y),
        radius=1,
        color="#00FF00",
        alpha=1,
        fill=False,
        zorder=10,
        linewidth=2,
    )
    ax.add_patch(circle)
    return fig, ax


def plot_frame_random(
    fig,
    ax,
    game,
    game_idx,
    pitch_result,
    das_idx,
    result_idx,
    player_column_id,
    new_frame_random,
):
    delta_x, delta_y = float(new_frame_random.split("_")[3]), float(
        new_frame_random.split("_")[4]
    )
    player_x, player_y = game.tracking_data.iloc[game_idx][
        [f"{player_column_id}_x", f"{player_column_id}_y"]
    ]
    player_x_new = player_x + delta_x
    player_y_new = player_y + delta_y
    game.tracking_data.at[game_idx, f"{player_column_id}_x"] = player_x_new
    game.tracking_data.at[game_idx, f"{player_column_id}_y"] = player_y_new

    fig_rand, ax_rand = plot_soccer_pitch(
        field_dimen=game.pitch_dimensions, pitch_color="white"
    )
    fig_rand, ax_rand = plot_tracking_data(
        game,
        game_idx,
        fig=fig_rand,
        ax=ax_rand,
        team_colors=["red", "blue"],
        add_player_possession=True,
        variable_of_interest=round(float(pitch_result.das.iloc[das_idx]), 2),
    )
    try:
        accessible_space.plot_expected_completion_surface(
            pitch_result.dangerous_result, frame_index=result_idx
        )
    except:
        st.warning(f"Fehler beim Plotten von DAS")

    player_x, player_y = game.tracking_data.iloc[game_idx][
        [f"{player_column_id}_x", f"{player_column_id}_y"]
    ]
    circle = plt.Circle(
        (player_x, player_y),
        radius=1,
        color="#00FF00",
        alpha=1,
        fill=False,
        zorder=10,
        linewidth=2,
    )
    ax_rand.add_patch(circle)
    return fig_rand, ax_rand
