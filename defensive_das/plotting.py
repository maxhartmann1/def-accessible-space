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
    df = df[["frame", "ball_possession", "AS", "DAS"]]
    df = df.drop_duplicates(subset=["frame"], keep="first")
    df = df.groupby("ball_possession").agg(
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
                "ball_possession:N",
                scale=color_scale,
            ),
            order=alt.Order("ball_possession:N", sort="descending"),
            tooltip=["ball_possession", "DAS_count"],
        )
        .properties(title="DAS Count", width=600)
    )

    # Ballbesitz berücksichtigen, wenn ergänzt
    if st.session_state.possesion_value_home > 0:
        df_poss = df_das_count.rename(columns={"DAS_count": "Possession"})
        df_poss.loc[df_poss["ball_possession"] == "home", "Possession"] = (
            st.session_state.possesion_value_home
        )
        df_poss.loc[df_poss["ball_possession"] == "away", "Possession"] = (
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
                    "ball_possession:N",
                    scale=color_scale,
                    legend=None,
                ),
                order=alt.Order("ball_possession:N", sort="descending"),
                tooltip=["ball_possession", "Possession"],
            )
            .properties(title="DAS Count", width=1000)
        )

    df_das = df_das.drop("DAS_count", axis=1)
    metriks = df_das.columns.tolist()
    df_das = df_das.reset_index()
    df_das["defending_team"] = df_das["ball_possession"].map(POSS_TO_DEF)
    df_das_melted = df_das.melt(
        id_vars=["ball_possession", "defending_team"],
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
                color=alt.Color("ball_possession:N", scale=color_scale),
            )
            .properties(title=metrik, width=150, height=250)
        )
        charts.append(chart)

    (
        st.altair_chart(alt.vconcat(count_chart, poss_chart, alt.hconcat(*charts)))
        if st.session_state.possesion_value_home > 0
        else st.altair_chart(alt.vconcat(count_chart, alt.hconcat(*charts)))
    )


def plot_frame_origin(match, frame, pitch_result, index_das, df_pre_frame):
    idx_databallpy = match.tracking_data.index[
        match.tracking_data["frame"] == frame
    ].tolist()[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    fig, ax = plot_soccer_pitch(
        fig=fig, ax=ax, field_dimen=match.pitch_dimensions, pitch_color="white"
    )
    fig, ax = plot_tracking_data(
        match,
        idx_databallpy,
        fig=fig,
        ax=ax,
        team_colors=["blue", "red"],
        add_player_possession=True,
        variable_of_interest=round(float(pitch_result.das.iloc[index_das]), 2),
    )

    try:
        accessible_space.plot_expected_completion_surface(
            pitch_result.dangerous_result, frame_index=index_das
        )
    except:
        st.warning(f"Fehler beim Plotten von DAS")

    for _, row in df_pre_frame.iterrows():
        if row["team_id"] != row["ball_possession"] and row["team_id"] != "ball":
            player_x, player_y = row["player_x"], row["player_y"]
            circle = plt.Circle(
                (player_x, player_y),
                color=TEAM_COLORS[row["team_id"]],
                alpha=0.3,
                fill=True,
            )
            ax.add_patch(circle)
    return fig, ax


def plot_optimal_positions(fig, ax, match, frame):
    match_def = match.copy()
    match_def.tracking_data = match_def.tracking_data[
        match_def.tracking_data["frame"] == frame
    ]
    idx_databallpy = match_def.tracking_data.index[
        match_def.tracking_data["frame"] == frame
    ].tolist()[0]
    off_team = match_def.tracking_data.loc[idx_databallpy]["ball_possession"]
    for col in match_def.tracking_data.columns.tolist():
        if off_team in col:
            match_def.tracking_data.loc[idx_databallpy, col] = None
    fig, ax = plot_tracking_data(
        match_def,
        idx_databallpy,
        fig=fig,
        ax=ax,
        team_colors=["lightblue", "pink"],
    )
    return fig, ax
