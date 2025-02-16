import streamlit as st
import numpy as np
import prep_das_def
import resources
import matplotlib.pyplot as plt
from databallpy.visualize import plot_soccer_pitch, plot_tracking_data
import accessible_space


@st.cache_data
def get_recheable_area(df, idx, frame, fps, kick_off_frame):
    st.write(f"Index: {idx}, Frame: {frame}, FPS: {fps}, Kick-Off: {kick_off_frame}")
    if frame == kick_off_frame:
        df_before = df[df["frame"] == frame]
    else:
        df_before = df[df["frame"] == frame - fps]
    return df_before


@st.cache_data
def get_highest_dangerous_values(
    pitch_result, df_pre_frame, df_tracking, frame, das_frame_index, match
):
    st.write("# Optimierung Verteidigerpositionen")

    # Verteidiger Filtern
    def_team = resources.get_poss_to_def()[
        df_tracking[df_tracking["frame"] == frame]["ball_possession"].iloc[0]
    ]
    df_verteidiger = (
        df_tracking[
            (df_tracking["frame"] == frame) & (df_tracking["team_id"] == def_team)
        ]
        .dropna(subset=["player_x", "player_y"])
        .reset_index(drop=True)
    )

    # Index für Match-Objekt bestimmen
    idx_match = match.tracking_data.index[
        match.tracking_data["frame"] == frame
    ].tolist()[0]

    # Variablen vereinfachen / definieren
    dangerous_result, frame_index = (
        pitch_result.dangerous_result,
        pitch_result.frame_index,
    )
    idx_das = frame_index[das_frame_index]
    x_grid, y_grid = dangerous_result.x_grid[idx_das], dangerous_result.y_grid[idx_das]
    attack_poss_density = dangerous_result.attack_poss_density[idx_das]
    df_tracking_oneframe = df_tracking[df_tracking["frame"] == frame]
    das_start = pitch_result.das.loc[das_frame_index]
    match_optimized = match.copy()
    pitch_result_optimized = pitch_result

    for i, row in df_verteidiger.iterrows():
        df_tracking_temp = df_tracking_oneframe.copy()
        player_id = row["player_id"]
        x_center, y_center = df_pre_frame.loc[
            df_pre_frame["player_id"] == player_id, ["player_x", "player_y"]
        ].values[0]

        distances = np.sqrt((x_grid - x_center) ** 2 + (y_grid - y_center) ** 2)
        mask = distances <= resources.get_max_opt_distance()
        max_value = np.max(attack_poss_density[mask])
        max_indices = np.where((attack_poss_density == max_value) & mask)
        max_x, max_y = x_grid[max_indices][0], y_grid[max_indices][0]

        df_tracking_temp.loc[
            df_tracking_temp["player_id"] == player_id, ["player_x", "player_y"]
        ] = (max_x, max_y)

        pitch_result_temp = prep_das_def.get_dangerous_accessible_space(
            df_tracking_temp
        )
        das_temp = pitch_result_temp.das.iloc[0]

        if das_temp < das_start:
            df_tracking_oneframe = df_tracking_temp
            das_start = das_temp
            match_optimized.tracking_data.loc[
                match_optimized.tracking_data["frame"] == frame,
                [f"{player_id}_x", f"{player_id}_y"],
            ] = (max_x, max_y)
            pitch_result_optimized = pitch_result_temp
        else:
            pass

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_tracking_oneframe)
    with col2:
        st.dataframe(
            match_optimized.tracking_data.loc[
                match_optimized.tracking_data["frame"] == frame
            ]
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    fig, ax = plot_soccer_pitch(
        fig=fig,
        ax=ax,
        field_dimen=match_optimized.pitch_dimensions,
        pitch_color="white",
    )
    fig, ax = plot_tracking_data(
        match_optimized,
        idx_match,
        fig=fig,
        ax=ax,
        team_colors=["blue", "red"],
        variable_of_interest=round(float(pitch_result_optimized.das.iloc[idx_das]), 2),
        add_player_possession=True,
    )
    try:
        accessible_space.plot_expected_completion_surface(
            pitch_result_optimized.dangerous_result, frame_index=idx_das
        )
    except:
        st.warning(f"Fehler beim Plotten von DAS")
    st.pyplot(fig)


# @st.cache_data
def get_highest_dangerous_values_manually(
    pitch_result, df_pre_frame, df_tracking, frame, das_frame_index, match
):
    st.write("### Optimierung Verteidigungsposition")
    st.write(
        f"Angreifende Mannschaft: {df_tracking[df_tracking['frame'] == frame]['ball_possession'].iloc[0]}, Frame: {frame}, das_frame_index: {das_frame_index}"
    )
    # Verteidiger filtern
    def_team = resources.get_poss_to_def()[
        df_tracking[df_tracking["frame"] == frame]["ball_possession"].iloc[0]
    ]
    df_opt = (
        df_tracking[
            (df_tracking["frame"] == frame) & (df_tracking["team_id"] == def_team)
        ]
        .dropna(subset=["player_x", "player_y"])
        .reset_index(drop=True)
    )
    idx_match = match.tracking_data.index[
        match.tracking_data["frame"] == frame
    ].tolist()[0]
    st.dataframe(match.tracking_data.iloc[idx_match])

    # Variablen definieren
    dangerous_result, frame_index = (
        pitch_result.dangerous_result,
        pitch_result.frame_index,
    )
    idx = frame_index[das_frame_index]
    x_grid = dangerous_result.x_grid[idx]
    y_grid = dangerous_result.y_grid[idx]
    attack_poss_density = dangerous_result.attack_poss_density[idx]
    df_tracking_try = df_tracking[df_tracking["frame"] == frame]
    pitch_result_try = prep_das_def.get_dangerous_accessible_space(df_tracking_try)
    das_org = pitch_result.das.loc[das_frame_index]

    # Manuelle Iteration
    if "das_low" not in st.session_state:
        st.session_state.das_low = das_org
    if "iteration_index" not in st.session_state:
        st.session_state.iteration_index = 0
    if "df_tracking_try" not in st.session_state:
        st.session_state.df_tracking_try = df_tracking_try.copy()
    if "match_try" not in st.session_state:
        st.session_state.match_try = match.copy()

    st.write(f"DAS Value in Frame: {das_org}")
    st.write(
        f"Aktuelle Interation: {st.session_state.iteration_index + 1} von {len(df_opt)}"
    )

    if st.session_state.iteration_index < len(df_opt):
        match_try = st.session_state.match_try.copy()
        df_tracking_try = st.session_state.df_tracking_try.copy()

        row = df_opt.iloc[st.session_state.iteration_index]
        player_id = row["player_id"]

        x_center, y_center = df_pre_frame.loc[
            df_pre_frame["player_id"] == player_id, ["player_x", "player_y"]
        ].values[0]

        distances = np.sqrt((x_grid - x_center) ** 2 + (y_grid - y_center) ** 2)
        mask = distances <= resources.get_max_opt_distance()
        max_value = np.max(attack_poss_density[mask])
        max_indices = np.where((attack_poss_density == max_value) & mask)
        max_x = x_grid[max_indices][0]
        max_y = y_grid[max_indices][0]

        df_tracking_try.loc[
            df_tracking_try["player_id"] == player_id, ["player_x", "player_y"]
        ] = (max_x, max_y)

        match_try.tracking_data.loc[
            match_try.tracking_data["frame"] == frame,
            [f"{player_id}_x", f"{player_id}_y"],
        ] = (max_x, max_y)

        pitch_result_try = prep_das_def.get_dangerous_accessible_space(df_tracking_try)

        st.write(
            f"### Try Frame in Iteration {st.session_state.iteration_index + 1}: {pitch_result_try.das.iloc[0]}"
        )
        st.write(f"Player to move: {player_id}")

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Match Try")
            st.dataframe(match_try.tracking_data.iloc[idx_match])
        with col2:
            st.write("### DF Try")
            st.dataframe(df_tracking_try)

        # Pitch after Iteration
        das_try = float(pitch_result_try.das.iloc[0])
        fig, ax = plt.subplots(figsize=(10, 6))
        fig, ax = plot_soccer_pitch(
            fig=fig, ax=ax, field_dimen=match_try.pitch_dimensions, pitch_color="white"
        )
        fig, ax = plot_tracking_data(
            match_try,
            idx_match,
            fig=fig,
            ax=ax,
            team_colors=["blue", "red"],
            variable_of_interest=round(das_try, 2),
            add_player_possession=True,
        )
        st.pyplot(fig)

        st.write(f"DAS Try: {das_try}, DAS low: {st.session_state.das_low}")
        if das_try >= st.session_state.das_low:
            match_try = st.session_state.match_try.copy()
            df_tracking_try = st.session_state.df_tracking_try.copy()

        if st.button("Nächste Iteration"):
            st.session_state.das_low = min(st.session_state.das_low, das_try)
            st.session_state.df_tracking_try = df_tracking_try
            st.session_state.match_try = match_try

            st.session_state.iteration_index += 1
            st.rerun()
        if st.button("Reset Iteration"):
            st.session_state.df_tracking_try = df_tracking[
                df_tracking["frame"] == frame
            ].copy()
            st.session_state.das_low = das_org
            st.session_state.match_try = match.copy()
            st.session_state.iteration_index = 0
            st.rerun()

    else:
        st.write("Alle Iterationen abgeschlossen.")
        if st.button("Reset Iteration"):
            st.session_state.df_tracking_try = df_tracking[
                df_tracking["frame"] == frame
            ].copy()
            st.session_state.das_low = das_org
            st.session_state.match_try = match.copy()
            st.session_state.iteration_index = 0
            st.rerun()
