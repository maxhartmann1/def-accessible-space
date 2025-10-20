from databallpy.utils.constants import OPEN_GAME_IDS_DFL
from databallpy import get_saved_game, get_open_game
from pathlib import Path
from optimizer import optimize_player_position
from time import time
import pandas as pd
import numpy as np
import os
import accessible_space
import joblib
import logging
import sys

from importlib.metadata import version, PackageNotFoundError

logging.basicConfig(
    filename="berechnung.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def calculate(args):

    # Preperation
    game_id = choose_game(args.game_id)
    game = load_game(game_id)
    frame_step_size = select_frame_step_size(game, args.step_size)
    df_tracking_filtered = filter_tracking_data(game, frame_step_size)
    df_frameified = frameify_tracking_data(df_tracking_filtered, game)

    # DAS Berechnung
    pitch_result, df_frameified, frame_list = calculate_pitch_result(
        df_frameified, game_id, frame_step_size
    )
    params = {
        "max_radius": args.radius,
        "opt_step_size": args.opt_step_size,
        "min_dist": args.min_dist,
        "n": args.n,
    }

    # Spieler optimieren
    if "none" in (s.casefold() for s in args.player):
        print("DAS nur für Ausgangsframes")
        return
    elif "all" in (s.casefold() for s in args.player):
        players = game.get_column_ids()
    else:
        players = args.player

    if args.cut > 0:
        df_frameified = filter_tracking_data_post_pitch_result(
            df_frameified, frame_list, args.cut
        )
        # print(df_frameified.shape)
        # print(frame_list.shape)
        args.method = [m + f"_cut_{args.cut}" for m in args.method]

    calculate_player_optization(
        df_frameified,
        players,
        game,
        game_id,
        frame_list,
        args.method,
        frame_step_size,
        params,
        args.cut,
    )


def choose_game(game_id):
    games = OPEN_GAME_IDS_DFL
    games["metrica"] = "Metrica Anonymisiertes Spiel"

    # Entwicklung
    if game_id in games.keys():
        return game_id

    print("Spiel auswählen")
    for key, value in games.items():
        print(f"{key}: {value}")
    while True:
        game_id = input("Spiel-ID auswählen: ").strip()
        if game_id in games:
            break
        else:
            print("Spiel nicht vorhanden.")
    return game_id


def load_game(game_id):
    if game_id == "metrica":
        path = "datasets/"
    else:
        path = "datasets/IDSSE/"

    if "notebooks" in str(Path.cwd()):
        path = "../" + path

    if os.path.exists(path + game_id):
        game = get_saved_game(name=game_id, path=path)
    else:
        if game_id == "metrica":
            game = get_open_game(game_id)
        else:
            game = get_open_game("dfl", game_id)

    col_ids = game.get_column_ids() + ["ball"]
    game.tracking_data.add_velocity(column_ids=col_ids, max_velocity=50)
    game.synchronise_tracking_and_event_data()
    if game_id == "metrica":
        game.tracking_data.add_team_possession(game.event_data, game.home_team_id)
    game.tracking_data.add_individual_player_possession()
    return game


def select_frame_step_size(game, step_size):
    if step_size:
        return step_size
    else:
        return game.tracking_data.frame_rate


def filter_tracking_data(game, frame_step_size):
    df = pd.DataFrame(game.tracking_data.copy())
    df_tracking_filtered = df[df["player_possession"].notna()]
    df_tracking_filtered = df_tracking_filtered[
        game.tracking_data.frame_rate :: frame_step_size
    ]
    return df_tracking_filtered


def filter_tracking_data_post_pitch_result(df, frame_list, cut):
    # print(df.shape)
    df = df[df["DAS"] > cut]
    # print(df.shape)
    return df


def frameify_tracking_data(df_tracking_filtered, game):
    coordinate_cols = []
    player_to_team = {}
    players = game.get_column_ids()
    players.append("ball")
    for player in players:
        coordinate_cols.append(
            [f"{player}_x", f"{player}_y", f"{player}_vx", f"{player}_vy"]
        )
        player_to_team[str(player)] = player.split("_")[0]

    df_tracking = accessible_space.per_object_frameify_tracking_data(
        df_tracking_filtered,
        "frame",
        coordinate_cols,
        players,
        player_to_team,
        new_coordinate_cols=("player_x", "player_y", "player_vx", "player_vy"),
    )
    df_tracking = df_tracking[df_tracking["player_x"].notna()]
    return df_tracking


def calculate_pitch_result(df_frameified, game_id, frame_step_size):
    pitch_result_path = (
        Path("cache_new") / f"pitch_results/{game_id}_step{frame_step_size}.pkl"
    )
    pitch_result_path.parent.mkdir(parents=True, exist_ok=True)
    if pitch_result_path.exists():
        pitch_result = joblib.load(pitch_result_path)
        print("Pitch Result geladen aus Cache.")
    else:
        start_time = time()
        pitch_result = accessible_space.get_dangerous_accessible_space(
            df_frameified,
            frame_col="frame",
            player_col="player_id",
            ball_player_id="ball",
            x_col="player_x",
            y_col="player_y",
            vx_col="player_vx",
            vy_col="player_vy",
            team_col="team_id",
            period_col="period_id",
            team_in_possession_col="team_possession",
            attacking_direction_col=None,
            infer_attacking_direction=True,
            respect_offside=True,
            player_in_possession_col="player_possession",
            use_progress_bar=True,
        )
        duration = time() - start_time
        joblib.dump(pitch_result, pitch_result_path)
        logging.info(
            f"Pitch Result: step size: {frame_step_size} | game: {game_id} | zeit: {duration:.4f} | anzahl frame: {df_frameified["frame"].nunique()}"
        )
        print("Pitch Result berechnet.")

    df_frameified["AS"] = pitch_result.acc_space
    df_frameified["DAS"] = pitch_result.das
    frame_list = df_frameified["frame"].unique()
    frame_list_path = (
        Path("cache_new")
        / f"pitch_results/frame_list_{game_id}_step{frame_step_size}.csv"
    )
    frame_list_path.parent.mkdir(parents=True, exist_ok=True)
    if not frame_list_path.exists():
        np.savetxt(frame_list_path, frame_list, delimiter=",", fmt="%.0f")

    return pitch_result, df_frameified, frame_list


def calculate_player_optization(
    df_frameified,
    players,
    game,
    game_id,
    frame_list,
    methods,
    frame_step_size,
    params,
    cut,
):
    df = pd.DataFrame(game.tracking_data.copy())
    df_pre_frames = get_pre_frames(df, game.tracking_data.frame_rate, frame_list)
    param_path = f"R_{params["max_radius"]}-S_{params["opt_step_size"]}-D_{params["min_dist"]}-N_{params["n"]}"
    param_path = param_path + f"-C_{cut}" if cut > 0 else param_path
    for player in players:
        subset = df_frameified[df_frameified["player_id"] == player]
        if subset.empty:
            print("empty")
            continue
        for method in methods:
            simulation_path = get_simulation_path(
                game_id, player, method, frame_step_size
            )
            (
                pitch_result_optimized,
                frame_list_optimized,
                df_frameified_optimized,
                df_frameified_simulated,
            ) = find_optimal_position(
                param_path,
                simulation_path,
                df_pre_frames,
                df_frameified,
                game_id,
                frame_list,
                player,
                params,
                method,
                frame_step_size,
            )
            reduce_path = (
                Path("cache_new")
                / f"optimization/{game_id}/{player}/step{frame_step_size}/{method}/{param_path}"
            )
            df_reduced = reduce_df_optimization(df_frameified_optimized, reduce_path)


def get_pre_frames(df, frame_rate, frame_list):
    pre_frame_list = frame_list - frame_rate
    df_pre_frames = df[df["frame"].isin(pre_frame_list)]
    return df_pre_frames


def get_simulation_path(game_id, player, method, frame_step_size):
    simulation_path = (
        Path("cache_new")
        / f"optimization/{game_id}/{player}/step{frame_step_size}/{method}"
    )
    simulation_path.mkdir(parents=True, exist_ok=True)
    return simulation_path


def find_optimal_position(
    param_path,
    simulation_path,
    df_pre_frames,
    df_frameified,
    game_id,
    frame_list,
    player,
    params,
    method,
    frame_step_size,
):
    df_optimized_path = simulation_path / param_path / f"df_optimized.csv"
    df_simulated_path = simulation_path / param_path / f"df_simulated.csv"
    pitch_result_path = simulation_path / param_path / f"pitch_result.pkl"
    frame_list_path = simulation_path / param_path / f"frame_list.csv"

    if (
        df_optimized_path.exists()
        and pitch_result_path.exists()
        and frame_list_path.exists()
        and df_simulated_path.exists()
    ):
        df_frameified_optimized = pd.read_csv(df_optimized_path)
        df_frameified_simulated = pd.read_csv(
            df_simulated_path, dtype={"databallpy_event": str}
        )
        pitch_result_optimized = joblib.load(pitch_result_path)
        frame_list_optimized = np.loadtxt(frame_list_path, delimiter=",", dtype=str)

    elif method == "grid_search":
        print("grid_search")
        sys.exit(0)

    else:
        (
            df_frameified_simulated,
            df_frameified_optimized,
            pitch_result_optimized,
            frame_list_optimized,
        ) = optimize_player_position(
            df_frameified,
            player,
            frame_list,
            params,
            df_pre_frames,
            method,
            game_id,
            frame_step_size,
        )
        frame_list_path.parent.mkdir(parents=True, exist_ok=True)
        df_frameified_simulated.to_csv(df_simulated_path, index=False)
        df_frameified_optimized.to_csv(df_optimized_path, index=False)
        joblib.dump(pitch_result_optimized, pitch_result_path)
        np.savetxt(frame_list_path, frame_list_optimized, delimiter=",", fmt="%s")
    return (
        pitch_result_optimized,
        frame_list_optimized,
        df_frameified_optimized,
        df_frameified_simulated,
    )


def reduce_df_optimization(df_frameified_optimized, reduce_path):
    reduce_path.mkdir(parents=True, exist_ok=True)
    reduce_path = reduce_path / "df_reduced.csv"
    if reduce_path.exists():
        df_reduced = pd.read_csv(reduce_path)
    else:
        df_reduced = df_frameified_optimized[
            df_frameified_optimized["opt_player"]
            == df_frameified_optimized["player_id"]
        ][["player_id", "frame", "DAS", "DAS_new", "new_frame"]]
    df_reduced["DAS_potential"] = (df_reduced["DAS"] - df_reduced["DAS_new"]).clip(
        lower=0
    )
    split_cols = df_reduced["new_frame"].str.split("_", expand=True)
    df_reduced["move_x"] = split_cols.iloc[:, -2].astype(float)
    df_reduced["move_y"] = split_cols.iloc[:, -1].astype(float)
    df_reduced["distance"] = np.sqrt(
        np.square(df_reduced["move_x"]) + np.square(df_reduced["move_y"])
    )
    df_reduced.to_csv(reduce_path, index=False)
    return df_reduced
