import os
import pandas as pd
import numpy as np
from databallpy import get_saved_game
from databallpy.features.player_possession import (
    get_distance_between_ball_and_players,
    get_initial_possessions,
    get_start_end_idxs,
    get_valid_gains,
    get_ball_losses_and_updated_gain_idxs,
)

pd.set_option("display.max_rows", None)


def possession_spread_report(min_run_len=10):
    games = os.listdir("datasets/IDSSE")
    for game_id in games:
        if game_id != "J03WMX":
            continue
        game = get_saved_game(name=game_id, path="datasets/IDSSE")
        game.tracking_data.add_velocity(
            column_ids="ball",
            max_velocity=50.0,
        )
        game.synchronise_tracking_and_event_data()

        game.tracking_data.add_individual_player_possession()
        print(game.tracking_data[game.tracking_data["frame"] == 29802])
        print(game.tracking_data["player_possession"].value_counts().head())
        td = game.tracking_data
        mask_individual = td["player_possession"].notna()
        minute = td["gametime_td"].apply(parse_minute).astype(int)
        print("Anteil mit Besitz gesamt:", mask_individual.mean())
        print("Anteil mit Besitz pro Minute:")
        print(mask_individual.groupby(minute).mean())
        break
        game.tracking_data.add_individual_player_possession()
        td = game.tracking_data
        # td.to_csv(f"quality/td_{game_id}.csv", index=False)
        fr = td.frame_rate
        mask_ball_status = td["ball_status"] == "alive"
        mask_team_possession = td["team_possession"].notna()
        mask_individual = td["player_possession"].notna()

        minute = td["gametime_td"].apply(parse_minute).astype(int)

        print("Anteil mit Besitz gesamt:", mask_individual.mean())
        print("Anteil mit Besitz pro Minute:")
        print(mask_individual.groupby(minute).mean())
        # per_min_counts = mask_individual.groupby(minute).sum()
        # per_min_total = pd.Series(1, index=minute.index).groupby(minute).sum()
        # per_min_rate = (per_min_counts / per_min_total).fillna(0.0)

        distances_df = get_distance_between_ball_and_players(td)
        initial_possession = get_initial_possessions(1.5, distances_df)

        starts, ends = get_start_end_idxs(initial_possession)

        valid = get_valid_gains(td, starts, ends, 5.0, 10.0, 0)
        gain_starts, losses = get_ball_losses_and_updated_gain_idxs(
            starts, ends, valid, initial_possession
        )
        print("gain_starts[:20]", gain_starts[:20])
        print("losses[:20]", losses[:20])
        print("Letzte paar:", gain_starts[-10:], losses[-10:])

        print("Anzahl Läufe:", len(starts))
        print("Valid gains gesamt:", int(valid.sum()))
        gain_minute = pd.Series(valid).groupby(minute.iloc[starts].values).sum()
        print(gain_minute)

        pos_raw = np.full(len(td), None, dtype=object)
        for s, e in zip(gain_starts, losses):
            pos_raw[s:e] = initial_possession[s]

        assigned_raw = pd.Series(pos_raw, index=td.index).notna()
        alive_mask = td["ball_status"].astype(str).str.strip().str.lower().eq("alive")
        print("Roh (ohne alive) pro Minute:")
        print(assigned_raw.groupby(minute).mean())

        print("Roh & alive pro Minute:")
        print((assigned_raw & alive_mask).groupby(minute).mean())

        print("Final (aus DF) pro Minute:")
        print(td["player_possession"].notna().groupby(minute).mean())

        lengths = losses - gain_starts
        print("Median Lauf-Länge (Frames):", np.median(lengths))
        print("Läufe mit Länge <= 1:", int((lengths <= 1).sum()))

        starts_valid = pd.Series(
            [initial_possession[s] is not None for s in gain_starts]
        )
        print("Anteil gültiger initial_possession an Starts:", starts_valid.mean())

        overlap = (assigned_raw & alive_mask).sum() / max(1, assigned_raw.sum())
        print("Overlap Roh-Zuweisung mit alive:", overlap)

        # print(per_min_total)

        # print(per_min_rate)
        break
        alive = td["ball_status"] == "alive"
        print("Anteil alive gesamt:", alive.mean())
        print("Anteil alive pro Minute:")
        print(alive.groupby(minute).mean())

        ball_ok = td[["ball_x", "ball_y"]].notna().all(axis=1)
        print("Ballpos ok pro Minute:")
        print(ball_ok.groupby(minute).mean())

        x_cols = [c for c in td.columns if c.endswith("_x") and c != "ball_x"]
        y_cols = [c[:-2] + "_y" for c in x_cols]

        def min_player_dist(row):
            bx, by = row["ball_x"], row["ball_y"]
            xs = row[x_cols].values
            ys = row[y_cols].values
            d = np.sqrt((xs - bx) ** 2 + (ys - by) ** 2)
            return np.nanmin(d) if np.isfinite(d).any() else np.nan

        min_d = td[["ball_x", "ball_y"] + x_cols + y_cols].apply(
            min_player_dist, axis=1
        )
        print("Share(min_dist <=1.5m) pro Minute:")
        print((min_d <= 1.5).groupby(minute).mean())

        if "ball_velocity" in td.columns:
            print("Share(ball_velocity <= 5.0 m/s) pro Minute:")
            print((td["ball_velocity"] <= 5.0).groupby(minute).mean())
        else:
            print("WARN: 'ball_velocity' fehlt - add_velocity() vorher ausführen")
        break


def parse_minute(s):
    if s == "Break":
        return -1
    elif "+" in s:
        main, extra = s.split("+", 1)
        mm, ss = map(int, main.split(":"))
        ex_mm, ex_ss = map(int, extra.split(":"))
        return int(mm + ex_mm + (1 if ex_ss > 0 else 0))
    else:
        mm, ss = map(int, s.split(":"))
        return int(mm + (1 if ss > 0 else 0))
