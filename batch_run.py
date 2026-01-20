import itertools
import subprocess

players = [
    [
        # "home_11",
        # "home_19",
        # "home_23",
        # "home_25",
        # "home_33",
        # "home_5",
        # "home_7",
        # "home_9",
        "all"
    ]
]
game_ids = [
    # "J03WR9",
    "J03WOH",
    "J03WOY",
    "J03WPY",
    "J03WQQ",
    "J03WMX",
    "J03WN1",
]
methods_config = {
    # "grid_search": {"ns": [5], "opt_step_sizes": [1.5]},
    "all_positions": {"ns": [20], "opt_step_sizes": [1]},
    # "random": {"ns": [20, 50], "opt_step_sizes": [1]},
}

cuts = [0.1]


def main():
    for player in players:
        for game in game_ids:
            for method, cfg in methods_config.items():
                for n in cfg["ns"]:
                    for step in cfg["opt_step_sizes"]:
                        for cut in cuts:
                            cmd = [
                                "python3",
                                "app.py",
                                "--player",
                                *player,
                                "--method",
                                method,
                                "--n",
                                str(n),
                                "--opt_step_size",
                                str(step),
                                "--game_id",
                                game,
                            ]
                            # nur anhängen, wenn cut gesetzt ist
                            if cut is not None:
                                cmd += ["--cut", str(cut)]

                            print("Starte:", " ".join(cmd))
                            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
