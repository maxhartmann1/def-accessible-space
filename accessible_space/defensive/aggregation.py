def compute_player_level_pdd(df_frame_level):
    temp_df = df_frame_level.copy()
    df_player_level = df_frame_level.groupby("player_id").agg(
        SUMME_DAS=("DAS", "sum"),
        SUMME_DAS_NEW=("DAS_new", "sum"),
        NUMBER_OF_FRAMES=("DAS", "count"),
    )
    df_player_level["PDD"] = (
        df_player_level["SUMME_DAS"] - df_player_level["SUMME_DAS_NEW"]
    ) / df_player_level["SUMME_DAS"]
    df_player_level["PDD_Absolute"] = (
        df_player_level["SUMME_DAS"] - df_player_level["SUMME_DAS_NEW"]
    )
    return df_player_level
