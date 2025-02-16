METRICA_MAPPING = {"FIFATMA": "home", "FIFATMB": "away"}
TEAM_COLORS = {"home": "blue", "away": "red"}
MAX_OPT_DISTANCE = 5
POSS_TO_DEF = {"home": "away", "away": "home"}


def get_metrica_mapping():
    return METRICA_MAPPING


def get_team_colors():
    return TEAM_COLORS


def get_max_opt_distance():
    return MAX_OPT_DISTANCE


def get_poss_to_def():
    return POSS_TO_DEF
