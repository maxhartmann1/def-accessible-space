from dataclasses import dataclass


@dataclass(frozen=True)
class PDDConfig:
    frame_filter_rate: int = 125
    max_radius: float = 5.0
    optimization_method: str = "all_positions"
    das_threshold: float = 0.1
    discretization_step: float = 1.0
    min_distance_to_teammates: float = 2.0
    frame_rate: int = 25


@dataclass(frozen=True)
class TrackingColumnSchema:
    x: str = "x"
    y: str = "y"
    vx: str = "vx"
    vy: str = "vy"
    ball_prefix: str = "ball"
    separator: str = "_"
    home_team: str = "home"
    away_team: str = "away"
    frame_column: str = "frame"
    team_column: str = "team_id"
    period_column: str = "period_id"
    team_possession_col: str = "team_possession"
    player_in_possession_col: str = "player_possession"
