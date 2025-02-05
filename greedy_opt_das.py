import numpy as np
import streamlit as st
from scipy.ndimage import gaussian_filter

# Parameter Optimierung
MAX_SPEED = 3
MIN_DISTANCE = 2
ITERATIONS = 10
MAX_MOVE_RADIUS = 5


def extract_defender_positions(
    df, id_col="player_id", x_col="player_x", y_col="player_y"
):
    return df[[id_col, x_col, y_col]].values


def find_highest_das_spot(das_map, defender_pos, search_radius=5):
    x, y = int(defender_pos[1]), int(defender_pos[2])
    x_min, x_max = max(0, x - search_radius), min(das_map.shape[0], x + search_radius)
    y_min, y_max = max(0, y - search_radius), min(das_map.shape[1], y + search_radius)

    subregion = das_map[x_min:x_max, y_min:y_max]
    if subregion.size == 0:
        return np.array([x, y])
    max_idx = np.unravel_index(np.argmax(subregion), subregion.shape)
    return np.array([defender_pos[0], x_min + max_idx[0], y_min + max_idx[1]])


def optimize_defensive_positions(das_map, defenders):
    original_positions = np.copy(defenders)
    for _ in range(ITERATIONS):
        new_positions = []
        for defender in defenders:
            best_target = find_highest_das_spot(das_map, defender)

            defender_id = defender[0]
            defender_pos = defender[1:].astype(float)
            best_target_pos = best_target[1:].astype(float)

            direction = best_target_pos - defender_pos
            distance = np.linalg.norm(direction)
            if distance > MAX_SPEED:
                direction = direction / distance * MAX_SPEED

            new_pos = np.array([defender_id, *(defender_pos + direction)], dtype=object)

            total_movement = np.linalg.norm(
                new_pos[1:] - original_positions[defenders[:, 0] == defender[0], 1:][0]
            )
            if total_movement > MAX_MOVE_RADIUS:
                direction = direction / total_movement * MAX_MOVE_RADIUS
                new_pos[1:] = (
                    original_positions[defenders[:, 0] == defender[0], 1:][0]
                    + direction
                )
            new_positions.append(new_pos)

        for i, pos in enumerate(new_positions):
            for j in range(i + 1, len(new_positions)):
                if np.linalg.norm(pos[1:] - new_positions[j][1:]) < MIN_DISTANCE:
                    new_positions[j][1:] += (
                        np.random.uniform(-1, 1, size=2) * MIN_DISTANCE
                    )

        defenders = np.array(new_positions)

    return defenders
