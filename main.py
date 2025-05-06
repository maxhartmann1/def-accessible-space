import importlib

import accessible_space.tests.test_model

import accessible_space.core
import accessible_space.interface
importlib.reload(accessible_space.core)
importlib.reload(accessible_space.interface)

from accessible_space.tests.test_model import _get_butterfly_data

if __name__ == '__main__':
    accessible_space.tests.test_model.test_additional_defender_decreases_as_and_additional_attacker_increases_as(_get_butterfly_data)
