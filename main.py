import importlib

import accessible_space.tests.test_model

import accessible_space.core
import accessible_space.interface
importlib.reload(accessible_space.core)
importlib.reload(accessible_space.interface)

from accessible_space.tests.test_model import _get_butterfly_data
import accessible_space.tests.test_real_world_data

if __name__ == '__main__':
    import streamlit as st

    # print all members of accessible_space.interface
    import accessible_space.interface
    st.write(dir(accessible_space.interface))

    accessible_space.tests.test_real_world_data.test_real_world_data(1)
