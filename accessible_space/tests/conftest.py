import pytest
import numpy as np
import packaging.version
import matplotlib


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    matplotlib.use("Agg")

    use_legacy = packaging.version.Version(np.__version__) >= packaging.version.Version("1.22")
    if use_legacy:
        try:
            np.set_printoptions(legacy="1.21")  # Uniform numpy printing for doctests
        except UserWarning:
            pass
