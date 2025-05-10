import pytest
import numpy as np
import packaging.version


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    # get numpy version
    np_version = np.__version__

    # only use legacy printing with 1.22 or higher
    use_legacy = packaging.version.Version(np_version) >= packaging.version.Version("1.22")
    # raise ValueError(f"{np_version}, {use_legacy}, {packaging.version.Version(np_version)}, {packaging.version.Version('1.22')}")
    if use_legacy:
        try:
            np.set_printoptions(legacy="1.21")  # Uniform numpy printing for doctests
        except UserWarning:
            pass
