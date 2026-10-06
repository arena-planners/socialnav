from __future__ import annotations

import numpy as np
import pytest

from scene import street_frame


@pytest.fixture(scope="session")
def frame() -> np.ndarray:
    return street_frame()
