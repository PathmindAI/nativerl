import os

import numpy as np

if os.environ.get("USE_PY_NATIVERL"):
    import pathmind_training.pynativerl as nativerl
else:
    import nativerl


def test_nativerl_arrays():
    np_arr = np.array([2.0, 3.0, 3.0, 4.0], dtype=np.float32)
    arr = nativerl.Array(np_arr)

    assert len(arr) == 4

    term_contributions_dict: dict = {}

    flat_arr = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

    for i in range(0, 2):
        term_contributions_dict[str(i)] = flat_arr

    max_array = np.zeros(len(arr), dtype=np.float32)
    for values in term_contributions_dict.values():
        max_array = np.array(
            [max(max_array[i], abs(values[i])) for i in range(len(arr))]
        )
    term_contributions = nativerl.Array(max_array)
    if hasattr(nativerl, "FloatVector"):
        # only in c++ version
        assert term_contributions.values() == nativerl.FloatVector([2.0, 4.0, 6.0, 8.0])
