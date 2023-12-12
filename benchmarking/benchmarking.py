"""
Measure Execution Time using timeit
"""

import timeit
import json
from skimage import data
from segmentation import n_cut, k_means, mean_shift
import numpy as np


# Wrapper Class for Capturing Algorithm Output
class CaptureReturnValue:
    """
    Wraps Return Value while Timing
    """

    def __init__(self, func):
        self.func = func
        self.return_value = None

    def __call__(self, *args, **kwargs):
        self.return_value = self.func(*args, **kwargs)


# Load Image from SciKit Library
chelsea = data.chelsea()
coffee = data.coffee()
astronaut = data.astronaut()
rocket = data.rocket()

# Set Parameters
seg_list = np.linspace(2, 20, 19, dtype=int).tolist()
bandwith_list = np.linspace(100, 20, 19, dtype=int).tolist()
crv = CaptureReturnValue(mean_shift)

# Set Image
img = astronaut

# Iterate through Parameters to Compute Time
for index, val in enumerate(seg_list):
    exe_time_ncut = timeit.repeat(lambda: n_cut(img, segments=val), repeat=10, number=1)
    exe_time_kmeans = timeit.repeat(
        lambda: k_means(img, clusters=val), repeat=10, number=1
    )
    exe_time_mean_shift = timeit.repeat(
        lambda: crv(img, bandwith=bandwith_list[index]), repeat=10, number=1
    )
    # Store results as a dictionary and save in json format
    result = {
        "run_id": index,
        "segments": val,
        "bandwith": bandwith_list[index],
        "mean-shift-segments": crv.return_value,
        "n-cut_t": exe_time_ncut,
        "k-means_t": exe_time_kmeans,
        "mean-shift_t": exe_time_mean_shift,
    }
    filename = f"test_data/astronaut/timing_run_{index}.json"
    with open(filename, "w") as file_object:
        json.dump(result, file_object)
