# Measure Execution Time using timeit

import timeit
import json
from skimage import data
from testing import n_cut, k_means, mean_shift


class CaptureReturnValue:
    """
    Wraps Return Value while Timing
    """

    def __init__(self, func):
        self.func = func
        self.return_value = None

    def __call__(self, *args, **kwargs):
        self.return_value = self.func(*args, **kwargs)


img = data.coffee()

seg_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
bandwith_list = [100, 90, 80, 70, 60, 50, 40, 30, 20]

crv = CaptureReturnValue(mean_shift)


for index, val in enumerate(seg_list):
    exe_time_ncut = timeit.repeat(lambda: n_cut(img, segments=val), repeat=10, number=1)
    exe_time_kmeans = timeit.repeat(
        lambda: k_means(img, clusters=val), repeat=10, number=1
    )
    exe_time_mean_shift = timeit.repeat(
        lambda: crv(img, bandwith=bandwith_list[index]), repeat=10, number=1
    )
    result = {
        "run_id": index,
        "segments": val,
        "bandwith": bandwith_list[index],
        "mean-shift-segments": crv.return_value,
        "n-cut_t": exe_time_ncut,
        "k-means_t": exe_time_kmeans,
        "mean-shift_t": exe_time_mean_shift,
    }
    filename = f"test_data/timing_run_{index}.json"
    with open(filename, "w") as file_object:
        json.dump(result, file_object)
