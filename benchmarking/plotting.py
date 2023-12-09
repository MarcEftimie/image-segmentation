"""
Plotting Timing Data
"""
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import data as data_image


PATH = "test_data/astronaut/run_256"
time_data = []

json_files = glob.glob(os.path.join(PATH, "*.json"))

for f in json_files:
    data = json.load(open(f))
    data["n-cut_t"] = np.round(np.mean(data["n-cut_t"]), 2)
    data["k-means_t"] = np.round(np.mean(data["k-means_t"]), 2)
    data["mean-shift_t"] = np.round(np.mean(data["mean-shift_t"]), 2)
    time_data.append(data)

time_data = pd.DataFrame.from_dict(time_data)
time_data = time_data.sort_values(by=["segments"])
print(time_data)
