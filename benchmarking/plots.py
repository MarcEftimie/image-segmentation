import matplotlib.pyplot as plt
import numpy as np
import json

# ------------------ ASTRONAUT IMAGE ------------------

# Data
data_64 = json.load(open("test_data/astronaut/run_64/timing_run_0.json"))
data_128 = json.load(open("test_data/astronaut/run_128/timing_run_0.json"))
data_256 = json.load(open("test_data/astronaut/run_256/timing_run_0.json"))

# Average times for each algorithm
n_cut_avg = np.mean(data_64["n-cut_t"])
slic_avg = data_64["slic_t"]  # Only one value
k_means_avg = np.mean(data_64["k-means_t"])
mean_shift_avg = np.mean(data_64["mean-shift_t"])

# Average times for 128x128 image
n_cut_avg_128 = np.mean(data_128["n-cut_t"])
slic_avg_128 = data_128["slic_t"]  # Only one value
k_means_avg_128 = np.mean(data_128["k-means_t"])
mean_shift_avg_128 = np.mean(data_128["mean-shift_t"])

# Average times for 256x256 image
n_cut_avg_256 = np.mean(data_256["n-cut_t"])
slic_avg_256 = data_256["slic_t"]  # Only one value
k_means_avg_256 = np.mean(data_256["k-means_t"])
mean_shift_avg_256 = np.mean(data_256["mean-shift_t"])

# Corresponding average times for 64x64 image
avg_times_64_img_1 = [slic_avg, n_cut_avg, k_means_avg, mean_shift_avg]

# Corresponding average times for 128x128 image
avg_times_128_img_1 = [slic_avg_128, n_cut_avg_128, k_means_avg_128, mean_shift_avg_128]

# Corresponding average times for 256x256 image
avg_times_256_img_1 = [slic_avg_256, n_cut_avg_256, k_means_avg_256, mean_shift_avg_256]

# ------------------ CHELSEA IMAGE ------------------

# Data
data_64 = json.load(open("test_data/chelsea/run_64/timing_run_0.json"))
data_128 = json.load(open("test_data/chelsea/run_128/timing_run_0.json"))
data_256 = json.load(open("test_data/chelsea/run_256/timing_run_0.json"))

# Average times for each algorithm
n_cut_avg = np.mean(data_64["n-cut_t"])
slic_avg = data_64["slic_t"]  # Only one value
k_means_avg = np.mean(data_64["k-means_t"])
mean_shift_avg = np.mean(data_64["mean-shift_t"])

# Average times for 128x128 image
n_cut_avg_128 = np.mean(data_128["n-cut_t"])
slic_avg_128 = data_128["slic_t"]  # Only one value
k_means_avg_128 = np.mean(data_128["k-means_t"])
mean_shift_avg_128 = np.mean(data_128["mean-shift_t"])

# Average times for 256x256 image
n_cut_avg_256 = np.mean(data_256["n-cut_t"])
slic_avg_256 = data_256["slic_t"]  # Only one value
k_means_avg_256 = np.mean(data_256["k-means_t"])
mean_shift_avg_256 = np.mean(data_256["mean-shift_t"])

# Corresponding average times for 64x64 image
avg_times_64_img_2 = [slic_avg, n_cut_avg, k_means_avg, mean_shift_avg]

# Corresponding average times for 128x128 image
avg_times_128_img_2 = [slic_avg_128, n_cut_avg_128, k_means_avg_128, mean_shift_avg_128]

# Corresponding average times for 256x256 image
avg_times_256_img_2 = [slic_avg_256, n_cut_avg_256, k_means_avg_256, mean_shift_avg_256]

# ------------------ COFFEE IMAGE ------------------

# Data
data_64 = json.load(open("test_data/coffee/run_64/timing_run_0.json"))
data_128 = json.load(open("test_data/coffee/run_128/timing_run_0.json"))
data_256 = json.load(open("test_data/coffee/run_256/timing_run_0.json"))

# Average times for each algorithm
n_cut_avg = np.mean(data_64["n-cut_t"])
slic_avg = data_64["slic_t"]  # Only one value
k_means_avg = np.mean(data_64["k-means_t"])
mean_shift_avg = np.mean(data_64["mean-shift_t"])

# Average times for 128x128 image
n_cut_avg_128 = np.mean(data_128["n-cut_t"])
slic_avg_128 = data_128["slic_t"]  # Only one value
k_means_avg_128 = np.mean(data_128["k-means_t"])
mean_shift_avg_128 = np.mean(data_128["mean-shift_t"])

# Average times for 256x256 image
n_cut_avg_256 = np.mean(data_256["n-cut_t"])
slic_avg_256 = data_256["slic_t"]  # Only one value
k_means_avg_256 = np.mean(data_256["k-means_t"])
mean_shift_avg_256 = np.mean(data_256["mean-shift_t"])

# Corresponding average times for 64x64 image
avg_times_64_img_3 = [slic_avg, n_cut_avg, k_means_avg, mean_shift_avg]

# Corresponding average times for 128x128 image
avg_times_128_img_3 = [slic_avg_128, n_cut_avg_128, k_means_avg_128, mean_shift_avg_128]

# Corresponding average times for 256x256 image
avg_times_256_img_3 = [slic_avg_256, n_cut_avg_256, k_means_avg_256, mean_shift_avg_256]

# ------------------ PLOTTING ------------------

# Algorithm names
algorithms = ["SLIC", "N-Cut", "K-Means", "Mean-Shift"]

plt.figure(figsize=(10, 6))

# Astronaut Image
plt.scatter(
    algorithms, avg_times_64_img_1, color="blue", marker="o", label="64x64 Astronaut"
)
plt.scatter(
    algorithms, avg_times_128_img_1, color="red", marker="o", label="128x128 Astronaut"
)
plt.scatter(
    algorithms,
    avg_times_256_img_1,
    color="green",
    marker="o",
    label="256x256 Astronaut",
)

# Chelsea Image
plt.scatter(
    algorithms, avg_times_64_img_2, color="blue", marker="x", label="64x64 Chelsea"
)
plt.scatter(
    algorithms, avg_times_128_img_2, color="red", marker="x", label="128x128 Chelsea"
)
plt.scatter(
    algorithms, avg_times_256_img_2, color="green", marker="x", label="256x256 Chelsea"
)

# Coffee Image
plt.scatter(
    algorithms, avg_times_64_img_2, color="blue", marker="^", label="64x64 Coffee"
)
plt.scatter(
    algorithms, avg_times_128_img_2, color="red", marker="^", label="128x128 Coffee"
)
plt.scatter(
    algorithms, avg_times_256_img_2, color="green", marker="^", label="256x256 Coffee"
)

plt.xlabel("Segmentation Algorithms")
plt.ylabel("Average Time (seconds)")
plt.title("Average Segmentation Time by Algorithm for Different Images and Resolutions")
plt.legend()
plt.show()
