import cv2
import matplotlib.pyplot as plt
import numpy as np


def convert_to_rgb_feature_space(image):
    # Convert image to RGB feature space
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot the result in 3D RGB feature space
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    rgb_image = image.reshape(-1, 3)
    rgb_image = np.flip(rgb_image, axis=1)

    # return rgb_image

    # Calculate the color based on the position of the point in RGB space
    colors = rgb_image / 255.0  # Normalize the colors to [0, 1] for matplotlib

    # Scatter plot, using RGB channels as coordinates and the calculated color
    ax.scatter(
        rgb_image[:, 0],  # Red channel
        rgb_image[:, 1],  # Green channel
        rgb_image[:, 2],  # Blue channel
        c=colors,  # Use the normalized RGB colors for each point
        marker=".",
    )

    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    ax.set_title("3D RGB Feature Space")

    plt.show()


def k_mean(feature_space, k):
    feature_max = feature_space.max()
    feature_min = feature_space.min()

    centers = np.array
    print(feature_max, feature_min)


rgb_feature_space = convert_to_rgb_feature_space(cv2.imread("./images/monkey.jpg"))
