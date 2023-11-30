"""
Mean Shift Clustering Image Segmentation
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

RADIUS = 5

def find_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) **2)

def mean_shift(image):
    centroids = {}

    # Create a copy of the image
    image_copy = image.copy()

    # Loop over every pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            window = image[i - RADIUS : i + RADIUS, j - RADIUS : j + RADIUS]

            # get valid dist pixels in window
            valid_points = []
            for k in range(window.shape[0]):
                for l in range(window.shape[1]):
                    if find_distance((i, j), (k, l)) <= RADIUS:
                        valid_points.append(window[k, l])
            
            # if len(valid_points) != 0:
            #     print(len(valid_points))
            #     break

            valid_points = np.array(valid_points)

            # Calculate the mean of the valid points
            mean = np.mean(valid_points, axis=0).astype(np.uint8)

            # If the mean is not a centroid
            if (i,j) not in centroids.keys():
                # Add the mean as a centroid
                centroids[(i, j)] = mean
    
    # Loop over every pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Set the pixel to the centroid value
            image_copy[i, j] = centroids[(i, j)]
    
    print(len(set(centroids.values())))
    
    # Display the image
    plt.imshow(image_copy, cmap="gray")

    plt.show()

# Load image
image = cv2.imread("./images/monkey.jpg")
# convert to black and white
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (10, 10))
#  shwo image
plt.imshow(image, cmap="gray")
plt.show()

mean_shift(image)
