"""
Main file to run the image segmentation algorithm.
"""

import cv2
from nx_image_2_graph import buildGraph
from nx_augmenting_path import augmentingPath
from our_nx_augmenting_path import augmentingPath2
from nx_user_input import get_user_input, displayCut, colorPixel, SCALE_FACTOR
from time import time
import networkx as nx
import matplotlib.pyplot as plt

image = cv2.imread("./images/test1.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (30, 30))

SOURCE = image.size     # source node index (index of last pixel + 1 to avoid conflict)
SINK = image.size + 1   # sink node index (index of last pixel + 2 to avoid conflict)

# get user input for source and sink points
source_points, sink_points = get_user_input(image)

# build the graph from the image
graph, source_sink = buildGraph(image, SOURCE, SINK, source_points, sink_points)

time1 = time()
# run the augmenting path algorithm to find the cuts
cuts = augmentingPath(graph, image.size, image.size + 1)

time2 = time()
print("Time taken: ", time2 - time1)

# convert the image to RGB for coloring
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# color the source and sink points red on the image
for i,j in source_sink:
    colorPixel(image, i, j, red=True)

# display the cut on the image
image = displayCut(image, cuts)

# increase the image size for better visualization
image = cv2.resize(image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
cv2.imshow("Segmented image", image)
cv2.waitKey(0)
