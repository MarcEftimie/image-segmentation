import numpy as np
import cv2
import networkx as nx
from nx_augmenting_path import augmentingPath
from our_nx_augmenting_path import augmentingPath2
from math import exp, pow
from nx_user_input import get_user_input, SCALE_FACTOR

def boundaryPenalty(p1, p2):
    return 100 * exp(-pow(int(p1) - int(p2), 2) / (2 * pow(30, 2)))


def connectPixels(graph, image):
    max_capacity = -float("inf")
    image_height, image_width = image.shape
    for row in range(image_height):
        for col in range(image_width):
            x = row * image_width + col

            # Don't add bottom edges for pixels on the bottom edge
            if row + 1 < image_height:
                y = (row + 1) * image_width + col
                bp = boundaryPenalty(image[row][col], image[row + 1][col])
                if int(bp) > 0:
                    graph.add_edge(x, y, capacity=int(bp))
                    graph.add_edge(y, x, capacity=int(bp))
                max_capacity = max(max_capacity, bp)

            # Don't add right edges for pixels on the right edge
            if col + 1 < image_width:
                y = row * image_width + col + 1
                bp = boundaryPenalty(image[row][col], image[row][col + 1])
                if int(bp) > 0:
                    graph.add_edge(x, y, capacity=int(bp))
                    graph.add_edge(y, x, capacity=int(bp))
                max_capacity = max(max_capacity, bp)

    return int(max_capacity)


def connectSourceAndSink(graph, source_pixel, sink_pixel, source_points, sink_points, max_capacity):
    width, height = image.shape
    SOURCE = image.size
    SINK = image.size + 1
    lst = []
    for point in source_points:
        point_idx = point[0] * width + point[1]
        lst.append(point_idx)
        graph.add_edge(SOURCE, point_idx, capacity=max_capacity)
    for point in sink_points:
        point_idx = point[0] * width + point[1]
        lst.append(point_idx)
        graph.add_edge(point_idx, SINK, capacity=max_capacity)
    return lst

def buildGraph(image, source_points, sink_points):
    pixel_graph = nx.DiGraph()
    for x in range(image.size + 2):
        for y in range(image.size + 2):
            pixel_graph.add_edge(x, y, capacity=0)
    max_capacity = connectPixels(pixel_graph, image)
    source_sink = connectSourceAndSink(pixel_graph, 0, image.size - 1, source_points, sink_points, max_capacity)
    return pixel_graph, source_sink

def colorPixel(i, j):
    try:
        image[i][j] = (255, 0, 0)
    except:
        print(i, j)
def colorPixelr(i, j):
    try:
        image[i][j] = (0, 0, 255)
    except:
        print(i, j)

def displayCut(image, cuts):

    r, c, _ = image.shape
    for c in cuts:
        if (
            c[0] != image.size - 2
            and c[0] != image.size - 1
            and c[1] != image.size - 2
            and c[1] != image.size - 1
        ):
            colorPixel(c[0] // r, c[0] % r)
            colorPixel(c[1] // r, c[1] % r)
    return image

if __name__ == "__main__":
    image = cv2.imread("./images/test1.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (30, 30))
    source_points, sink_points = get_user_input(image)

    graph, source_sink = buildGraph(image, source_points, sink_points)

    cuts = augmentingPath2(graph, image.size, image.size + 1)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for i in source_sink:
        colorPixelr(i // 30, i % 30)
    image = displayCut(image, cuts)
    image = cv2.resize(image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    print(cuts)
