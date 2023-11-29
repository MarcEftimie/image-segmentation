import numpy as np
import cv2
import networkx as nx


def boundaryPenalty(p1, p2):
    if p1 > p2:
        return p1 - p2
    return p2 - p1


def connectPixels(graph, image):
    max_capacity = -float("inf")
    image_height, image_width = image.shape
    for row in range(image_height):
        for col in range(image_width):
            pixel_count = row * image_width + col

            # Don't add bottom edges for pixels on the bottom edge
            if row != image_height - 1:
                bp = boundaryPenalty(image[row][col], image[row + 1][col])
                graph.add_edge(pixel_count, pixel_count + image_width, weight=bp)
                max_capacity = max(max_capacity, bp)

            # Don't add right edges for pixels on the right edge
            if col != image_width - 1:
                bp = boundaryPenalty(image[row][col], image[row][col + 1])
                graph.add_edge(pixel_count, pixel_count + 1, weight=bp)
                max_capacity = max(max_capacity, bp)

    return max_capacity


def connectSourceAndSink(graph, source_pixel, sink_pixel, max_capacity):
    graph.add_edge("SOURCE", source_pixel, weight=max_capacity)
    graph.add_edge("SINK", sink_pixel, weight=max_capacity)


def buildGraph(image):
    pixel_graph = nx.Graph()
    max_capacity = connectPixels(pixel_graph, image)
    connectSourceAndSink(pixel_graph, 0, image.size - 1, max_capacity)
    print(pixel_graph.edges.data())


image = cv2.imread("./images/monkey.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (10, 10))
buildGraph(image)
