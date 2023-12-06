from math import exp, pow
import numpy as np
import cv2
import networkx as nx
from nx_augmenting_path import augmentingPath
from our_nx_augmenting_path import augmentingPath2
from nx_user_input import get_user_input, SCALE_FACTOR

def boundaryPenalty(p1, p2):
    """
    Calculate the boundary penalty between two pixels.

    Args:
        p1 (int): The intensity value of the first pixel.
        p2 (int): The intensity value of the second pixel.

    Returns:
        float: The calculated boundary penalty value.
    """
    return 100 * exp(-pow(int(p1) - int(p2), 2) / (2 * pow(30, 2)))


def connectPixels(graph, image):
    """
    Connects each pixel in the image to its adjacent pixels in the graph with an edge,
    where the edge capacity is determined by the boundary penalty between pixels.

    Args:
        graph (networkx.DiGraph): The graph to which the edges are added.
        image (np.array): The image represented as a 2D numpy array.

    Returns:
        int: The maximum capacity found among all the added edges.
    """
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
    """
    Connects source and sink nodes to the designated source and sink points in the graph,
    with edges having capacities set to the maximum capacity.

    Args:
        graph (networkx.DiGraph): The graph where the nodes and edges are added.
        source_pixel, sink_pixel (int): Indices of the source and sink pixels in the graph.
        source_points, sink_points (list of tuples): Points to be connected to the source and sink.
        max_capacity (int): The maximum capacity to be used for the edges connecting to source and sink.

    Returns:
        list: A list containing the indices of the source and sink points.
    """
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
    """
    Builds a directed graph from the given image, connecting pixels and adding source and sink.

    Args:
        image (np.array): The image represented as a 2D numpy array.
        source_points, sink_points (list of tuples): Points to be used as source and sink in the graph.

    Returns:
        tuple: A tuple containing the graph and a list of indices of source and sink points.
    """
    pixel_graph = nx.DiGraph()
    for x in range(image.size + 2):
        for y in range(image.size + 2):
            pixel_graph.add_edge(x, y, capacity=0)
    max_capacity = connectPixels(pixel_graph, image)
    source_sink = connectSourceAndSink(pixel_graph, 0, image.size - 1, source_points, sink_points, max_capacity)
    return pixel_graph, source_sink

def colorPixel(image, i, j, red=False):
    """
    Colors a pixel at a specified position in the global `image` array.

    Args:
        image (np.array): The image to draw on.
        i, j (int): The row and column indices of the pixel in the image.
        red (bool): Whether to color the pixel red or blue.

    Returns:
        None
    """
    try:
        if red:
            image[i][j] = (0, 0, 255)
        else:
            image[i][j] = (255, 0, 0)
    except:
        print(i, j)

def displayCut(image, cuts):
    """
    Displays the cut on the image by coloring the pixels on the cut.

    Args:
        image (np.array): The image on which the cut is to be displayed.
        cuts (list of tuples): The list of cuts where each cut is a tuple of two pixel indices.

    Returns:
        np.array: The image with the cut displayed.
    """
    r, c, _ = image.shape
    for c in cuts:
        if (
            c[0] != image.size - 2
            and c[0] != image.size - 1
            and c[1] != image.size - 2
            and c[1] != image.size - 1
        ):
            colorPixel(image, c[0] // r, c[0] % r)
            colorPixel(image, c[1] // r, c[1] % r)
    return image

if __name__ == "__main__":
    image = cv2.imread("./images/test1.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (30, 30))
    source_points, sink_points = get_user_input(image)

    graph, source_sink = buildGraph(image, source_points, sink_points)

    cuts = augmentingPath2(graph, image.size, image.size + 1)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for i in source_sink:
        colorPixel(image, i // 30, i % 30, red=True)
    image = displayCut(image, cuts)
    image = cv2.resize(image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    print(cuts)
