"""
Functions for converting an image to a graph and displaying the cut on the image.
"""

from math import exp, pow
import numpy as np
import networkx as nx

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
                    # convert bp to in since capacities can only be integers
                    graph.add_edge(x, y, capacity=int(bp)) 
                    graph.add_edge(y, x, capacity=int(bp))
                max_capacity = max(max_capacity, bp)

    return int(max_capacity)


def connectSourceAndSink(image, graph, SOURCE, SINK, source_points, sink_points, max_capacity):
    """
    Connects source and sink nodes to the designated source and sink points in the graph,
    with edges having capacities set to the maximum capacity.

    Args:
        graph (networkx.DiGraph): The graph where the nodes and edges are added.
        SOURCE, SINK (int): The source and sink node indices.
        source_points, sink_points (list of tuples): Points to be connected to the source and sink.
        max_capacity (int): The maximum capacity to be used for the edges connecting to source and sink.

    Returns:
        list: A list containing the indices of the source and sink points.
    """
    width, height = image.shape
    lst = []
    for point in source_points:
        point_idx = point[0] * width + point[1]     # row * width + col
        lst.append(point_idx)
        graph.add_edge(SOURCE, point_idx, capacity=max_capacity)
    for point in sink_points:
        point_idx = point[0] * width + point[1]     # row * width + col
        lst.append(point_idx)
        graph.add_edge(point_idx, SINK, capacity=max_capacity)
    return lst

def buildGraph(image, SOURCE, SINK, source_points, sink_points):
    """
    Builds a directed graph from the given image, connecting pixels and adding source and sink.

    Args:
        image (np.array): The image represented as a 2D numpy array.
        SOURCE, SINK (int): The source and sink node indices.
        source_points, sink_points (list of tuples): Points to be used as source and sink in the graph.

    Returns:
        tuple: A tuple containing the graph and a list of indices of source and sink points.
    """
    pixel_graph = nx.DiGraph()

    # add pixels as nodes to create adjacency matrix (represented as a graph)
    for x in range(image.size + 2):
        for y in range(image.size + 2):
            pixel_graph.add_edge(x, y, capacity=0)

    max_capacity = connectPixels(pixel_graph, image)

    # connect source and sink to user input points
    source_sink = connectSourceAndSink(image, pixel_graph, SOURCE, SINK, source_points, sink_points, max_capacity)
    return pixel_graph, source_sink

