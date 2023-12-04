import numpy as np
import cv2
import networkx as nx
from nx_augmenting_path import augmentingPath
from math import exp, pow


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


def connectSourceAndSink(graph, source_pixel, sink_pixel, max_capacity):
    width, height = image.shape
    SOURCE = image.size
    SINK = image.size + 1
    graph.add_edge(SOURCE, source_pixel, capacity=max_capacity)
    graph.add_edge(SOURCE, source_pixel + 1, capacity=max_capacity)
    graph.add_edge(SOURCE, source_pixel + 2, capacity=max_capacity)
    graph.add_edge(SOURCE, source_pixel + width, capacity=max_capacity)
    graph.add_edge(SOURCE, source_pixel + width + 1, capacity=max_capacity)
    graph.add_edge(SOURCE, source_pixel + 2 * width, capacity=max_capacity)
    graph.add_edge(sink_pixel, SINK, capacity=max_capacity)
    graph.add_edge(sink_pixel - 1, SINK, capacity=max_capacity)
    graph.add_edge(sink_pixel - 2, SINK, capacity=max_capacity)
    graph.add_edge(sink_pixel - width, SINK, capacity=max_capacity)
    graph.add_edge(sink_pixel - width - 1, SINK, capacity=max_capacity)
    graph.add_edge(sink_pixel - 2 * width, SINK, capacity=max_capacity)
    return [source_pixel, source_pixel + 1, source_pixel + 2, source_pixel + width, source_pixel + width + 1, source_pixel + 2 * width, sink_pixel, sink_pixel - 1, sink_pixel - 2, sink_pixel - width, sink_pixel - width - 1, sink_pixel - 2 * width]

def buildGraph(image):
    pixel_graph = nx.DiGraph()
    for x in range(image.size + 2):
        for y in range(image.size + 2):
            pixel_graph.add_edge(x, y, capacity=0)
    max_capacity = connectPixels(pixel_graph, image)
    source_sink = connectSourceAndSink(pixel_graph, 0, image.size - 1, max_capacity)
    return pixel_graph, source_sink

def bfs(graph, source, target):
    visited = {source}
    queue = [(source, [])]
    while queue:
        current, path = queue.pop(0)
        if current == target:
            return path
        for neighbor in graph.neighbors(current):
            residual = graph[current][neighbor]["weight"] - graph[current][
                neighbor
            ].get("flow", 0)
            if residual > 0 and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [(current, neighbor)]))
    return None

def ford_fulkerson(graph, source, sink):
    max_flow = 0
    path = bfs(graph, source, sink)
    while path:
        # Find minimum residual capacity of the edges along the path
        flow = min(graph[u][v]["weight"] - graph[u][v].get("flow", 0) for u, v in path)
        for u, v in path:
            if "flow" not in graph[u][v]:
                graph[u][v]["flow"] = 0
            if "flow" not in graph[v][u]:
                graph[v][u] = {"weight": 0, "flow": 0}
            # Update residual capacities of the forward and backward edges
            graph[u][v]["flow"] += flow
            graph[v][u]["flow"] -= flow
        max_flow += flow
        path = bfs(graph, source, sink)
    return graph, max_flow

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
    graph, source_sink = buildGraph(image)
    cuts = augmentingPath(graph, image.size, image.size + 1)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for i in source_sink:
        colorPixelr(i // 30, i % 30)
    image = displayCut(image, cuts)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    print(cuts)
