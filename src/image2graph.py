import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import exp, pow
from augmenting_path import augmentingPath


def boundaryPenalty(ip, iq):
    bp = 100 * exp(-pow(int(ip) - int(iq), 2) / (2 * pow(30, 2)))
    return bp


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
                graph[x][y] = graph[y][x] = bp
                max_capacity = max(max_capacity, bp)

            # Don't add right edges for pixels on the right edge
            if col + 1 < image_width:
                y = row * image_width + col + 1
                bp = boundaryPenalty(image[row][col], image[row][col + 1])
                graph[x][y] = graph[y][x] = bp
                max_capacity = max(max_capacity, bp)
    return max_capacity


def connectSourceAndSink(graph, image, source_pixel, sink_pixel, max_capacity):
    width, height = image.shape
    SOURCE = len(graph) - 2
    SINK = len(graph) - 1
    graph[SOURCE][source_pixel] = max_capacity
    graph[SOURCE][source_pixel + 1] = max_capacity
    graph[SOURCE][source_pixel + 2] = max_capacity
    graph[SOURCE][source_pixel + width] = max_capacity
    graph[SOURCE][source_pixel + width + 1] = max_capacity
    graph[SOURCE][source_pixel + 2 * width] = max_capacity
    graph[sink_pixel][SINK] = max_capacity
    graph[sink_pixel - 1][SINK] = max_capacity
    graph[sink_pixel - 2][SINK] = max_capacity
    graph[sink_pixel - width ][SINK] = max_capacity
    graph[sink_pixel - width - 1][SINK] = max_capacity
    graph[sink_pixel - 2 * width][SINK] = max_capacity
    return [source_pixel, source_pixel + 1, source_pixel + 2, source_pixel + width, source_pixel + width + 1, source_pixel + 2 * width, sink_pixel, sink_pixel - 1, sink_pixel - 2, sink_pixel - width, sink_pixel - width - 1, sink_pixel - 2 * width]


def buildGraph(image):
    pixel_graph = np.zeros((image.size + 2, image.size + 2), dtype="int32")
    max_capacity = connectPixels(pixel_graph, image)
    lst = connectSourceAndSink(pixel_graph, image, 0, image.size - 1, max_capacity)
    return pixel_graph, lst


def bfs_find_path(graph, source, sink):
    rows, cols = graph.shape
    queue = [(source, [])]
    visited = set()
    while queue:
        current, path = queue.pop(0)
        if current == sink:
            return path
        visited.add(current)
        for neighbour in range(cols):
            if graph[current, neighbour] > 0 and neighbour not in visited:
                queue.append((neighbour, path + [(current, neighbour)]))
    return None


def augment_flow(graph, path):
    flow = min(graph[u, v] for u, v in path)
    for u, v in path:
        graph[u, v] -= flow
        graph[v, u] += flow


def find_min_cut(graph, source):
    rows, cols = graph.shape
    visited = {source}
    queue = [source]
    while queue:
        u = queue.pop(0)
        for v in range(cols):
            if graph[u, v] > 0 and v not in visited:
                visited.add(v)
                queue.append(v)
    return visited


def augmentingPath2(graph, source, sink):
    path = bfs_find_path(graph, source, sink)
    while path:
        augment_flow(graph, path)
        path = bfs_find_path(graph, source, sink)

    visited = find_min_cut(graph, source)
    cuts = []
    for u in visited:
        for v in range(graph.shape[1]):
            if v not in visited and graph[u, v] == 0:
                cuts.append((u, v))
    return cuts

def colorPixel(i, j):
    try:
        image[i][j] = (255, 0, 0)
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
    graph, lst = buildGraph(image)
    cuts = augmentingPath2(graph, len(graph) - 2, len(graph) - 1)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for i in lst:
        colorPixel(i // 30, i % 30)
    image = displayCut(image, cuts)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    print("Cuts for segmentation:", cuts)

# def bfs(graph, source, target):
#     visited = {source}
#     queue = [(source, [])]
#     while queue:
#         current, path = queue.pop(0)
#         if current == target:
#             return path
#         for neighbor in graph.neighbors(current):
#             residual = graph[current][neighbor]["weight"] - graph[current][
#                 neighbor
#             ].get("flow", 0)
#             if residual > 0 and neighbor not in visited:
#                 visited.add(neighbor)
#                 queue.append((neighbor, path + [(current, neighbor)]))
#     return None


# def ford_fulkerson(graph, source, sink):
#     max_flow = 0
#     path = bfs(graph, source, sink)
#     while path:
#         # Find minimum residual capacity of the edges along the path
#         flow = min(graph[u][v]["weight"] - graph[u][v].get("flow", 0) for u, v in path)
#         for u, v in path:
#             if "flow" not in graph[u][v]:
#                 graph[u][v]["flow"] = 0
#             if "flow" not in graph[v][u]:
#                 graph[v][u] = {"weight": 0, "flow": 0}
#             # Update residual capacities of the forward and backward edges
#             graph[u][v]["flow"] += flow
#             graph[v][u]["flow"] -= flow
#         max_flow += flow
#         path = bfs(graph, source, sink)
#     return graph, max_flow
