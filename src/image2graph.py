import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import exp, pow


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
    graph[sink_pixel][SINK - 1] = max_capacity
    graph[sink_pixel][SINK - 2] = max_capacity
    graph[sink_pixel][SINK - width] = max_capacity
    graph[sink_pixel][SINK - width - 1] = max_capacity
    graph[sink_pixel][SINK - 2 * width] = max_capacity


def buildGraph(image):
    pixel_graph = np.zeros((image.size + 2, image.size + 2), dtype="int32")
    max_capacity = connectPixels(pixel_graph, image)
    connectSourceAndSink(pixel_graph, image, 0, image.size - 1, max_capacity)
    return pixel_graph


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


def augmentingPath(graph, source, sink):
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


def displayCut(image, cuts):
    def colorPixel(i, j):
        try:
            image[i][j] = (255, 0, 0)
        except:
            pass

    r, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
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
    image = cv2.imread("./images/test.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (10, 10))
    graph = buildGraph(image)
    print(graph.size)
    cuts = augmentingPath(graph, len(graph) - 2, len(graph) - 1)
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
