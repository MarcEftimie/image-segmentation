import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import exp, pow
from augmenting_path import augmentingPath


def boundaryPenalty(ip, iq):
    return 100 * exp(-pow(int(ip) - int(iq), 2) / (2 * pow(30, 2)))

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


def bfs(rGraph, s, t, parent):
    visited = [False] * len(rGraph)
    queue = []

    queue.append(s)
    visited[s] = True

    while queue:
        u = queue.pop(0)

        for ind, val in enumerate(rGraph[u]):
            if visited[ind] == False and val > 0:
                queue.append(ind)
                visited[ind] = True
                parent[ind] = u

    return True if visited[t] else False

def augmentingPath2(rGraph, source, sink):
    parent = [-1] * len(rGraph)
    max_flow = 0
    cuts = []

    while bfs(rGraph, source, sink, parent):
        path_flow = float("Inf")
        s = sink

        while s != source:
            path_flow = min(path_flow, rGraph[parent[s]][s])
            s = parent[s]

        max_flow += path_flow
        v = sink

        while v != source:
            u = parent[v]
            rGraph[u][v] -= path_flow
            rGraph[v][u] += path_flow
            if rGraph[u][v] == 0 and rGraph[v][u] > 0:
                cuts.append((u, v))
            v = parent[v]

    return cuts

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
    graph, lst = buildGraph(image)
    qgraph = graph.copy()
    cuts = augmentingPath(qgraph, len(graph) - 2, len(graph) - 1)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for i in lst:
        colorPixelr(i // 30, i % 30)
    image = displayCut(image, cuts)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    print("Cuts for segmentation:", cuts)

