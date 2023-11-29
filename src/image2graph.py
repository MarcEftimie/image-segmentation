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
    return pixel_graph


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


# Your existing buildGraph and other functions

if __name__ == "__main__":
    image = cv2.imread("./images/monkey.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (50, 50))
    graph = buildGraph(image)
    graph, max_flow = ford_fulkerson(graph, "SOURCE", "SINK")
    print(graph.edges.data())
