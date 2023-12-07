"""
Julie Jiang's implementation of the Ford-Fulkerson algorithm modified to work with NetworkX graphs.
(https://github.com/julie-jiang/image-segmentation/blob/master/augmentingPath.py)
"""

from queue import Queue
import numpy as np
import networkx as nx
import sys

def bfs(rGraph, V, s, t, parent):
    """
    Performs a Breadth-First Search (BFS) on the residual graph to find a path from source to sink.

    Args:
        rGraph (networkx.DiGraph): The residual graph.
        V (int): The number of vertices in the graph.
        s (int): The index of the source node.
        t (int): The index of the sink node.
        parent (array): The array to store the path found by BFS.

    Returns:
        (bool): True if a path is found from source to sink, False otherwise.
    """
    q = Queue()
    visited = np.zeros(V, dtype=bool)
    q.put(s)
    visited[s] = True
    parent[s]  = -1

    while not q.empty():
        u = q.get()
        for v in range(V):
            if (not visited[v]) and rGraph.get_edge_data(u, v)["capacity"] > 0:
                q.put(v)
                parent[v] = u
                visited[v] = True

    return visited[t]

def dfs(rGraph, V, s, visited):
    """
    Performs a Depth-First Search (DFS) on the residual graph to mark reachable vertices from source.

    Args:
        rGraph (networkx.DiGraph): The residual graph.
        V (int): The number of vertices in the graph.
        s (int): The index of the source node.
        visited (array): An array to keep track of visited vertices.
    """
    stack = [s]
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            stack.extend([u for u in range(V) if rGraph.get_edge_data(v, u)["capacity"] > 0])

def augmentingPath(graph, s, t):
    """
    Implements the Ford-Fulkerson algorithm to find the maximum flow and the corresponding minimum cuts
    in a flow network.

    Args:
        graph (networkx.DiGraph): The original graph.
        s (int): The index of the source node.
        t (int): The index of the sink node.

    Returns:
        (list): A list of tuples representing the cuts in the graph.
    """
    print("Running augmenting path algorithm")
    # Deep copy of graph to create residual graph
    rGraph = graph.copy()
    V = nx.number_of_nodes(graph)
    parent = np.zeros(V, dtype='int32')

    # Keep finding new paths until no path can be found
    while bfs(rGraph, V, s, t, parent):
        pathFlow = float("inf")
        v = t
        # Find the minimum capacity in the found path
        while v != s:
            u = parent[v]
            pathFlow = min(pathFlow, rGraph.get_edge_data(u, v)["capacity"])
            v = parent[v]

        # Update the capacities in the residual graph along the path
        v = t
        while v != s:
            u = parent[v]
            rGraph.get_edge_data(u, v)["capacity"] -= pathFlow
            rGraph.get_edge_data(v, u)["capacity"] += pathFlow
            v = parent[v]

    # Mark the reachable nodes from the source in the residual graph
    visited = np.zeros(V, dtype=bool)
    dfs(rGraph, V, s, visited)

    # Find the edges that cross from the visited to the unvisited vertices, which form the cuts
    cuts = []
    for i in range(V):
        for j in range(V):
            if visited[i] and not visited[j] and graph.get_edge_data(i, j)["capacity"]:
                cuts.append((i, j))

    return cuts
