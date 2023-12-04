from queue import Queue
import numpy as np
import networkx as nx
import sys

def bfs(rGraph, V, s, t, parent):
    q = Queue()
    visited = np.zeros(V, dtype=bool)
    q.put(s)
    visited[s] = True
    parent[s]  = -1
    lst = {}
    while not q.empty():
        u = q.get()
        for v in range(V):
            if (not visited[v]) and rGraph.get_edge_data(u, v)["capacity"] > 0:
                q.put(v)
                parent[v] = u
                visited[v] = True
    
    return visited[t]

def dfs(rGraph, V, s, visited):
    stack = [s]
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            stack.extend([u for u in range(V) if rGraph.get_edge_data(v, u)["capacity"] > 0])

def augmentingPath(graph, s, t):
    print("Running augmenting path algorithm")
    # do deep copy of graph to rGraph
    rGraph = graph.copy()
    V = nx.number_of_nodes(graph)
    parent = np.zeros(V, dtype='int32')
    while bfs(rGraph, V, s, t, parent):
        pathFlow = float("inf")
        v = t
        while v != s:
            u = parent[v]
            pathFlow = min(pathFlow, rGraph.get_edge_data(u, v)["capacity"])
            v = parent[v]

        v = t
        while v != s:
            u = parent[v]
            rGraph.get_edge_data(u, v)["capacity"] -= pathFlow
            rGraph.get_edge_data(v, u)["capacity"] += pathFlow
            v = parent[v]

    visited = np.zeros(V, dtype=bool)
    dfs(rGraph, V, s, visited)

    cuts = []

    for i in range(V):
        for j in range(V):
            if visited[i] and not visited[j] and graph.get_edge_data(i, j)["capacity"]:
                cuts.append((i, j))
    return cuts
