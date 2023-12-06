def augmentingPath2(graph, source, sink):
    def bfs():
        visited = {node: False for node in graph.nodes}
        queue = []
        parent = {node: -1 for node in graph.nodes}

        queue.append(source)
        visited[source] = True

        while queue:
            u = queue.pop(0)
            for v in graph.neighbors(u):
                if not visited[v] and graph[u][v]['capacity'] > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u
                    if v == sink:
                        return True, parent
        return False, parent

    max_flow = 0
    cuts = []
    path_flow, parent = bfs()

    while path_flow:
        min_capacity = float('inf')
        s = sink
        while s != source:
            min_capacity = min(min_capacity, graph[parent[s]][s]['capacity'])
            s = parent[s]

        max_flow += min_capacity
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v]['capacity'] -= min_capacity
            graph[v][u]['capacity'] += min_capacity
            if graph[u][v]['capacity'] == 0:
                cuts.append((u, v))
            v = parent[v]

        path_flow, parent = bfs()

    return cuts
