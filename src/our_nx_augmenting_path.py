"""
Our implementation of the Ford-Fulkerson algorithm using Breadth-First Search (BFS) to find augmenting paths.
"""

def augmentingPath2(graph, source, sink):
    """
    Finds the augmenting paths in the given graph from source to sink and returns the cuts
    corresponding to the maximum flow.

    This function implements the Ford-Fulkerson algorithm using Breadth-First Search (BFS)
    to find augmenting paths.

    Args:
        graph (networkx.DiGraph): The graph on which the Ford-Fulkerson algorithm is applied.
        source (int): The index of the source node in the graph.
        sink (int): The index of the sink node in the graph.

    Returns:
        (list): A list of tuples representing the cuts in the graph, where each cut is a tuple of two node indices.
    """

    def bfs():
        """
        Performs a Breadth-First Search to find an augmenting path from source to sink.

        Returns:
            (tuple): A tuple containing a boolean indicating if a path was found, and a dictionary representing
                the parent of each node in the found path.
        """
        visited = {node: False for node in graph.nodes}  # Track visited nodes
        queue = []  # Queue for BFS
        parent = {node: -1 for node in graph.nodes}  # Store parent of each node

        # Initialize BFS from the source
        queue.append(source)
        visited[source] = True

        while queue:
            u = queue.pop(0)
            for v in graph.neighbors(u):
                # If the neighbor hasn't been visited and the edge has remaining capacity
                if not visited[v] and graph[u][v]['capacity'] > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u
                    # If we reached the sink, return the path found
                    if v == sink:
                        return True, parent
        # If no path is found
        return False, parent

    max_flow = 0  # Initialize maximum flow to zero
    cuts = []
    path_flow, parent = bfs()  # Find the first augmenting path

    while path_flow:
        # Find the minimum capacity in the found augmenting path
        min_capacity = float('inf')
        s = sink
        while s != source:
            min_capacity = min(min_capacity, graph[parent[s]][s]['capacity'])
            s = parent[s]

        # Update maximum flow
        max_flow += min_capacity

        # Update residual capacities of the edges and reverse edges
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v]['capacity'] -= min_capacity
            graph[v][u]['capacity'] += min_capacity
            # If the capacity becomes zero, add the edge to cuts
            if graph[u][v]['capacity'] == 0:
                cuts.append((u, v))
            v = parent[v]

        # Find the next augmenting path
        path_flow, parent = bfs()

    return cuts
