# dependencies for our dijkstra's implementation
from queue import PriorityQueue
from math import inf
# graph dependency  
import networkx as nx


"""Dijkstra's shortest path algorithm"""
def dijkstra(graph: 'networkx.classes.graph.Graph', start: str, end: str) -> 'List':
    """Get the shortest path of nodes by going backwards through prev list
    credits: https://github.com/blkrt/dijkstra-python/blob/3dfeaa789e013567cd1d55c9a4db659309dea7a5/dijkstra.py#L5-L10"""
    def backtrace(prev, start, end):
        node = end
        path = []
        while node != start:
            path.append(node)
            node = prev[node]
        path.append(node) 
        path.reverse()
        return path
        
    """get the cost of edges from node -> node
    cost(u,v) = edge_weight(u,v)"""
    def cost(u, v):
        return graph.get_edge_data(u,v).get('weight')
        
    """main algorithm"""
    # predecessor of current node on shortest path 
    prev = {} 
    # initialize distances from start -> given node i.e. dist[node] = dist(start: str, node: str)
    dist = {v: inf for v in list(nx.nodes(graph))} 
    # nodes we've visited
    visited = set() 
    # prioritize nodes from start -> node with the shortest distance!
    ## elements stored as tuples (distance, node) 
    pq = PriorityQueue()  
    
    dist[start] = 0  # dist from start -> start is zero
    pq.put((dist[start], start))
    
    while 0 != pq.qsize():
        curr_cost, curr = pq.get()
        visited.add(curr)
        print(f'visiting {curr}')
        # look at curr's adjacent nodes
        for neighbor in dict(graph.adjacency()).get(curr):
            # if we found a shorter path 
            path = dist[curr] + cost(curr, neighbor)
            if path < dist[neighbor]:
                # update the distance, we found a shorter one!
                dist[neighbor] = path
                # update the previous node to be prev on new shortest path
                prev[neighbor] = curr
                # if we haven't visited the neighbor
                if neighbor not in visited:
                    # insert into priority queue and mark as visited
                    visited.add(neighbor)
                    pq.put((dist[neighbor],neighbor))
                # otherwise update the entry in the priority queue
                else:
                    # remove old
                    _ = pq.get((dist[neighbor],neighbor))
                    # insert new
                    pq.put((dist[neighbor],neighbor))
    print("=== Dijkstra's Algo Output ===")
    print("Distances")
    print(dist)
    print("Visited")
    print(visited)
    print("Previous")
    print(prev)
    # we are done after every possible path has been checked 
    return backtrace(prev, start, end), dist[end]
    
import matplotlib.pyplot as plt  # for drawing pretty graphs


def draw_graph_paths(G, shortest_path=[], pos=nx.planar_layout):
    # node colors
    color_map = []
    start_node = shortest_path[0] if len(shortest_path) else None
    end_node = shortest_path[-1] if len(shortest_path) else None
    for node in G.nodes():
        if node == start_node:
            color_map.append('#42b395')
        elif node == end_node:
            color_map.append('#fd4659')
        else:
            color_map.append('#b7c9e2')
    # nodes
    pos = pos(G)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=color_map)
    # edges
    if 0 != len(shortest_path):
        shortest_path = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)] 
    paths = [(u, v) for (u, v) in G.edges() if (u,v) not in set(shortest_path)]
    nx.draw_networkx_edges(G, pos, edgelist=paths, width=6)
    nx.draw_networkx_edges(G, pos, edgelist=shortest_path, width=6, alpha=0.5, edge_color='r', style='dashed')
    # labels  
    nx.draw_networkx_edge_labels(G,pos,edge_labels=nx.get_edge_attributes(G,'weight'))
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    plt.show()
G = nx.Graph()
G.add_edge('A', 'B', weight=4)
G.add_edge('B', 'D', weight=2)
G.add_edge('A', 'C', weight=3)
G.add_edge('C', 'D', weight=4)
print(nx.shortest_path(G, 'A', 'D', weight='weight'))
print(G.is_directed())

shortest_path, distance = dijkstra(G, 'A', 'D')
print("shortest path")
print(shortest_path)
print("distance")
print(distance)

#draw_graph_paths(G, shortest_path=shortest_path)


# ======================================================
"""Example from https://brilliant.org/wiki/dijkstras-short-path-finder/#implementation"""
G1 = nx.Graph()
G1.add_edge('S', 'A', weight=3)
G1.add_edge('S', 'C', weight=2)
G1.add_edge('S', 'F', weight=6)
G1.add_edge('A', 'D', weight=1)
G1.add_edge('A', 'B', weight=6)
G1.add_edge('C', 'A', weight=2)
G1.add_edge('C', 'D', weight=3)
G1.add_edge('F', 'E', weight=2)
G1.add_edge('B', 'E', weight=1)
G1.add_edge('D', 'E', weight=4)
print(nx.shortest_path(G1, 'S', 'B', weight='weight'))
print(G1.is_directed())

shortest_path, distance = dijkstra(G1, 'S', 'B')
print("shortest path")
print(shortest_path)
print("distance")
print(distance)

draw_graph_paths(G1, shortest_path=shortest_path)

# ==============================================
"""Example from Computerphile https://youtu.be/G2azC3A4OQTE?t=132"""
G2 = nx.Graph()
G2.add_edge('S', 'A', weight=7)
G2.add_edge('S', 'B', weight=2)
G2.add_edge('S', 'C', weight=3)
G2.add_edge('A', 'B', weight=3)
G2.add_edge('A', 'D', weight=4)
G2.add_edge('C', 'L', weight=2)
G2.add_edge('B', 'D', weight=4)
G2.add_edge('B', 'H', weight=1)
G2.add_edge('D', 'F', weight=5)
G2.add_edge('L', 'I', weight=4)
G2.add_edge('L', 'J', weight=4)
G2.add_edge('H', 'F', weight=3)
G2.add_edge('H', 'G', weight=2)
G2.add_edge('I', 'K', weight=4)
G2.add_edge('I', 'J', weight=6)
G2.add_edge('J', 'K', weight=4)
G2.add_edge('G', 'E', weight=2)
G2.add_edge('K', 'E', weight=5)
print(nx.shortest_path(G2, 'S', 'E', weight='weight'))
print(G2.is_directed())

shortest_path, distance = dijkstra(G2, 'S', 'E')
print("shortest path")
print(shortest_path)
print("distance")
print(distance)

#draw_graph_paths(G2, shortest_path=shortest_path)