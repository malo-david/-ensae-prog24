"""
This is the graph module. It contains a minimalistic Graph class.
"""
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    """
    A class representing undirected graphs as adjacency lists. 

    Attributes: 
    -----------
    nodes: NodeType
        A list of nodes. Nodes can be of any immutable type, e.g., integer, float, or string.
        We will usually use a list of integers 1, ..., n.
    graph: dict
        A dictionnary that contains the adjacency list of each node in the form
        graph[node] = [neighbor1, neighbor2, ...]
    nb_nodes: int
        The number of nodes.
    nb_edges: int
        The number of edges. 
    edges: list[tuple[NodeType, NodeType]]
        The list of all edges
    """

    def __init__(self, nodes=[]):
        """
        Initializes the graph with a set of nodes, and no edges. 

        Parameters: 
        -----------
        nodes: list, optional
            A list of nodes. Default is empty.
        """
        self.nodes = nodes 
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
        self.edges = []
        
    def __str__(self):
        """
        Prints the graph as a list of neighbors for each node (one per line)
        """
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output

    def __repr__(self): 
        """
        Returns a representation of the graph with number of nodes and edges.
        """
        return f"<graph.Graph: nb_nodes={self.nb_nodes}, nb_edges={self.nb_edges}>"

    def add_edge(self, node1, node2):
        """
        Adds an edge to the graph. Graphs are not oriented, hence an edge is added to the adjacency list of both end nodes. 
        When adding an edge between two nodes, if one of the ones does not exist it is added to the list of nodes.

        Parameters: 
        -----------
        node1: NodeType
            First end (node) of the edge
        node2: NodeType
            Second end (node) of the edge
        """
        if node1 not in self.graph:
            self.graph[node1] = []
            self.nb_nodes += 1
            self.nodes.append(node1)
        if node2 not in self.graph:
            self.graph[node2] = []
            self.nb_nodes += 1
            self.nodes.append(node2)

        self.graph[node1].append(node2)
        self.graph[node2].append(node1)
        self.nb_edges += 1
        self.edges.append((node1, node2))

    def bfs2(self, src, dst): 
        """
        Finds a shortest path from src to dst by BFS.  

        Parameters: 
        -----------
        src: NodeType
            The source node.
        dst: NodeType
            The destination node.

        Output: 
        -------
        path: list[NodeType] | None
            The shortest path from src to dst. Returns None if dst is not reachable from src
        """ 
        # This code was written from pseudo-codes found on the links proposed by the TP
        visited = set() # Initialize a set to keep track of visited nodes. At first we used a list but it didn't work, thus we tried with a set and it was much more pratcical.
        queue = deque([(src, [src])]) # Initialize a deque for BFS queue, starting with the source node and its path : useful since it supports fast operations such as adding or removing elements from both ends of the queue
        while queue: # Iterate until the queue is empty
            node, path = queue.popleft() # Pop the first element (node, path) from the queue
            # If the destination node is reached, return the path
            if node == dst:
                return path
            if node not in visited: # If the node has not been visited yet
                visited.add(node) # Mark the node as visited
                for neighbor in self.graph[node]:# Iterate through the neighbors of the current node
                    queue.append((neighbor, path + [neighbor])) # Append the neighbor to the path and add it to the queue
        return None # If the destination node is not reachable, return None

    def test_bfs2():
        graph = Graph({1: [2, 3], 2: [3, 4], 3: [4], 4: [3]})
        src = 1
        dst = 4
        expected_path = [1, 3, 4]
        assert graph.bfs2(src, dst) == expected_path

    def bfs2_quest8(self, src, dst): 
        visited = set()
        queue = deque([(src, [src])])
        while queue: 
            node, path = queue.popleft()
            if node == dst:
                return path
            if node not in visited: 
                visited.add(node) 
                for neighbor in self.graph[node]:
                    if neighbor not in visited and neighbor != src: # we don't visit the node src which can improve the algorithm
                        queue.append((neighbor, path + [neighbor]))
        return None

    def plot_graph(self, filename):
        G = nx.Graph()
        G.add_nodes_from(self.nodes) # Ajouter les nœuds au graphe
        G.add_edges_from(self.edges) # Ajouter les arêtes au graphe
        # Dessiner le graphe
        pos = nx.spring_layout(G)  # Layout du graphe
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray', linewidths=1, font_size=12)
        # Afficher le graphe
        plt.title("Graph Representation")
        plt.savefig(filename)
        plt.close()

    @classmethod
    def graph_from_file(cls, file_name):
        """
        Reads a text file and returns the graph as an object of the Graph class.

        The file should have the following format: 
            The first line of the file is 'n m'
            The next m lines have 'node1 node2'
        The nodes (node1, node2) should be named 1..n

        Parameters: 
        -----------
        file_name: str
            The name of the file

        Outputs: 
        -----------
        graph: Graph
            An object of the class Graph with the graph from file_name.
        """
        with open(file_name, "r") as file:
            n, m = map(int, file.readline().split())
            graph = Graph(range(1, n+1))
            for _ in range(m):
                edge = list(map(int, file.readline().split()))
                if len(edge) == 2:
                    node1, node2 = edge
                    graph.add_edge(node1, node2) # will add dist=1 by default
                else:
                    raise Exception("Format incorrect")
        return graph


#Tests
g2 = Graph.graph_from_file("input/graph1.in")

"""g2.plot_graph("graph.png")"""
