from grid import Grid
from graph import Graph
from collections import deque
import heapq

g = Grid(2,3, [[6,1,4],[2,3,5]])

class Solver(): 
    """
    A solver class, to be implemented.
    """

    def __init__(self, grid):
        self.grid = grid
    
    def get_solution(self):
        """
        Solves the grid and returns the sequence of swaps at the format 
        [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...]. 
        """
        etat0 = self.grid.state
        L=[]
        etat = etat0
        for element in range(self.grid.m*self.grid.n):
            #We define i0 and j0 as the coordinates of the cell of "element" if the grid was sorted.
            #The condition below is equivalent to this cell touching the right border, in which case the definition of i0 and j0 defers.
            if (element+1)%self.grid.n != 0:
                (i0, j0) = ((element+1)//self.grid.n, ((element+1)%self.grid.n)-1)
            else:
                (i0,j0) = ((element+1)//self.grid.n-1, self.grid.n-1)
            for i in range(self.grid.m):
                for j in range(self.grid.n):
                    if etat[i][j] == element+1:
                        #We define deltai and deltaj as the distance in terms of columns and in terms of lines from which the cell of element is from its position in the sorted grid.
                        deltai=abs(i-i0)
                        deltaj=abs(j-j0)
                        #We start by swaping the cell horizontally (by columns), and stop when j=j0 (correct column).
                        #Then, we do the same thing but vertically (by lines) until i=i0.
                        if j==j0:
                            pass
                        else:
                            for k in range(abs(j-j0)):
                                if j0<j:
                                    L.append(((i,j-k),(i,j-(k+1))))
                                else:
                                    L.append(((i,j+k),(i,j+(k+1))))
                            j=j0
                        if i==i0:
                            pass
                        else:
                            for l in range(abs(i-i0)):
                                if i0<i:
                                    L.append(((i-l,j),(i-(l+1),j)))
                                else:
                                    L.append(((i+l,j),(i+(l+1),j)))
                            i=i0
                    else:
                        pass
            #Here, "etat" corresponds to the state of the grid after doint the swaps described by the list L.
            # (len(L)-(deltai+deltaj)) corresponds to the number of swaps we made at this step, for this element.
            etat = Grid(self.grid.m, self.grid.n, etat).swap_seq(L[(len(L)-(deltai+deltaj)):])
        print("la séquence de swap est la suivante", L) 
    ''' The time complexity of this naive solution is in O(((nm)^2)*(n+m)).
    It is not optimal, as we can see in the following example :
    Considering the following grid [[2,3],[1,4]], this algorithm will do the 4 following steps until getting to the sorted state :
    [[2,3],[1,4]] -> [[1,3],[2,4]] -> [[1,3],[4,2]] -> [[1,2],[4,3]] -> [[1,2],[3,4]].
    But with two swaps : 
    [[2,3],[1,4]] -> [[3,2],[1,4]] -> [[1,2],[3,4]]
    Which proves that this solution isn't optimal. '''

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
                for neighbor in self.grid.graph[node]:# Iterate through the neighbors of the current node
                    queue.append((neighbor, path + [neighbor])) # Append the neighbor to the path and add it to the queue
        return None # If the destination node is not reachable, return None

    def find_optimal_solution(self):
        """
        Finds the optimal solution for the swap puzzle using BFS on the graph of all possible states.

        Returns:
        --------
        solution: list[tuple[tuple[int]]] | None
            The optimal solution as a list of swap pairs, where each swap pair is a tuple of two cells.
            Returns None if no solution is found.
        """
        all_states = self.grid.generate_possible_states() # Generate all possible states of the grid
        graph = Graph(nodes=all_states) # Construct the graph of all possible states
        for state in all_states:
            # Generate valid neighbor states by performing all possible swaps
            for i in range(self.grid.m):
                for j in range(self.grid.n - 1):
                    neighbor_state = self.grid.state_after_swap(state, (i, j), (i, j + 1))
                    if neighbor_state in all_states:
                        graph.add_edge(state, neighbor_state)
            for i in range(self.grid.m - 1):
                for j in range(self.grid.n):
                    neighbor_state = self.grid.state_after_swap(state, (i, j), (i + 1, j))
                    if neighbor_state in all_states:
                        graph.add_edge(state, neighbor_state)
        """print(graph)""" #Optional but allows us to see the graph that we just created
        # Apply BFS to find the optimal solution
        start_state = tuple(map(tuple, self.grid.state)) #initial state of the grid 
        print("état de départ :", start_state) #useful when we struggled to find errors
        target_state = tuple(tuple(range(i*self.grid.n+1, (i+1)*self.grid.n+1)) for i in range(self.grid.m)) #final state of the grid completely sorted
        print("état d'arrivée :",target_state)
        graph.plot_graph("graph3.png") #Optional but fun and practical to see the graph
        path = graph.bfs2(start_state, target_state) #bfs applied to the graph to search for the best path
        swap__seq = self.grid.swap_seq_from_path(path)


#Contrairement au code d'avant, celui-ci rend bien un chemin optimal le plus court possible.
# La complexité du programme est en O((n*m)!) + O((n*m)!) = O((n*m)!), le premier terme correspondant à la complexité de l'algorithme BfS, 
# et le second à la création du graphe de toutes les possibilités. """
    
    def find_optimal_solution_2(self):
        # Convertir l'état initial en tuple pour une utilisation dans les ensembles
        start_state = tuple(map(tuple, self.grid.state))
        # Créer un ensemble pour garder une trace des états déjà visités
        visited = set()
        # Créer une file pour parcourir les nœuds dans l'ordre BFS
        queue = deque([(start_state, [])])
        # on initialise la liste de tous les états intermédiaires
        states = [start_state]
        
        # Tant que la file n'est pas vide, continuer le BFS
        while queue:
            node, path = queue.popleft()  # Retirer le premier nœud de la file
            if node in visited:  # Vérifier si le nœud a déjà été visité
                continue
            visited.add(node)  # Marquer le nœud comme visité
            if self.grid.is_sorted(node):  # Vérifier si le nœud est dans l'état final trié
                for i in range(len(path)):
                    states.append(path[i][1])
                swap__seq = self.grid.swap_seq_from_path(states)
            # Obtenir les voisins du nœud actuel
            neighbors = self.grid.get_neighbors(node)
            # Ajouter les voisins non visités à la file avec le chemin mis à jour
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, path + [(node, neighbor)]))
        
        return None  # Retourner None si aucune solution n'est trouvée

    def find_optimal_solution_astar(self):
        start_state = tuple(map(tuple, self.grid.state))  # Convert the initial state to a tuple for use in sets
        target_state = tuple(tuple(range(i*self.grid.n+1, (i+1)*self.grid.n+1)) for i in range(self.grid.m))  # Convert the target state to a tuple for use in sets
        open_set = [(self.grid.heuristic_manhattan_distance(start_state, target_state), 0, start_state, [])]  # Initialize the priority queue with the initial state
        heapq.heapify(open_set)  # Turn the list into a heap to obtain a priority queue
        closed_set = set()  # Set to keep track of visited states
        
        while open_set:
            _, cost, current_state, path = heapq.heappop(open_set)  # Extract the node with the lowest cost from the priority queue
            if current_state in closed_set:  # Check if the node has already been visited
                continue
            closed_set.add(current_state)  # Add the node to the set of visited nodes
            
            if current_state == target_state:  # Check if the current state is the target state
                print("la séquence de swaps est : ", self.grid.swaps_seq_from_path_2(start_state, current_state))  # Return the sequence of swaps if the target state is reached
                break

            neighbors = self.grid.get_neighbors(current_state)  # Get valid neighbors of the current node
            for neighbor in neighbors:
                if neighbor not in closed_set:  # Check if the neighbor has not been visited
                    g = cost + 1  # Cost to reach this neighbor
                    h = self.grid.heuristic_manhattan_distance(neighbor, target_state)  # Heuristic value of the neighbor
                    f = g + h  # Total cost (actual cost + heuristic value)
                    heapq.heappush(open_set, (f, g, neighbor, path + [(current_state, neighbor)]))  # Add the neighbor to the priority queue with its total cost and updated path
        return None  # Return None if no solution is found
