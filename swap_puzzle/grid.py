"""
This is the grid module. It contains the Grid class and its associated methods.
"""
import sys
import heapq
from collections import deque
import sys 
sys.path.append("swap_puzzle/")
import unittest 
import random
import matplotlib.pyplot as plt
from itertools import permutations
from graph import Graph

class Grid():
    """
    A class representing the grid from the swap puzzle. It supports rectangular grids. 

    Attributes: 
    -----------
    m: int
        Number of lines in the grid
    n: int
        Number of columns in the grid
    state: list[list[int]]
        The state of the grid, a list of list such that state[i][j] is the number in the cell (i, j), i.e., in the i-th line and j-th column. 
        Note: lines are numbered 0..m and columns are numbered 0..n.
    """
    
    def __init__(self, m, n, initial_state = []):
        """
        Initializes the grid.

        Parameters: 
        -----------
        m: int
            Number of lines in the grid
        n: int
            Number of columns in the grid
        initial_state: list[list[int]]
            The intiail state of the grid. Default is empty (then the grid is created sorted).
        """
        self.m = m
        self.n = n
        if not initial_state:
            initial_state = [list(range(i*n+1, (i+1)*n+1)) for i in range(m)]            
        self.state = initial_state

    def __str__(self): 
        """
        Prints the state of the grid as text.
        """
        output = f"The grid is in the following state:\n"
        for i in range(self.m): 
            output += f"{self.state[i]}\n"
        return output

    def __repr__(self): 
        """
        Returns a representation of the grid with number of rows and columns.
        """
        return f"<grid.Grid: m={self.m}, n={self.n}>"

    def repr_grille(self):
                fig, ax = plt.subplots() # Création de la figure et des axes
                for i in range(self.m): # Parcours de chaque cellule dans la grille
                    for j in range(self.n):
                        ax.text(j, i, str(self.state[i][j]), ha='center', va='center', color='black', fontsize=12) # Placement du texte au centre de la cellule avec la valeur de la grille
                        ax.plot([j-0.5, j+0.5], [i+0.5, i+0.5], color='black')  # Dessin des lignes horizontales entre les cellules
                        ax.plot([j-0.5, j+0.5], [i-1.5, i-1.5], color='black')
                        ax.plot([j+0.5, j+0.5], [i-0.5, i+0.5], color='black')  # Dessin des lignes verticales entre les cellules
                        ax.plot([j-1.5, j-1.5], [i-0.5, i+0.5], color='black')  
                # Définition des limites des axes pour que la grille soit bien centrée
                ax.set_xlim(-0.5, self.n-0.5)
                ax.set_ylim(self.m-0.5, -0.5)
                ax.set_aspect('equal') # Aspect ratio égal pour que les cellules soient carrées
                ax.axis('off') # Désactivation des axes
                plt.show() 
                plt.savefig("grille.png") #Rien n'apparraissait avec le plt.show() seulement. J'ai donc décidé d'enregistrer l'image quand on run la méthode.

    def is_sorted_2(self):
        """
        Checks is the current state of the grid is sorte and returns the answer as a boolean.
        """
        #We check line by line if the cells of the grid are in the sorted order. As soon as result=False, the "while" stops and returns False.
        element_precedent = self.state[0][0]
        result = True
        i=0
        while result == True and i<len(self.state):
            for element in self.state[i]:
                if element_precedent > element : 
                    result = False
                else:
                    element_precedent = element
            i+=1
        return result

    def test_is_sorted(self):
        # Vérifier si la grille initiale est triée
        self.assertTrue(self.grid.is_sorted())

    def swap(self, cell1, cell2):
        """
        Implements the swap operation between two cells. Raises an exception if the swap is not allowed.

        Parameters: 
        -----------
        cell1, cell2: tuple[int]
            The two cells to swap. They must be in the format (i, j) where i is the line and j the column number of the cell. 
        """
        etat = self.state
        #We proceed to define the line and column numbers of both cells.
        i1=cell1[0]
        j1=cell1[1]
        i2=cell2[0]
        j2=cell2[1]
        if (i1==i2 and abs(j1-j2)==1) or (j1==j2 and abs(i1-i2)== 1): #the condition corresponds to the condition of a legit swap (one side apart + no border move).
            etat[i1][j1], etat[i2][j2] = etat[i2][j2], etat[i1][j1]
        else:
            raise Exception("The swap isn't valid")
        return etat

    def test_swap(self):
        self.grid.swap((0, 0), (0, 1))
        self.assertEqual(self.grid.state, [[2, 1, 3], [4, 5, 6], [7, 8, 9]]) # On vérifie si l'état de la grille a changé après l'échange

    def swap_seq(self, cell_pair_list):
        """
        Executes a sequence of swaps. 

        Parameters: 
        -----------
        cell_pair_list: list[tuple[tuple[int]]]
            List of swaps, each swap being a tuple of two cells (each cell being a tuple of integers). 
            So the format should be [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
        """
        for element in cell_pair_list:
            self.swap(element[0], element[1])
        return self.state

    def test_swap_seq(self):
        cell_pair_list = [((0, 0), (0, 1)), ((1, 1), (1, 2))]
        self.grid.swap_seq(cell_pair_list)
        self.assertEqual(self.grid.state, [[2, 3, 1], [4, 6, 5], [7, 8, 9]]) # On vérifie si la grille est dans l'état attendu après la séquence d'échanges

    def generate_possible_states(self):
        """
        Generates all possible states of the grid.

        Returns:
        --------
        states: set
            A set containing all possible states of the grid.
        """
        states = set()  # Initialize an empty set to store unique grid states
        cells = [(i, j) for i in range(self.m) for j in range(self.n)]  # Get all cell positions
        permutations_cells = permutations(cells) # Generate all permutations of cell positions
        # For each permutation, create a grid state and add it to the set of states
        for perm in permutations_cells:
            new_state = [[0 for _ in range(self.n)] for _ in range(self.m)]  # Initialize a new grid state
            for index, (i, j) in enumerate(perm):  # Assign numbers to cells based on the permutation
                new_state[i][j] = index + 1
            states.add(tuple(map(tuple, new_state)))  # Add the new state to the set of states
        return states

    def test_generate_possible_states(self):
        grid_2x2 = Grid(2, 2)
        possible_states = grid_2x2.generate_possible_states()
        self.assertEqual(len(possible_states), 24) # On vérifie si le nombre d'états possibles est correct pour une grille 2x2
   
    def find_optimal_solution(self):
        """
        Finds the optimal solution for the swap puzzle using BFS on the graph of all possible states.

        Returns:
        --------
        solution: list[tuple[tuple[int]]] | None
            The optimal solution as a list of swap pairs, where each swap pair is a tuple of two cells.
            Returns None if no solution is found.
        """
        all_states = self.generate_possible_states() # Generate all possible states of the grid
        graph = Graph(nodes=all_states) # Construct the graph of all possible states
        for state in all_states:
            # Generate valid neighbor states by performing all possible swaps
            for i in range(self.m):
                for j in range(self.n - 1):
                    neighbor_state = self.state_after_swap(state, (i, j), (i, j + 1))
                    if neighbor_state in all_states:
                        graph.add_edge(state, neighbor_state)
            for i in range(self.m - 1):
                for j in range(self.n):
                    neighbor_state = self.state_after_swap(state, (i, j), (i + 1, j))
                    if neighbor_state in all_states:
                        graph.add_edge(state, neighbor_state)
        """print(graph) #Optional but allows us to see the graph that we just created"""
        # Apply BFS to find the optimal solution
        start_state = tuple(map(tuple, self.state)) #initial state of the grid 
        """print(start_state)""" #useful when we struggled to find errors
        target_state = tuple(tuple(range(i*self.n+1, (i+1)*self.n+1)) for i in range(self.m)) #final state of the grid completely sorted
        """print(target_state)"""
        graph.plot_graph("graph3.png") #Optional but fun and practical to see the graph
        path = graph.bfs2(start_state, target_state) #bfs applied to the graph to search for the best path
        swap__seq = self.swap_seq_from_path(path)
        return path

    
    def state_after_swap(self, state, cell1, cell2):
        """
        Returns the state of the grid after performing a swap between two cells.

        Parameters:
        -----------
        state: tuple[tuple[int]]
            The current state of the grid as a tuple of tuples.
        cell1, cell2: tuple[int]
            The two cells to swap. They must be in the format (i, j) where i is the line and j the column number of the cell. 

        Returns:
        --------
        new_state: tuple[tuple[int]]
            The state of the grid after the swap operation.
        """
        # Convert state to list of lists for easier manipulation
        new_state = [list(row) for row in state]
        i1, j1 = cell1
        i2, j2 = cell2
        new_state[i1][j1], new_state[i2][j2] = new_state[i2][j2], new_state[i1][j1]
        return tuple(map(tuple, new_state))

    def test_state_after_swap(self):
        grid = Grid(3, 3)
        state = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
        new_state = grid.state_after_swap(state, (0, 0), (0, 1))
        expected_state = ((2, 1, 3), (4, 5, 6), (7, 8, 9))
        self.assertEqual(new_state, expected_state)

    def swap_seq_from_path(self,path):
        swap_seq=[]
        for i in range(len(path)-1):
            current_state = path[i]
            next_state = path[i+1]
            diff_cell_index = None
            for row in range(self.m):
                for col in range(self.n):
                    if current_state[row][col] != next_state[row][col]:
                        diff_cell_index = (row,col)
                        break
                if diff_cell_index:
                    break
            same_cell_index = None
            for row in range(self.m):
                for col in range(self.n):
                    if next_state[row][col] == current_state[diff_cell_index[0]][diff_cell_index[1]]:
                        same_cell_index = (row,col)
                        break
                if same_cell_index:
                    break
            swap_seq.append((diff_cell_index, same_cell_index))
        print("la séquence de swap est :", swap_seq)


    def find_optimal_solution_2(self):
        # Convertir l'état initial en tuple pour une utilisation dans les ensembles
        start_state = tuple(map(tuple, self.state))
        # Créer un ensemble pour garder une trace des états déjà visités
        visited = set()
        # Créer une file pour parcourir les nœuds dans l'ordre BFS
        queue = deque([(start_state, [])])
        states = [start_state]
        
        # Tant que la file n'est pas vide, continuer le BFS
        while queue:
            node, path = queue.popleft()  # Retirer le premier nœud de la file
            if node in visited:  # Vérifier si le nœud a déjà été visité
                continue
            visited.add(node)  # Marquer le nœud comme visité
            if self.is_sorted(node):  # Vérifier si le nœud est dans l'état final trié
                for i in range(len(path)):
                    states.append(path[i][1])
                swap__seq = self.swap_seq_from_path(states)
                return states
                 # Retourner la séquence de swap
            # Obtenir les voisins du nœud actuel
            neighbors = self.get_neighbors(node)
            # Ajouter les voisins non visités à la file avec le chemin mis à jour
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, path + [(node, neighbor)]))
        
        return None  # Retourner None si aucune solution n'est trouvée

    def is_sorted(self, state):
        """
        Vérifie si l'état de la grille est trié.

        Parameters:
        -----------
        state: tuple[tuple[int]]
            L'état actuel de la grille sous forme de tuple de tuples.

        Returns:
        --------
        bool:
            True si l'état de la grille est trié, False sinon.
        """
        flattened_state = [cell for row in state for cell in row]
        return flattened_state == sorted(flattened_state)

    def get_neighbors(self, state):
        """
        Obtient les voisins valides d'un état donné.

        Parameters:
        -----------
        state: tuple[tuple[int]]
            L'état actuel de la grille sous forme de tuple de tuples.

        Returns:
        --------
        neighbors: list[tuple[tuple[int]]]
            Liste des voisins valides de l'état donné.
        """
        neighbors = []
        for i in range(len(state)):
            for j in range(len(state[0])):
                for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                    if 0 <= ni < len(state) and 0 <= nj < len(state[0]):
                        new_state = self.state_after_swap(state, (i, j), (ni, nj))
                        neighbors.append(new_state)
        return neighbors
    
    def heuristic_manhattan_distance(self, state, target_state): # très efficace pour les grandes grilles : 5x5
        # Calculate the Manhattan distance between current state and target state
        distance = 0
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] != target_state[i][j]:
                    target_i, target_j = divmod(state[i][j] - 1, len(state[0]))
                    distance += abs(i - target_i) + abs(j - target_j)
        return distance

    def find_optimal_solution_astar(self):
        start_state = tuple(map(tuple, self.state))  # Convert the initial state to a tuple for use in sets
        target_state = tuple(tuple(range(i*self.n+1, (i+1)*self.n+1)) for i in range(self.m))  # Convert the target state to a tuple for use in sets
        open_set = [(self.heuristic_manhattan_distance(start_state, target_state), 0, start_state, [])]  # Initialize the priority queue with the initial state
        heapq.heapify(open_set)  # Turn the list into a heap to obtain a priority queue
        closed_set = set()  # Set to keep track of visited states
        
        while open_set:
            _, cost, current_state, path = heapq.heappop(open_set)  # Extract the node with the lowest cost from the priority queue
            if current_state in closed_set:  # Check if the node has already been visited
                continue
            closed_set.add(current_state)  # Add the node to the set of visited nodes
            
            if current_state == target_state:  # Check if the current state is the target state
                print("la séquence de swaps est : ",self.swaps_seq_from_path_2(start_state, current_state))
                return self.swaps_seq_from_path_2(start_state, current_state)  # Return the sequence of swaps if the target state is reached
                break

            neighbors = self.get_neighbors(current_state)  # Get valid neighbors of the current node
            for neighbor in neighbors:
                if neighbor not in closed_set:  # Check if the neighbor has not been visited
                    g = cost + 1  # Cost to reach this neighbor
                    h = self.heuristic_manhattan_distance(neighbor, target_state)  # Heuristic value of the neighbor
                    f = g + h  # Total cost (actual cost + heuristic value)
                    heapq.heappush(open_set, (f, g, neighbor, path + [(current_state, neighbor)]))  # Add the neighbor to the priority queue with its total cost and updated path
        return None  # Return None if no solution is found

    def swaps_seq_from_path_2(self, state1, state2): #j'ai fait une deuxième méthode swaps_seq_from_path car la première ne marchait pas avec le path que retourne l'algorithme de A* 
        swaps = []  # Initialize list to store swaps
        cell_map = {}  # Dictionary to map cell values to their positions
        # Create a mapping of cell values to their positions in state1
        for i in range(len(state1)):
            for j in range(len(state1[0])):
                cell_value = state1[i][j]
                cell_map[cell_value] = (i, j)
        # Compare state2 with state1 and find cells that need to be swapped
        for i in range(len(state2)):
            for j in range(len(state2[0])):
                cell_value = state2[i][j]
                if cell_value in cell_map:
                    initial_cell = cell_map[cell_value]
                    if initial_cell != (i, j):
                        swaps.append((initial_cell, (i, j)))  # Add swap to the list

        return swaps

    # j'ai fait le code de trois autres heuristiques : une qui comptabilise le nombre de carreaux mal placés, une avec la distance de hamming, et une combinaison linéaire d'heuristiques 

    def heuristic_misplaced_tiles(self, state, target_state): # peu efficace pour les grandes grilles
        misplaced_tiles = 0
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] != target_state[i][j]:
                    misplaced_tiles += 1
        return misplaced_tiles

    def heuristic_hamming_distance(self, state, target_state): # peu efficace pour les grandes grilles également
        hamming_distance = 0
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] != target_state[i][j]:
                    hamming_distance += 1
        return hamming_distance

    def heuristic_linear_combination(self, state, target_state): # très efficace : logique car gros coefficient sur 
        manhattan_distance = self.heuristic_manhattan_distance(state, target_state)
        misplaced_tiles = self.heuristic_misplaced_tiles(state, target_state)
        linear_combination = 0.8*manhattan_distance + 0.2*misplaced_tiles
        return linear_combination
    

    @classmethod
    def grid_from_file(cls, file_name): 
        """
        Creates a grid object from class Grid, initialized with the information from the file file_name.
        
        Parameters: 
        -----------
        file_name: str
            Name of the file to load. The file must be of the format: 
            - first line contains "m n" 
            - next m lines contain n integers that represent the state of the corresponding cell

        Output: 
        -------
        grid: Grid
            The grid
        """
        with open(file_name, "r") as file:
            m, n = map(int, file.readline().split())
            initial_state = [[] for i_line in range(m)]
            for i_line in range(m):
                line_state = list(map(int, file.readline().split()))
                if len(line_state) != n: 
                    raise Exception("Format incorrect")
                initial_state[i_line] = line_state
            grid = Grid(m, n, initial_state)
        return grid

#tests

grille = Grid(2,3,[[3,5,6],[2,1,4]])


"""print(grille)
grille2 = grille.swap((2,1),(2,2))
print(grille2)
grille3 = grille.swap_seq([((2,1),(2,2)),((1,0),(1,1))])
print(grille3)
grille4= Grid(3,3, [])
print(grille.is_sorted())

print(grille)
print(grille.get_solution())
grille_test = grille.swap_seq(grille.get_solution())
print('hello2')
print(grille_test)

grille_6 = Grid(3,4,[[3,5,6,12],[2,4,7,1],[11,10,8,9]])
print(grille_6)
grille_test6 = grille_6.swap_seq(grille_6.get_solution())
print(grille_6.get_solution())
print(grille_test6)
"""

"""possible_states = grille.generate_possible_states()
print("Les états possibles sont :",  possible_states)"""

"""optimal_solution = grille.find_optimal_solution_astar()"""
