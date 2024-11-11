
from grid import Grid
from graph import Graph
from solver import Solver

g = Grid(2, 3, [[4,3,6],[1,2,5]])
g2 = Grid(2, 3, [[4,3,6],[1,2,5]])
g3 = Grid(3, 3, [[4,9,6],[1,2,5],[3,7,8]])
g4 = Grid(5,5,[[3,4,6,11,25],[2,1,5,12,21],[9,18,7,8,16],[15,10,13,24,14],[19,22,20,23,17] ])
g5 = Graph.graph_from_file("input/graph1.in")

"""
data_path = "../input/"
file_name = data_path + "grid0.in"

print(file_name)

g = Grid.grid_from_file(file_name)
print(g) """

# Question 3 :
print("voici la méthode naive de résolution d’une grille 2x3: ")
solver = Solver(g)
solver.get_solution()

print("")

# Question 5 :
print("voici l'utilisation du BFS pour trouver un plus court chemin dans un graph:")
solver5 = Solver(g5)
shortest_path = solver5.bfs2(1, 19)
print("Plus court chemin de", 1, "à", 19, ":", shortest_path)

print("")

#Question 5-6-7 : 
print("voici la résolution en utilisant la méthode BFS d'une grille 2x3: ")
solver2 = Solver(g2)
solver2.find_optimal_solution()

print("")

#Question 8:
print("voici la résolution en utilisant la méthode BFS optimisée d'une grille 3x3: ")
solver3 = Solver(g3)
solver3.find_optimal_solution_2()

print("")

#Algorithme A*:
print("voici la résolution en utilisant la méthode A* et l'heuristique de la distance Manhattan d'une grille 5x5: ")
solver4 = Solver(g4)
solver4.find_optimal_solution_astar()


 

"""
# à modifier
data_path = "../input/"
file_name_2 = data_path + "graph1.in"
graph = Graph.graph_from_file(file_name_2)
print(graph) """
