import numpy as np
import copy
import random
import signal
import sys
import time
from threading import Thread
from parse import parse_input_file, parse_input
from graph import Graph


class RandApprox:
	def __init__(self, graph):
		self.graph = graph

	def randomized_approximation(self):
		vertices = self.graph.get_vertices()
		clusters = []
		while len(vertices) > 0:
			u = vertices[int(len(vertices)*random.random())]
			n_u = list(self.graph.get_neighbors(u))
			cluster = n_u + [u]
			clusters.append(cluster)
			for c in cluster:
				self.graph.del_vertex(c)
			vertices = self.graph.get_vertices()
		solution = self.transitive_closure(clusters)
		return clusters, solution

	def transitive_closure(self, clusters):
		edges = np.zeros(self.graph._adj_matrix.shape, dtype=bool)
		for c_i in range(len(clusters)):
			for c_j in range(c_i+1, len(clusters)):
				for i in clusters[c_i]:
					for j in clusters[c_j]:
						if self.graph._adj_matrix[i][j]:
							edges[i][j] = edges[j][i] = True
		for cluster in clusters:
			for i in range(len(cluster)):
				for j in range(i+1, len(cluster)):
					if not self.graph._adj_matrix[cluster[i]][cluster[j]]:
						edges[cluster[i]][cluster[j]] = edges[cluster[j]][cluster[i]] = True
		return edges


# class Greedy(Thread):
# 	def __init__(self, graph):
# 		super().__init__()
# 		self.graph = graph
# 		self.solution = None
# 		self.cost = np.Inf

# 	def run(self):
# 		self.solution, self.cost = self.greedy_sol()
class Greedy:
	def __init__(self, graph):
		self.graph = graph

	def greedy_sol(self):
		edges, total_cost = [], 0
		for component in self.graph.connected_components():
			subgraph = self.graph.induced_subgraph(component)
			scores = self.edge_scores(subgraph)
			deletions, del_cost = self.greedy(copy.deepcopy(subgraph), scores)
			edges_comp = self.transitive_closure_del(subgraph, deletions)
			edges += [(component[i], component[j]) for i, j in edges_comp]
			total_cost += del_cost
		return edges, total_cost

	def greedy(self, graph, scores):
		cost = self.transitive_closure_cost(graph)
		if cost == 0:
			return [], 0
		deletions, del_cost = [], 0
		components = graph.connected_components()
		while len(components) == 1:
			u, v = self.remove_culprit(graph, scores)
			deletions.append((u, v))
			del_cost += graph.weights[u][v]
			if del_cost >= cost:
				return [], cost
			components = graph.connected_components()
		del_cost = self.adjust_deletions(graph, components[0], components[1], deletions, del_cost)
		graph_1 = graph.induced_subgraph(components[0])
		graph_2 = graph.induced_subgraph(components[1])
		scores1 = scores[components[0],:][:,components[0]]
		list_1, cost_1 = self.greedy(graph_1, scores1)
		if del_cost + cost_1 >= cost:
			return [], cost
		scores2 = scores[components[1],:][:,components[1]]
		list_2, cost_2 = self.greedy(graph_2, scores2)
		if del_cost + cost_1 + cost_2 >= cost:
			return [], cost
		list_1 = [(components[0][i], components[0][j]) for i, j in list_1]
		list_2 = [(components[1][i], components[1][j]) for i, j in list_2]

		return deletions + list_1 + list_2, del_cost + cost_1 + cost_2

	def edge_scores(self, graph):
		scores = np.zeros(graph.weights.shape)
		for u, v in graph.get_edges():
			deviation = 0
			for w in graph.common_neighbors(u, v):
				deviation -= min(graph.weights[u][v], abs(graph.weights[v][w]), abs(graph.weights[u][w]))
			for w in graph.diff_neighbors(u, v):
				deviation += min(graph.weights[u][v], abs(graph.weights[v][w]), abs(graph.weights[u][w]))
			deviation -= graph.weights[u][v]
			scores[u][v] = scores[v][u] = deviation
		return scores

	def update_edge_scores(self, scores, graph, u, v):
		for w in graph.single_neighbors(v, u):
			diff = min(abs(graph.weights[u][v]), abs(graph.weights[v][w]), abs(graph.weights[u][w]))
			scores[v][w] -= diff
			scores[w][v] -= diff
		for w in graph.single_neighbors(u, v):
			diff = min(abs(graph.weights[u][v]), abs(graph.weights[v][w]), abs(graph.weights[u][w]))
			scores[u][w] -= diff
			scores[w][u] -= diff
		for w in graph.common_neighbors(u, v):
			diff = min(abs(graph.weights[u][v]), abs(graph.weights[v][w]), abs(graph.weights[u][w]))
			scores[u][w] += diff
			scores[w][u] += diff
			scores[v][w] += diff
			scores[w][v] += diff

	def edge_highest_score(self, graph, scores):
		edge, max_score = None, -float('inf')
		for u, v in graph.get_edges():
			if scores[u][v] > max_score:
				edge, max_score = (u, v), scores[u][v]
		return edge

	def remove_culprit(self, graph, scores):
		u, v = self.edge_highest_score(graph, scores)
		self.update_edge_scores(scores, graph, u, v)
		graph.del_edge(u, v)
		return (u, v)


	def transitive_closure_cost(self, graph):
		return -np.sum(graph.weights[np.triu(np.invert(graph._adj_matrix))])

	def transitive_closure_del(self, graph, deletions):
		edges = []
		for u, v in deletions:
			graph.del_edge(u, v)
		for component in graph.connected_components():
			for i in range(len(component)):
				for j in range(i+1, len(component)):
					if not graph._adj_matrix[component[i]][component[j]]:
						edges.append((component[i], component[j]))
		return deletions + edges

	def adjust_deletions(self, graph, component_1, component_2, deletions, del_cost):
		for u, v in copy.copy(deletions):
			if (u in component_1 and v in component_1) or (u in component_2 and v in component_2):
				graph.add_edge(u, v)
				deletions.remove((u, v))
				del_cost -= graph.weights[u][v]
		return del_cost



class LocalSearch(Thread):
	def __init__(self, graph, solution, clusters):
		super().__init__()
		self.graph = graph
		self.solution = solution
		self.clusters = clusters

	def run(self):
		self.local_search()

	def terminate(self):
		self.sigint = True

	def merge_clusters(self, c_i, c_j):
		edges_del = [(i, j) for i in self.clusters[c_i] for j in self.clusters[c_j] if self.graph._adj_matrix[i][j]]
		edges_add = [(i, j) for i in self.clusters[c_i] for j in self.clusters[c_j] if not self.graph._adj_matrix[i][j]]
		cost_cluster = sum([self.graph.weights[i][j] for i, j in edges_del])
		cost_merge = sum([-self.graph.weights[i][j] for i, j in edges_add])
		if cost_merge < cost_cluster:
			for i, j in edges_del:
				self.solution[i][j] = self.solution[j][i] = False
			for i, j in edges_add:
				self.solution[i][j] = self.solution[j][i] = True
			self.clusters[c_i] += self.clusters[c_j]
			del self.clusters[c_j]
			return True
		return False

	def worth_move_vertex(self, v, c_i, c_j):
		cost_cluster_i = cost_cluster_j = 0
		for i in self.clusters[c_i]:
			if self.graph._adj_matrix[v][i]:
				cost_cluster_j += self.graph.weights[v][i]
			else:
				cost_cluster_i += -self.graph.weights[v][i]
		for j in self.clusters[c_j]:
			if self.graph._adj_matrix[v][j]:
				cost_cluster_i += self.graph.weights[v][j]
			else:
				cost_cluster_j += -self.graph.weights[v][j]
		return cost_cluster_j < cost_cluster_i


	def move_vertex(self, v, c_i, c_j):
		for i in self.clusters[c_i]:
			if self.graph._adj_matrix[v][i]:
				self.solution[v][i] = self.solution[i][v] = True
			else:
				self.solution[v][i] = self.solution[i][v] = False
		if len(self.clusters[c_i]) == 1:
			del self.clusters[c_i]
			c_j -= 1
		else:
			self.clusters[c_i].remove(v)
		for j in self.clusters[c_j]:
			if self.graph._adj_matrix[v][j]:
				self.solution[v][j] = self.solution[j][v] = False
			else:
				self.solution[v][j] = self.solution[j][v] = True
		self.clusters[c_j].append(v)

	def local_search(self):
		self.sigint = False
		n = self.graph.weights.shape[0]
		while not self.sigint:
			c_i = int((len(self.clusters)-1)*random.random())
			v = self.clusters[c_i][int((len(self.clusters[c_i])-1)*random.random())]
			for c_j in range(len(self.clusters)):
				if c_j != c_i and self.worth_move_vertex(v, c_i, c_j):
					self.move_vertex(v, c_i, c_j)
					break

	def get_solution(self):
		edges = np.where(np.triu(self.solution))
		return zip(edges[0], edges[1])


def upper_bound(weights, time):
	graph = Graph(weights)
	rand_solver = RandApprox(graph)
	clusters, sol_rand = rand_solver.randomized_approximation()

	local_search_rand = LocalSearch(graph, sol_rand, clusters)
	local_search_rand.start()

	greedy_solver = Greedy(Graph(weights))
	greedy_solver.start()
	greedy_solver.join(time)

	local_search_rand.terminate()
	local_search_rand.join()

	# cost_greedy = sum([abs(graph.weights[v][w]) for v, w in greedy_solver.get_solution()])
	cost_greedy = greedy_solver.cost
	cost_rand = sum([abs(graph.weights[v][w]) for v, w in local_search_rand.get_solution()])

	return min(cost_greedy, cost_rand)


def upper_bound2(weights, time):
	graph = Graph(weights)
	rand_solver = RandApprox(graph)
	clusters, sol_rand = rand_solver.randomized_approximation()

	local_search_rand = LocalSearch(graph, sol_rand, clusters)
	local_search_rand.start()

	greedy_solver = Greedy(Graph(weights))
	greedy_solver.start()
	greedy_solver.join(time)

	local_search_rand.terminate()
	local_search_rand.join()

	cost_greedy = greedy_solver.cost
	cost_rand = sum([abs(graph.weights[v][w]) for v, w in local_search_rand.get_solution()])

	if cost_rand < cost_greedy:
		return local_search_rand.get_solution()
	else:
		return greedy_solver.solution


# if __name__ == "__main__":
#     # weights = parse_input_file(sys.argv[1])
#     weights = parse_input()
#     graph = Graph(weights)
#     rand_solver = RandApprox(graph)
#     clusters, sol_rand = rand_solver.randomized_approximation()

#     local_search_rand = LocalSearch(graph, sol_rand, clusters)
#     local_search_rand.start()
#     local_search_greedy = None

#     def handler(signum, frame):
#         local_search_rand.terminate()
#         local_search_rand.join()
#         if local_search_greedy is not None:
#             local_search_greedy.terminate()
#             local_search_greedy.join()
#             cost_greedy = sum([abs(graph.weights[v][w]) for v, w in local_search_greedy.get_solution()])
#             cost_rand = sum([abs(graph.weights[v][w]) for v, w in local_search_rand.get_solution()])
#             if cost_greedy < cost_rand:
#                 solution = local_search_greedy.get_solution()
#             else:
#                 solution = local_search_rand.get_solution()
#         else:
#             solution = local_search_rand.get_solution()
#         k = 0
#         for v, w in solution:
#             k += abs(graph.weights[v][w])
#             print("%d %d" % (v+1, w+1))
#         print("#k: %d" % (k))
#         exit(0)

#     signal.signal(signal.SIGINT, handler)

#     greedy_solver = Greedy(Graph(weights))
#     solution, cost = greedy_solver.greedy_sol()

#     # clusters = graph.connected_components()
#     # local_search_greedy = LocalSearch(graph, solution, clusters)
#     # local_search_greedy.start()
#     # local_search_greedy.join()
#     local_search_rand.terminate()
#     local_search_rand.join()
#     # cost_greedy = sum([abs(graph.weights[v][w]) for v, w in local_search_greedy.get_solution()])
#     cost_greedy = sum([abs(graph.weights[v][w]) for v, w in solution])
#     cost_rand = sum([abs(graph.weights[v][w]) for v, w in local_search_rand.get_solution()])
#     # if cost_greedy < cost_rand:
#     # 	solution = local_search_greedy.get_solution()
#     # else:
#     # 	solution = local_search_rand.get_solution()
#     if cost_rand < cost_greedy:
#     	solution = local_search_rand.get_solution()
#     k = 0
#     for v, w in solution:
#         k += abs(graph.weights[v][w])
#         print("%d %d" % (v+1, w+1))
#     print("#k: %d" % (k))

if __name__ == "__main__":
    # weights = parse_input_file(sys.argv[1])
    weights = parse_input()

    # solution = upper_bound2(weights, 60)
    graph = Graph(weights)

    greedy_solver = Greedy(graph)
    solution, cost = greedy_solver.greedy_sol()

    k = 0
    for v, w in solution:
        k += abs(weights[v][w])
        print("%d %d" % (v+1, w+1))
    print("#k: %d" % (k))


