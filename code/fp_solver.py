import numpy as np
import copy
import heapq

from heuristic import Greedy, upper_bound
from graph_test import Graph
from ilp_solver import LPSolver
import reductionrules as rr


class FPSolver:
	def __init__(self, graph, params):
		self.graph = graph
		self.params = params

	def solve(self):
		# max_ub_time = 60
		# ub = upper_bound(self.graph.weights, max_ub_time)
		# _, ub = Greedy(Graph(self.graph.weights)).greedy_sol()

		solution = np.zeros(self.graph.weights.shape, dtype=bool)
		p3s = self.graph.get_sorted_p3s()
		edges = [(-self.graph.weights[u][v], (u, v)) for u, v in self.graph.get_edges()]
		# print(edges)
		heapq.heapify(edges)
		# print(edges)
		# print(heapq.heappop(edges))
		cost, solution, rec_steps = self.branch(self.graph, solution, p3s, edges, 0, np.Inf, 0)

		return self.get_solution(solution), rec_steps


	def branch(self, graph, solution, p3s, sorted_edges, cost, ub, depth):
		if len(p3s) == 0:
			return 0, solution, 1
		if depth % int(self.params[0]) == 0:
			_, heuristic = Greedy(Graph(graph.weights)).greedy_sol()
			ub = min(cost + heuristic, ub)
			if cost > ub:
				return ub, None, 1
		if cost > ub:
			return ub, None, 1
		if (depth+1) % int(self.params[1]) == 0:
			edge_disjoint_p3s = graph.edge_disjoint_p3s(p3s)
			if cost + self.lb2(edge_disjoint_p3s) > ub:
				return ub, None, 1
		if (depth+1) % self.params[2] == 0:
			lp = LPSolver(graph)
			lp.setup()
			lb = lp.solve()
			if cost + lb > ub:
				return ub, None, 1

		merged, merge_cost = [], 0
		# if depth % self.params[3] == 0:
		# 	merged_rr, d = rr.high_weight_exhaustive(graph, sorted_edges, ub-cost)
		# 	merge_cost += d
		# 	cost += d
		# 	merged = merged_rr + merged
		# 	if cost > ub:
		# 		for u, v in merged:
		# 			graph.unmerge(u, v)
		# 		return ub, None, 1
		# 	if len(merged) > 0:
		# 		p3s = graph.get_sorted_p3s()
		# 		if len(p3s) == 0:
		# 			solution = np.zeros(graph.weights.shape, dtype=bool)
		# 			self.reconstruct_solution(graph, merged, solution)
		# 			return merge_cost, solution, 1

		if depth % int(self.params[4]) == 0:
			edges = rr.heavy_non_edge(graph)
			for u, v in edges:
				graph.weights[u][v] -= 65535
				graph.weights[v][u] -= 65535
				graph.del_edge(u, v)

		if depth % self.params[5] == 0:
			components = graph.connected_components()
			if len(components) > 1:
				solution, rec_steps, c = np.zeros(graph.weights.shape, dtype=bool), 0, 0
				for component in components:
					subgraph = graph.induced_subgraph(component)
					sub_solution = np.zeros(subgraph.weights.shape, dtype=bool)
					p3s = subgraph.get_sorted_p3s()
					sol, sub_solution, steps = self.branch(subgraph, sub_solution, p3s, sorted_edges, cost + c, ub, depth+1)
					rec_steps += steps
					if sub_solution is None:
						for u, v in merged:
							graph.unmerge(u, v)
						return ub, None, rec_steps + 1
					c += sol
					if cost + c > ub:
						for u, v in merged:
							graph.unmerge(u, v)
						return ub, None, rec_steps + 1
					for i in range(sub_solution.shape[0]):
						for j in range(i+1, sub_solution.shape[0]):
							if sub_solution[i][j]:
								solution[component[i]][component[j]] = solution[component[j]][component[i]] = True
				self.reconstruct_solution(graph, merged, solution)
				return merge_cost + c, solution, rec_steps + 1

		# todo: improve lb1 with connected components?!
		# if depth % self.params[6] == 0:
		# 	if cost + self.lb1(p3s) > ub:
		# 		return ub, None, 1

		if (depth+1) % self.params[6] == 0:
			merged_rr, d = rr.heavy_edge_single_end_exhaustive(graph)
			merged = merged_rr + merged
			merge_cost += d
			cost += d
			if cost > ub:
				for u, v in merged:
					graph.unmerge(u, v)
				return ub, None, 1
			if len(merged_rr) > 0:
				p3s = graph.get_sorted_p3s()
				if len(p3s) == 0:
					solution = np.zeros(graph.weights.shape, dtype=bool)
					self.reconstruct_solution(graph, merged, solution)
					return merge_cost, solution, 1

		if (depth+1) % self.params[7] == 0:
			merged_rr, d = rr.heavy_edge_both_ends_exhaustive(graph)
			merged = merged_rr + merged
			merge_cost += d
			cost += d
			if cost > ub:
				for u, v in merged:
					graph.unmerge(u, v)
				return ub, None, 1
			if len(merged_rr) > 0:
				p3s = graph.get_sorted_p3s()
				if len(p3s) == 0:
					solution = np.zeros(graph.weights.shape, dtype=bool)
					self.reconstruct_solution(graph, merged, solution)
					return merge_cost, solution, 1

		# todo: param for branching heuristic
		# todo: update p3s when merging for new branching heuristic -> "deviation"
		# u, v = self.get_max_p3_edge(graph, p3s)

		u, v, w = p3s[0].data
		if graph.weights[w][v] > graph.weights[u][v]:
			u = w

		cost1, cost2 = graph.merge_cost(u, v), graph.weights[u][v]
		if cost1 < cost2:
			graph.merge(u, v)
			solution1 = np.zeros(graph.weights.shape, dtype=bool)
			p3s1 = graph.get_sorted_p3s() # todo: update p3s when merging
			sol1, solution1, rec_steps = self.branch(graph, solution1, p3s1, sorted_edges, cost + cost1, ub, depth+1)
			graph.unmerge(u, v)
			ub = min(cost + cost1 + sol1, ub)

			p3s2 = graph.update_p3s(p3s, u, v)
			graph.weights[u][v] -= 65535
			graph.weights[v][u] -= 65535
			graph.del_edge(u, v)
			solution2 = np.zeros(graph.weights.shape, dtype=bool)
			sol2, solution2, steps = self.branch(graph, solution2, p3s2, sorted_edges, cost + cost2, ub, depth+1)
			rec_steps += steps
			graph.weights[u][v] = graph.weights[v][u] = cost2
			graph.add_edge(u, v)
		else:
			p3s2 = graph.update_p3s(p3s, u, v)
			graph.weights[u][v] -= 65535
			graph.weights[v][u] -= 65535
			graph.del_edge(u, v)
			solution2 = np.zeros(graph.weights.shape, dtype=bool)
			sol2, solution2, rec_steps = self.branch(graph, solution2, p3s2, sorted_edges, cost + cost2, ub, depth+1)
			graph.weights[u][v] = graph.weights[v][u] = cost2
			graph.add_edge(u, v)
			ub = min(cost + cost2 + sol2, ub)

			graph.merge(u, v)
			solution1 = np.zeros(graph.weights.shape, dtype=bool)
			p3s1 = graph.get_sorted_p3s() # todo: update p3s when merging
			sol1, solution1, steps = self.branch(graph, solution1, p3s1, sorted_edges, cost + cost1, ub, depth+1)
			graph.unmerge(u, v)
			rec_steps += steps

		if solution1 is None and solution2 is None:
			for u, v in merged:
				graph.unmerge(u, v)
			return ub, None, rec_steps + 1
		if cost1 + sol1 < cost2 + sol2:
			self.update_solution(graph, u, v, solution1)
			self.reconstruct_solution(graph, merged, solution1)
			return merge_cost + cost1 + sol1, solution1, rec_steps + 1
		else:
			solution2[u][v] = solution2[v][u] = True
			self.reconstruct_solution(graph, merged, solution2)
			return merge_cost + cost2 + sol2, solution2, rec_steps + 1


	def reconstruct_solution(self, graph, merged, solution):
		for u, v in merged:
			graph.unmerge(u, v)
			self.update_solution(graph, u, v, solution)

	def update_solution(self, graph, u, v, solution):
		for w in range(graph.weights.shape[0]):
			# if w != u and w != v:
			if w != u and w != v and graph.mask[w]:
				if graph.weights[u][w] == 0 and graph.weights[v][w] == 0:
					if solution[u][w]:
						solution[v][w] = solution[w][v] = True
				elif graph.weights[u][w] + graph.weights[v][w] == 0:
					if solution[u][w]:
						if graph.weights[v][w] < 0:
							solution[u][w] = solution[w][u] = False
							solution[v][w] = solution[w][v] = True
					else:
						if graph.weights[v][w] > 0:
							solution[v][w] = solution[w][v] = True
						else:
							solution[u][w] = solution[w][u] = True
				elif self.sgn(graph, u, w) != self.sgn(graph, v, w):
					if solution[u][w]:
						if abs(graph.weights[u][w]) < abs(graph.weights[v][w]):
							solution[u][w] = solution[w][u] = False
							solution[v][w] = solution[w][v] = True
					else:
						if abs(graph.weights[u][w]) > abs(graph.weights[v][w]):
							solution[v][w] = solution[w][v] = True
						else:
							solution[u][w] = solution[w][u] = True
				else:
					if solution[u][w]:
						solution[v][w] = solution[w][v] = True


	def get_solution(self, solution):
		edges = np.where(np.triu(solution))
		return zip(edges[0], edges[1])

	
	def sgn(self, graph, u, v):
		return -1 if graph.weights[u][v] <= 0 else 1
		

	def cluster_editing(self):
		rec_steps = 0
		p3s = self.graph.get_sorted_p3s()
		if len(p3s) == 0:
			return [], rec_steps
		edge_disjoint_p3s = self.graph.edge_disjoint_p3s(p3s)
		k = max(self.lb1(p3s), self.lb2(edge_disjoint_p3s))
		edited_edges = np.zeros(self.graph.weights.shape, dtype=bool)
		permanent_edges = np.zeros(self.graph.weights.shape, dtype=bool)
		while True:
			solution, steps = self.ce_branch(self.graph, edited_edges, permanent_edges, p3s, k, 0)
			rec_steps += steps
			if solution is not None:
				return (solution, rec_steps)
			k += 1

	def ce_branch(self, graph, edited_edges, permanent_edges, p3s, k, rec_steps):
		if k < 0:
			return None, rec_steps + 1
		if len(p3s) == 0:
			return [], rec_steps + 1

		edge_disjoint_p3s = graph.edge_disjoint_p3s(p3s)
		if k < self.lb2(edge_disjoint_p3s):
			return None, rec_steps + 1
		# if k < self.lb1(p3s, p3_edges, t):
		# 	return None, rec_steps

		u, v, w = p3s[0].data

		sorted_p3 = sorted([(u, v), (v, w), (u, w)], key=lambda e: abs(graph.weights[e[0]][e[1]]), reverse=True)
		for i, j in sorted_p3:
			if not edited_edges[i][j] and not permanent_edges[i][j]:
				edited_edges[i][j] = edited_edges[j][i] = True
				if graph._adj_matrix[i][j]:
					graph.del_edge(i, j)
					updated_p3s = graph.update_p3s(p3s, i, j)
					solution, steps = self.ce_branch(graph, edited_edges, permanent_edges, updated_p3s, k - graph.weights[i][j], 0)
					rec_steps += steps
					if solution is not None:
						return solution + [(i, j)], rec_steps + 1
					graph.add_edge(i, j)
				else:
					graph.add_edge(i, j)
					updated_p3s = graph.update_p3s(p3s, i, j)
					solution, steps = self.ce_branch(graph, edited_edges, permanent_edges, updated_p3s, k + graph.weights[i][j], 0)
					rec_steps += steps
					if solution is not None:
						return solution + [(i, j)], rec_steps + 1
					graph.del_edge(i, j)
				edited_edges[i][j] = edited_edges[j][i] = False
				permanent_edges[i][j] = permanent_edges[j][i] = True

		permanent_edges[u][v] = permanent_edges[v][u] = False
		permanent_edges[v][w] = permanent_edges[w][v] = False
		permanent_edges[u][w] = permanent_edges[w][u] = False
		return None, rec_steps + 1

	def get_max_p3_edge(self, graph, p3s):
		t = np.zeros(graph.weights.shape)
		p3_edges = np.zeros(graph.weights.shape, dtype=bool)
		for p3 in p3s:
			u, v, w = p3.data
			t[u][v] += p3.min_edge_weight
			t[v][u] += p3.min_edge_weight
			t[v][w] += p3.min_edge_weight
			t[w][v] += p3.min_edge_weight
			p3_edges[u][v] = p3_edges[v][u] = True
			p3_edges[w][v] = p3_edges[v][w] = True
		n, max_value, edge = graph.weights.shape[0], np.NINF, None
		for i in range(n):
			for j in range(i+1, n):
				if p3_edges[i][j]:
					# t[i][j] -= graph.weights[i][j]
					if t[i][j] > max_value:
						max_value, edge = t[i][j], (i, j)
		return edge

	def get_max_p3(self, p3s):
		max_value = -1
		for p3 in p3s:
			if p3.min_edge_weight > max_value:
				max_p3, max_value = p3, p3.min_edge_weight
		return max_p3


	def lb1(self, p3s):
		t = np.zeros(self.graph.weights.shape)
		for p3 in p3s:
			u, v, w = p3.data
			t[u][v] += 1
			t[v][w] += 1
			t[u][w] += 1
			t[v][u] += 1
			t[w][v] += 1
			t[w][u] += 1
		p3_edges = np.where(np.triu(t > 0))
		if len(p3_edges[0]) == 0:
			return -1
		return len(p3s)*min([abs(self.graph.weights[u][v])/t[u][v] for u, v in zip(p3_edges[0], p3_edges[1])])

	def lb2(self, edge_disjoint_p3s):
		return sum([p3.min_edge_weight for p3 in edge_disjoint_p3s])



