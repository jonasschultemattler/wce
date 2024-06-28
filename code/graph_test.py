import numpy as np
import copy

class Graph:
	def __init__(self, weights):
		self.weights = weights
		self._adj_matrix = weights > 0

	def add_edge(self, u, v):
		self._adj_matrix[u][v] = self._adj_matrix[v][u] = True

	def del_edge(self, u, v):
		self._adj_matrix[u][v] = self._adj_matrix[v][u] = False

	def get_edges(self):
		edges = np.where(np.triu(self._adj_matrix))
		return zip(edges[0], edges[1])

	def get_neighbors(self, v):
		return np.where(self._adj_matrix[v])[0]

	def common_neighbors(self, v, w):
		return np.where(self._adj_matrix[v] & self._adj_matrix[w])[0]

	def all_neighbors(self, v, w):
		union = self._adj_matrix[v] | self._adj_matrix[w]
		union[v] = union[w] = False
		return np.where(union)[0]

	def single_neighbors(self, v, w):
		# returns neighbors of v that are not adjacent to w
		single_neighbors = self._adj_matrix[v] & np.invert(self._adj_matrix[w])
		single_neighbors[w] = False
		return np.where(single_neighbors)[0]

	def diff_neighbors(self, v, w):
		return np.where(self._adj_matrix[v] ^ self._adj_matrix[w])[0]

	def degree(self, v):
		return len(self.get_neighbors(v))

	def vertex_degrees(self):
		return [self.degree(v) for v in range(n)]

	def is_cluster(self):
		return not self.contains_p3()

	def contains_p3(self):
		return self.get_p3() is not None

	def get_p3(self):
		# O(n^3)
		# TODO O(n + m)
		for u in range(self.n):
			for v in range(u+1, self.n):
				for w in range(v+1, self.n):
					if graph._adj_matrix[u][v] and graph._adj_matrix[v][w] and not graph._adj_matrix[u][w]:
						return (u, v, w)
					if graph._adj_matrix[v][w] and graph._adj_matrix[w][u] and not graph._adj_matrix[u][v]:
						return (v, w, u)
					if graph._adj_matrix[w][u] and graph._adj_matrix[u][v] and not graph._adj_matrix[v][w]:
						return (w, u, v)

	def list_p3s(self, marked_edges):
		p3s = set()
		for u, v in self.get_edges():
			for w in self.single_neighbors(u, v):
				p3s.add(self.P3(w, u, v, self.weights, marked_edges))
			for w in self.single_neighbors(v, u):
				p3s.add(self.P3(u, v, w, self.weights, marked_edges))
		return p3s

	def sort_p3s(self, marked_edges):
		return sorted(self.list_p3s(marked_edges), reverse=True)


	def get_p3s(self):
		p3s = []
		for u, v in self.get_edges():
			for w in self.single_neighbors(u, v):
				p3s.append(self.P3_2(w, u, v, self.weights))
			for w in self.single_neighbors(v, u):
				p3s.append(self.P3_2(u, v, w, self.weights))
		return p3s

	def get_sorted_p3s(self):
		return sorted(self.get_p3s(), reverse=True)

	def update_p3s(self, p3s, u, v):
		updated_p3s = set(p3s)
		if self.weights[u][v] > 0:
			for w in self.common_neighbors(u, v):
				updated_p3s.add(self.P3_2(u, w, v, self.weights))
			for w in self.single_neighbors(u, v):
				updated_p3s.remove(self.P3_2(w, u, v, self.weights))
			for w in self.single_neighbors(v, u):
				updated_p3s.remove(self.P3_2(w, v, u, self.weights))
		else:
			for w in self.common_neighbors(u, v):
				updated_p3s.remove(self.P3_2(u, w, v, self.weights))
			for w in self.single_neighbors(u, v):
				updated_p3s.add(self.P3_2(w, u, v, self.weights))
			for w in self.single_neighbors(v, u):
				updated_p3s.add(self.P3_2(u, v, w, self.weights))
		return sorted(updated_p3s, reverse=True)

	def update_p3s_del(self, p3s, u, v):
		for w in self.common_neighbors(u, v):
			updated_p3s.add(self.P3_2(u, w, v, self.weights))
		for w in self.single_neighbors(u, v):
			updated_p3s.remove(self.P3_2(w, u, v, self.weights))
		for w in self.single_neighbors(v, u):
			updated_p3s.remove(self.P3_2(w, v, u, self.weights))
		return sorted(updated_p3s, reverse=True)

	def update_p3s_merge(self, p3s, u, v):
		pass

	def edge_disjoint_p3s(self, p3s):
		edge_disjoint_p3s = []
		p3_edges = np.zeros(self.weights.shape, dtype=bool)
		for p3 in p3s:
			u, v, w = p3.data
			if not p3_edges[u][v] and not p3_edges[v][w] and not p3_edges[u][w]:
				edge_disjoint_p3s.append(p3)
				p3_edges[u][v] = p3_edges[v][u] = True
				p3_edges[v][w] = p3_edges[w][v] = True
				p3_edges[u][w] = p3_edges[w][u] = True
		return edge_disjoint_p3s


	def merge(self, u, v):
		sign_mask = np.sign(self.weights[u]) != np.sign(self.weights[v])
		# sign_mask = np.array([self.sign(u, w) != self.sign(v, w) for w in range(self.weights.shape[0])])
		d = np.sum(np.minimum(np.abs(self.weights[u][sign_mask]), np.abs(self.weights[v][sign_mask])))
		new_weights = np.copy(self.weights)
		new_weights[u] = new_weights[:,u] = self.weights[u] + self.weights[v]
		new_weights[u][u] = 0
		new_weights = np.delete(np.delete(new_weights, v, 0), v , 1)
		adj_matrix = new_weights > 0
		return Graph(new_weights), d


	def sign(self, u, v):
		return -1 if self.weights[u][v] <= 0 else 1

	def connected_components(self):
		connected_components = []
		not_visited = np.ones(self._adj_matrix.shape[0], dtype=bool)
		root = 0
		while root != -1:
			queue = [root]
			component = []
			while len(queue) > 0:
				v = queue.pop(0)
				not_visited[v] = False
				component.append(v)
				for w in self.get_neighbors(v):
					if not_visited[w]:
						queue.append(w)
						not_visited[w] = False
			component.sort()
			connected_components.append(component)
			root = np.where(not_visited)[0][0] if np.any(not_visited) else -1
		return connected_components

	def induced_subgraph(self, vertices):
		weights = self.weights[vertices,:][:,vertices]
		return Graph(weights)


	class P3:
		def __init__(self, u, v, w, weights, marked_edges):
			# note (u, v, w) is same p3 as (w, v, u)
			# convention: store min value first
			if u < w:
				self.data = (u, v, w)
			else:
				self.data = (w, v, u)
			self.number_marked_edges = np.sum([marked_edges[u][v], marked_edges[v][w], marked_edges[u][w]])
			self.min_edge_weight = min([abs(weights[u][v]), abs(weights[v][w]), abs(weights[u][w])])
			# self.sum_edge_weight = sum([abs(weights[u][v]), abs(weights[v][w]), abs(weights[u][w])])

		def __lt__(self, other):
			if self.number_marked_edges == other.number_marked_edges:
				return self.min_edge_weight < other.min_edge_weight
			return self.number_marked_edges < other.number_marked_edges

		def __le__(self, other):
			if self.number_marked_edges == other.number_marked_edges:
				return self.min_edge_weight <= other.min_edge_weight
			return self.number_marked_edges < other.number_marked_edges

		def __hash__(self):
			return hash(self.data)

		def __eq__(self, other):
			return self.data == other.data


	class P3_2:
		def __init__(self, u, v, w, weights):
			# note (u, v, w) is same p3 as (w, v, u)
			# convention: store min value first
			if u < w:
				self.data = (u, v, w)
			else:
				self.data = (w, v, u)
			self.min_edge_weight = min([abs(weights[u][v]), abs(weights[v][w]), abs(weights[u][w])])
			# self.min_edge_weight = sum([abs(weights[u][v]), abs(weights[v][w]), abs(weights[u][w])])

		def __lt__(self, other):
			return self.min_edge_weight < other.min_edge_weight

		def __le__(self, other):
			return self.min_edge_weight <= other.min_edge_weight

		def __hash__(self):
			return hash(self.data)

		def __eq__(self, other):
			return self.data == other.data




