import numpy as np
import copy


class Graph:
	def __init__(self, weights):
		self.weights = weights
		self._adj_matrix = weights > 0
		self.mask = np.ones(self._adj_matrix.shape, dtype=bool)
		# self.mask = np.ones(self._adj_matrix.shape[0], dtype=bool)

	def get_vertices(self):
		return np.unique(np.where(self.mask)[0])
		# return np.where(self.mask)[0]

	def add_vertex(self, v):
		# self.mask[v] = self.mask[:,v] = np.ones(self.weights.shape[0], dtype=bool)
		mask = np.zeros(self.weights.shape[0], dtype=bool)
		for w in self.get_vertices():
			mask[w] = True
		mask[v] = True
		self.mask[v] = self.mask[:,v] = mask
		# self.mask[v] = True

	def del_vertex(self, v):
		self.mask[v] = self.mask[:,v] = np.zeros(self.weights.shape[0], dtype=bool)
		# self.mask[v] = False

	def add_edge(self, u, v):
		self._adj_matrix[u][v] = self._adj_matrix[v][u] = True

	def del_edge(self, u, v):
		self._adj_matrix[u][v] = self._adj_matrix[v][u] = False

	def get_edges(self):
		# mask = np.ones(self._adj_matrix.shape, dtype=bool)
		# mask[]
		# edges = np.where(np.triu(self._adj_matrix & mask))
		edges = np.where(np.triu(self._adj_matrix & self.mask))
		# edges = np.where(np.triu(self._adj_matrix))
		return zip(edges[0], edges[1])

	def get_neighbors(self, v):
		return np.where(self._adj_matrix[v] & self.mask[v])[0]
		# return np.where(self._adj_matrix[v])[0]

	def common_neighbors(self, v, w):
		return np.where((self._adj_matrix[v] & self.mask[v]) & (self._adj_matrix[w] & self.mask[w]))[0]
		# return np.where(self._adj_matrix[v] & self._adj_matrix[w])[0]
		# return np.intersect1d(self._adj_matrix[v] & self.mask[v], self._adj_matrix[w] & self.mask[w])

	def all_neighbors(self, v, w):
		union = (self._adj_matrix[v] & self.mask[v]) | (self._adj_matrix[w] & self.mask[w])
		# union = self._adj_matrix[v] | self._adj_matrix[w]
		union[v] = union[w] = False
		return np.where(union)[0]
		# return np.union1d(self._adj_matrix[v] & self.mask[v], self._adj_matrix[w] & self.mask[w])

	def single_neighbors(self, v, w):
		# returns neighbors of v that are not adjacent to w
		# single_neighbors = (self._adj_matrix[v] & self.mask[v]) & np.invert(self._adj_matrix[w] & self.mask[w])
		single_neighbors = (self._adj_matrix[v] & np.invert(self._adj_matrix[w])) & self.mask[v]
		# single_neighbors = self._adj_matrix[v] & np.invert(self._adj_matrix[w])
		single_neighbors[w] = False
		return np.where(single_neighbors)[0]
		# return np.setdiff1d(self._adj_matrix[v] & self.mask[v], self._adj_matrix[w] & self.mask[w])

	def diff_neighbors(self, v, w):
		return np.where((self._adj_matrix[v] ^ self._adj_matrix[w]) & self.mask[w])[0]
		# return np.where(self._adj_matrix[v] ^ self._adj_matrix[w])[0]

	def degree(self, v):
		return len(self.get_neighbors(v))

	def vertex_degrees(self):
		return [self.degree(v) for v in range(self._adj_matrix.shape[0])]

	def is_cluster(self):
		return not self.contains_p3()

	def contains_p3(self):
		return self.get_p3() is not None

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
		# todo: heap!
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


	def list_edge_disjoint_p3s(self):
		p3s = []
		p3_edges = np.zeros(self.weights.shape, dtype=bool)
		for u, v in sorted(self.get_edges(), key=lambda e: abs(self.weights[e[0]][e[1]]), reverse=True):
			if not p3_edges[u][v]:
				p3, w_max = None, -1
				for w in self.single_neighbors(u, v):
					if not p3_edges[u][w] and not p3_edges[v][w] and min(abs(self.weights[u][w]), abs(self.weights[v][w])) > w_max:
						p3, w_max = (w, u, v), min(abs(self.weights[u][w]), abs(self.weights[v][w]))
				for w in self.single_neighbors(v, u):
					if not p3_edges[v][w] and not p3_edges[u][w] and min(abs(self.weights[v][w]), abs(self.weights[u][w])) > w_max:
						p3, w_max = (u, v, w), min(abs(self.weights[v][w]), abs(self.weights[u][w]))
				if p3 is not None:
					u, v, w = p3
					p3s.append(self.P3_2(u, v, w, self.weights))
					p3_edges[u][v] = p3_edges[v][u] = True
					p3_edges[v][w] = p3_edges[w][v] = True
					p3_edges[u][w] = p3_edges[w][u] = True
		return p3s, p3_edges


	def sorted_edge_disjoint_p3s(self):
		p3s, p3_edges = self.list_edge_disjoint_p3s()
		return sorted(p3s, reverse=True), p3_edges


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


	def new_edge_disjoint_p3(self, p3_edges, u, v):
		p3_max, w_max = None, -1
		if self.weights[u][v] > 0:
			for w in self.common_neighbors(u, v):
				if not p3_edges[u][w] and not p3_edges[v][w]:
					new_p3 = self.P3_2(u, w, v, self.weights)
					if new_p3.min_edge_weight > w_max:
						p3_max, w_max = new_p3, new_p3.min_edge_weight
		else:
			for w in self.single_neighbors(u, v):
				if not p3_edges[u][w] and not p3_edges[v][w]:
					new_p3 = self.P3_2(w, u, v, self.weights)
					if new_p3.min_edge_weight > w_max:
						p3_max, w_max = new_p3, new_p3.min_edge_weight
			for w in self.single_neighbors(v, u):
				if not p3_edges[u][w] and not p3_edges[v][w]:
					new_p3 = self.P3_2(w, v, u, self.weights)
					if new_p3.min_edge_weight > w_max:
						p3_max, w_max = new_p3, new_p3.min_edge_weight
		return p3_max


	def sign(self, u, v):
		return -1 if self.weights[u][v] <= 0 else 1


	def merge(self, u, v):
		sign_mask = (np.sign(self.weights[u]) != np.sign(self.weights[v])) & self.mask[u]
		d = np.sum(np.minimum(np.abs(self.weights[u][sign_mask]), np.abs(self.weights[v][sign_mask])))
		self.weights[u] = self.weights[:,u] = self.weights[u] + self.weights[v]
		self.weights[u][u] = 0
		self._adj_matrix[u] = self._adj_matrix[:,u] = self.weights[u] > 0
		self.del_vertex(v)
		return d

	def unmerge(self, u, v):
		self.weights[u] = self.weights[:,u] = self.weights[u] - self.weights[v]
		self.weights[u][u] = 0
		self._adj_matrix[u] = self._adj_matrix[:,u] = self.weights[u] > 0
		self.add_vertex(v)


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


	def cc_size(self):
		cc_size = [0]*self.n
		for component in self.connected_components():
			size = len(component)
			for v in component:
				cc_size[v] = size
		return cc_size

	def find_p3(self):
		cc_size = self.cc_size()
		deg = self.vertex_degrees()
		for v in range(self.weights.shape[0]):
			if deg[v] < cc_size[v] - 1:
				return # TODO: BFS depth 2
		return None


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

		def __repr__(self):
			u, v, w = self.data
			return "({},{},{}) {}".format(u, v, w, self.min_edge_weight)



