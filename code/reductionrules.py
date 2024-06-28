import numpy as np
import heapq


# def high_weight(graph, k):
# 	too_expensive_edges = graph.too_expensive_edges(k)
# 	if len(too_expensive_edges) > 0:
# 		u, v = too_expensive_edges[0]
# 		d = graph.merge_cost(u, v)
# 		graph.merge(u, v)
# 		return [(u, v)], d
# 	return [], 0

# def high_weight_exhaustive(graph, k):
# 	merged, d = [], 0
# 	too_expensive_edges = graph.too_expensive_edges(k)
# 	while len(too_expensive_edges) > 0:
# 		u, v = too_expensive_edges[0]
# 		d += graph.merge_cost(u, v)
# 		graph.merge(u, v)
# 		merged.insert(0, (u, v))
# 		too_expensive_edges = graph.too_expensive_edges(k-d)
# 	return merged, d

def high_weight(graph, k):
	too_expensive_edges = graph.too_expensive_edges(k)
	if len(too_expensive_edges) > 0:
		u, v = too_expensive_edges[0]
		d = graph.merge_cost(u, v)
		graph.merge(u, v)
		return [(u, v)], d
	return [], 0

def high_weight_exhaustive(graph, edges, k):
	merged, d = [], 0
	while -edges[0][0] > k - d:
		u, v = heapq.heappop(edges)[1]
		d += graph.merge_cost(u, v)
		graph.merge(u, v)
		# update edges weights

		merged.insert(0, (u, v))
	print(merged)
	return merged, d
	# too_expensive_edges = graph.too_expensive_edges(k)
	# while len(too_expensive_edges) > 0:
	# 	u, v = too_expensive_edges[0]
	# 	d += graph.merge_cost(u, v)
	# 	graph.merge(u, v)
	# 	merged.insert(0, (u, v))
	# 	too_expensive_edges = graph.too_expensive_edges(k-d)
	# return merged, d

def heavy_non_edge(graph):
	edges = []
	for u, v in graph.get_edges():
		if graph.weights[u][v] < 0 and abs(graph.weights[u][v]) >= sum([graph.weights[u][w] for w in graph.get_neighbors(u)]):
			# edited_edges[u][v] = edited_edges[v][u] = True
			edges.append((u, v))
	return edges

def heavy_edge_single_end(graph):
	for u, v in graph.get_edges():
		if graph.weights[u][v] >= sum([abs(graph.weights[u][w]) for w in range(graph.weights.shape[0]) if w != u and w != v]):
			return (u, v)
	return None

def heavy_edge_single_end_exhaustive(graph):
	merged, d = [], 0
	heavy_edge = heavy_edge_single_end(graph)
	while heavy_edge is not None:
		u, v = heavy_edge
		d += graph.merge_cost(u, v)
		graph.merge(u, v)
		merged.insert(0, (u, v))
		heavy_edge = heavy_edge_single_end(graph)
	return merged, d

def heavy_edge_both_ends(graph):
	for u, v in graph.get_edges():
		if graph.weights[u][v] >= sum([graph.weights[u][w] for w in graph.get_neighbors(u) if w != v]) + sum([graph.weights[v][w] for w in graph.get_neighbors(v) if w != u]):
			return (u, v)
	return None

def heavy_edge_both_ends_exhaustive(graph):
	merged, d = [], 0
	heavy_edge = heavy_edge_both_ends(graph)
	while heavy_edge is not None:
		u, v = heavy_edge
		d += graph.merge_cost(u, v)
		graph.merge(u, v)
		merged.insert(0, (u, v))
		heavy_edge = heavy_edge_both_ends(graph)
	return merged, d


def closed_neigborhood(self, graph, u):
	return [v for v in graph.get_neighbors(u) if graph.weights[u][v] > 0] + [u] 
	# return graph.get_neighbors() + [u]


def deficiency(self, graph, u):
	n_u = self.closed_neigborhood(graph, u)
	return sum([abs(graph.weights[v][w]) for v in n_u for w in n_u if v != w and graph.weights[v][w] < 0])


def cut_weight(self, graph, u):
	n_u = self.closed_neigborhood(graph, u)
	return sum([graph.weights[v][w] for v in n_u for w in range(graph.weights.shape[0]) if w not in n_u and graph.weights[v][w] > 0])


def large_neighborhood(self, graph):
	for u in range(graph.weights.shape[0]):
		n_u = self.closed_neigborhood(graph, u)
		if len([(v, w) for v in n_u for w in n_u if v != w and graph.weights[v][w] == 0]) > 0:
			continue
		if 2*self.deficiency(graph, u) + self.cut_weight(graph, u) < len(n_u):
			return n_u
	return []


def rr_large_neighborhood(self, graph, merged_vertices):
	indices = [i for i in range(graph.weights.shape[0])]
	d = 0
	n_u = self.large_neighborhood(graph)
	for v in n_u:
		for w in n_u:
			if v != w:
				edges, cost = graph.merge(v, w)
				d += cost
				merged_vertices[indices[v]] = indices[u]
				indices = np.delete(indices, v, 0)
	return graph, merged_vertices, d
