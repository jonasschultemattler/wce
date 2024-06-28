#! /usr/bin/env python3
import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from graph import Graph
from parse import parse_input_file, parse_input

class LPSolver:
	def __init__(self, graph):
		self.graph = graph

	def setup(self):
		self.model = gp.Model()
		n = self.graph._adj_matrix.shape[0]
		self.edges = self.model.addVars([(i, j) for i in range(n) for j in range(i+1, n)], vtype=GRB.CONTINUOUS, lb=0, ub=1, name='edges')

		nonedges = [(i, j) for i in range(n) for j in range(i+1, n) if not self.graph._adj_matrix[i][j] and self.graph.weights[i][j] > -50000]
		obj = gp.quicksum((1 - self.edges[i, j])*self.graph.weights[i][j] for i, j in self.graph.get_edges()) + gp.quicksum(self.edges[i, j]*abs(self.graph.weights[i][j]) for i, j in nonedges)
		self.model.setObjective(obj, GRB.MINIMIZE)

		triples = [(u, v, w) for u in range(n) for v in range(u+1, n) for w in range(v+1, n)]
		self.model.addConstrs(self.edges[u, v] + self.edges[v, w] - self.edges[u, w] <= 1 for u, v, w in triples)
		self.model.addConstrs(self.edges[u, v] - self.edges[v, w] + self.edges[u, w] <= 1 for u, v, w in triples)
		self.model.addConstrs(-self.edges[u, v] + self.edges[v, w] + self.edges[u, w] <= 1 for u, v, w in triples)


	def solve(self):
		self.model.optimize()
		# n = self.graph._adj_matrix.shape[0]
		# solution = [(i, j) for i in range(n) for j in range(i+1, n) if (self.edges[i, j].x and not self.graph._adj_matrix[i][j]) or (not self.edges[i, j].x and self.graph._adj_matrix[i][j])]
		# return solution
		return self.model.ObjVal


class ILPSolver:
	def __init__(self, graph):
		self.graph = graph

	def setup_ilp(self):
		self.model = gp.Model()
		n = self.graph._adj_matrix.shape[0]
		self.edges = self.model.addVars([(i, j) for i in range(n) for j in range(i+1, n)], vtype=GRB.BINARY, name='edges')

		nonedges = [(i, j) for i in range(n) for j in range(i+1, n) if not self.graph._adj_matrix[i][j]]
		obj = gp.quicksum((1 - self.edges[i, j])*self.graph.weights[i][j] for i, j in self.graph.get_edges()) + gp.quicksum(self.edges[i, j]*abs(self.graph.weights[i][j]) for i, j in nonedges)
		self.model.setObjective(obj, GRB.MINIMIZE)

		triples = [(u, v, w) for u in range(n) for v in range(u+1, n) for w in range(v+1, n)]
		self.model.addConstrs(self.edges[u, v] + self.edges[v, w] - self.edges[u, w] <= 1 for u, v, w in triples)
		self.model.addConstrs(self.edges[u, v] - self.edges[v, w] + self.edges[u, w] <= 1 for u, v, w in triples)
		self.model.addConstrs(-self.edges[u, v] + self.edges[v, w] + self.edges[u, w] <= 1 for u, v, w in triples)


	def solve_ilp(self):
		self.model.optimize()
		n = self.graph._adj_matrix.shape[0]
		solution = [(i, j) for i in range(n) for j in range(i+1, n) if (self.edges[i, j].x and not self.graph._adj_matrix[i][j]) or (not self.edges[i, j].x and self.graph._adj_matrix[i][j])]
		return solution


if __name__ == "__main__":
	weights = parse_input_file(sys.argv[1])
    # weights = parse_input()
	graph = Graph(weights)

    # solver = ILPSolver(graph)

    # solver.setup_ilp()
    # solution = solver.solve_ilp()

    # k = 0
    # for v, w in solution:
    #     k += abs(graph.weights[v][w])
    #     print("%d %d" % (v+1, w+1))
    # print("#k: %d" % (k))
	solver = LPSolver(graph)
	solver.setup()
	solution = solver.solve()
	print(solution)

    # k = 0
    # for v, w in solution:
    #     k += abs(graph.weights[v][w])
    #     print("%d %d" % (v+1, w+1))
    # print("#k: %d" % (k))


