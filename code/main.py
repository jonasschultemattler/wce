#! /usr/bin/env python3
import sys
from graph import Graph
from fp_solver import FPSolver
from parse import parse_input, parse_input_file


if __name__ == "__main__":
    # weights = parse_input_file(sys.argv[1])
    weights = parse_input()
    
    graph = Graph(weights)
    # param0: compute greedy-upperbound in every param0 depth, in [1,2,4,..,256].
    # param1: compute lb2 in every param1 depth, in [1,2,4..,8,16].
    # param2: compute lp-lowerbound in every param2 depth, in [1,2,4,..,256].
    # param3: compute rr weight>k in every param2 depth, in [1,2,4..,64].
    # param4: compute rr heavynonedge in every param4 depth, in [1,2,4,..,128].
    # param5: solve connected components independently in every param5 depth, in [1,2,4,..,128].
    # param6: compute rr heavy_edge_single_end in every param6 depth, in [1,2,4,..,256].
    # param7: compute rr heavy_edge_both_ends in every param7 depth, in [1,2,4,..,256].
    params = [50, 1, 80, 1, 80, 80, 50, 40]
    solver = FPSolver(graph, params)
    # solver = FPSolver(graph.Graph(weights))

    solution, rec_steps = solver.solve()

    k = 0
    for v, w in solution:
        k += abs(weights[v][w])
        print("%d %d" % (v+1, w+1))
    print("#k: %d" % (k))
    print("#recursive steps: %d" % (rec_steps))


