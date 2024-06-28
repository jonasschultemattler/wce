import sys
import numpy as np

def strip_line(line):
    return line.strip().split('#')[0]


def parse_input_file(file_path):
    with open(file_path) as file:
        n = int(strip_line(file.readline()))
        weights = np.zeros((n, n), dtype=np.int64)
        for line in file:
            content = strip_line(line).split()
            u, v, w = int(content[0])-1, int(content[1])-1, int(content[2])
            weights[u][v] = weights[v][u] = w
    return weights


def parse_input():
    n = int(strip_line(sys.stdin.readline()))
    weights = np.zeros((n, n), dtype=np.int64)
    for line in sys.stdin:
        content = strip_line(line).split()
        u, v, w = int(content[0])-1, int(content[1])-1, int(content[2])
        weights[u][v] = weights[v][u] = w
    return weights
