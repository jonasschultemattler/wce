# Weighted Cluster Editing

Framework for the Weighted Cluster Editing Problem.
It implements the branch-and-bound algorithm by [Böcker et al.](https://www.sciencedirect.com/science/article/pii/S0304397509003521) and features:
## Heuristics
- Greedy-Algorithm by [Rahmann et al.](https://pub.uni-bielefeld.de/record/2499481)
- Randomized Approximation
- Local Search

For solution quality and running times see the [notebook](notebooks/performance.ipynb)

## Data Reduction Rules
- "Heavy Edge": Merge too expensive edges' vertices
- "Heavy Non Edge" 
- "Heavy Edge Sinlge End" 
- "Heavy Edge Both Ends"
- "Large Neighborhood"

For empirical evaluation see the [notebook](notebooks/performance.ipynb)

## Lower Bounds
- LP-Relaxation solved by Gurobi

Find comparison in [notebook](notebooks/performance.ipynb)

## Benchmarks

Datasets: random, real-world, action-seq [cite]

Performance Comparison with Gurobi

<!-- ![Cactus plot showing performance of solver and gurobi.](imgs/solver_vs_gurobi.png) -->
<img src="imgs/solver_vs_gurobi.png" width=75% alt="Cactus plot showing performance of solver and gurobi.">

## Getting started
1. Install Docker
2. Clone repo and cd into it
```bash
git clone https://github.com/jonasschultemattler/wce
cd wce
```
3. Build the Docker image:
```bash
docker build -f docker/Dockerfile -t wce .
```
4. Download datasets
```bash
wget https://fpt.akt.tu-berlin.de/pace2021/exact.tar.gz && tar -xf exact.tar.gz
```
5. Setup gurobi
6. Run solver

