# Weighted Cluster Editing

This package provides an exact solver for the Weighted Cluster Editing Problem.
It implements the branch-and-bound algorithm by cite and features:
## Heuristics
- Randomized Approximation: ...
- Greedy-Algorithm by cite
- Local Search refinement, see ...

todo: plots with solution quality and running times

## Data Reduction Rules
- "High Cost Edge": Merge too expensive edges
- "Heavy Non Edge" 
- "Heavy Edge Sinlge End" 
- "Heavy Edge Both Ends"
- "Large Neighborhood"

todo: plots with impact of reduction rules and running times

## Lower Bounds
- LP-Relaxation solved by Gurobi

## Benchmarks:
Datasets [cite]
Cactus-plot for comparison with Gurobi

## Getting started:
1. Install Docker
2. clone repo and cd into it
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

