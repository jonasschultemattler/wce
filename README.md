# Weighted Cluster Editing

This package provides an exact solver for the Weighted Cluster Editing Problem.
It implements the branch-and-bound algorithm by cite and features implementations of
## Heuristics
...
## Data Reductions
...
## Lower Bounds
...

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
4. Setup datasets
```bash
git clone https://github.com/PACE-challenge/Cluster-Editing-PACE-2021-instances
cd data && ./setup.sh
```
or download
```bash
wget https://fpt.akt.tu-berlin.de/pace2021/exact.tar.gz && tar -xf exact.tar.gz
```
5. Setup gurobi
6. Run exact solver

