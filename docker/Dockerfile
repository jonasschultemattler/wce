FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y time curl wget unzip python3 python3-pip
RUN python3 -m pip install --user scipy numpy gurobipy matplotlib pandas tqdm

WORKDIR /wce