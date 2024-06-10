FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y time python3 python3-pip
RUN python3 -m pip install --user scipy numpy gurobipy

WORKDIR /app