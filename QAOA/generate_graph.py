import time
import random
import numpy as np


# returns a graph with a set edge to vertices ratio
def set_density(n, r):
    edges = []
    nr_edges = int(round(n*r))
    while nr_edges:
        tmp_edge = [random.randrange(0, n, 1), random.randrange(0, n, 1)]
        if tmp_edge[0] != tmp_edge[1]:
            tmp_edge.sort()
            if tmp_edge not in edges:
                edges.append(tmp_edge)
                nr_edges -= 1
    return [n, edges]


# return a problem graph with a set edge to vertices ratio, and ensures that all nodes are included
def include_all(n, r):
    edges = []
    nr_edges = int(round(n*r))
    for i in range(n):
        flag = True
        timeout = time.time() + 30
        retry = True
        while flag:
            while retry:
                tmp_edge = [i, random.randrange(0, n-1, 1)]
                if tmp_edge[0] != tmp_edge[1]:
                    tmp_edge.sort()
                    if tmp_edge not in edges:
                        edges.append(tmp_edge)
                        flag = False
                        retry = False
            if time.time() > timeout:
                print('timed out')
                break
    nr_edges -= n
    while nr_edges:
        tmp_edge = [random.randrange(0, n-1, 1), random.randrange(0, n-1, 1)]
        if tmp_edge[0] != tmp_edge[1]:
            tmp_edge.sort()
            if tmp_edge not in edges:
                edges.append(tmp_edge)
                nr_edges -= 1
    return [n, edges]


# returns the problem graph where every edge has a set probability
def set_probability(n, p):
    edges = []
    for i in range(n-1):
        for j in range(i+1, n):
            if random.random() < p:
                edges.append([i, j])
    return [n, edges]


# return a regular graph
def regular_graph(n):
    edges = []
    for i in range(n-1):
        edges.append([i, i+1])
    edges.append([0, n-1])
    for i in range(n-2):
        edges.append([i, i+2])
    edges.append([0, n-2])
    edges.append([1, n-1])
    return [n, edges]


# return a fully connected graph (set_probability p=1) if weighted == True, give each edge a weight [1, 10]
def fully_connected(n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append([i, j])
    return [n, edges]

# transforms the graph to a weighted graph with edge value inf for disconnected vertices
def tsp_problem_set(n, method, *arg):
    if len(arg) == 1:
        p = arg[0]
        edge_list = method(n, p)
    else:
        edge_list = method(n)
    [n, full_list] = fully_connected(n)
    for i in range(len(full_list)):
        if full_list[i] not in edge_list:
            full_list[i].append(50)
        else:
            full_list[i].append(random.randrange(1, 10, 1))
    e = full_list
    A = [[0 for x in range(n)] for x in range(n)]
    D = [[0 for x in range(n)] for x in range(n)]

    for n in range(n):
        D[n][n] = 2*max([item[2] for item in e])
    for t in e:
        A[t[0]][t[1]] = 1
        A[t[1]][t[0]] = 1
        D[t[0]][t[1]] = t[2]
        D[t[1]][t[0]] = t[2]
    A = [item for sublist in A for item in sublist]
    D = [item for sublist in D for item in sublist]
    return [n, A, D]