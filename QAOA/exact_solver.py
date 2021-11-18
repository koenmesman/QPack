#!/usr/bin/env python
import generate_graph as gg
from itertools import permutations, combinations_with_replacement

def bitfield(n):
    return [int(digit) for digit in bin(n)[2:]]


def hamming_2(n):
  hamming_set = []
  for i in range(n):
    for j in range(i+1, n):
      set = [0]*n
      set[i] = 1
      set[j] = 1
      hamming_set.append(set)
  return hamming_set

def flatten(t):
    return [item for sublist in t for item in sublist]

def adjacencies(nodes):
    hamming_sets = hamming_2(nodes)
    #set = combinations(hamming_sets, nodes)
    superset = combinations_with_replacement(hamming_sets, nodes)
    flatset = []
    megaset = []
    for subset in superset:
      megaset.append(permutations(list(subset)))
    for subset in megaset:
      for subsubset in subset:
        #flatset.append(flatten(list(subsubset)))
        flatset.append(list(subsubset))

    #this still returns a lot of duplicates, finding the set of uniques is to be implemented
    return flatset


def mcp_batch_solver(max_size):
    opt_results = [0] * (max_size - 4)
    for i in range(5, max_size+1):
        opt_results[i - 5] = mcp_solver(i)
    return opt_results

def mcp_solver(size):
    print("calculating exact mcp solution")
    size, edges = gg.regular_graph(size)
    result = 0
    
    for n in range(2**(size-1)):
        node_array = bitfield(n)
        zero_array = [0]*(size-len(node_array))
        node_array = zero_array+node_array
        node_array = [-1 if x == 0 else 1 for x in node_array]
        c = 0
        for e in edges:
            c += 0.5 * (1 - int(node_array[e[0]]) * int(node_array[e[1]]))
        if c >= result:
            result = c
    return result



def dsp_solver(size):
    print("calculating exact dsp solution")
    result = 0
    c = 0
    size, edges = gg.regular_graph(size)
    connections = []
    for k in range(size):
        connections.append([k])
    for t in edges:
        connections[t[0]].append(t[1])
        connections[t[1]].append(t[0])
    for n in range(2 ** size):
        node_array = bitfield(n)
        zero_array = [0] * (size - len(node_array))
        node_array = zero_array + node_array
        T = 0
        for con in connections:
            tmp = 0
            for k in con:
                tmp = tmp or node_array[k]
                if tmp:
                    T += 1
                    break
        D = 0
        for j in range(size):
            D += 1 - node_array[j]
        c = (T + D)
        if c >= result:
            result = c
    return c

def dsp_score(max_size):
    opt_results = [0] * (max_size - 4)
    for i in range(5, max_size + 1):
        result = 0
        c = 0
        size, edges = gg.regular_graph(i)
        connections = []
        for k in range(size):
            connections.append([k])
        for t in edges:
            connections[t[0]].append(t[1])
            connections[t[1]].append(t[0])
        for n in range(2 ** size):
            node_array = bitfield(n)
            zero_array = [0] * (size - len(node_array))
            node_array = zero_array + node_array
            T = 0
            for con in connections:
                tmp = 0
                for k in con:
                    tmp = tmp or node_array[k]
                    if tmp:
                        T += 1
                        break
            D = 0
            for j in range(size):
                D += 1 - node_array[j]
            c = (T + D)
            if c >= result:
                result = c
        opt_results[i - 5] = result
    return opt_results


def tsp_solver(size):
    size, A, D = gg.tsp_problem_set(size, gg.regular_graph)
    D = [100, 50, 50, 50, 50, 50, 100, 50, 50, 50, 50, 50, 100, 50, 50, 50, 50, 50, 100, 50, 50, 50, 50, 50, 100]
    coupling = []
    result = 10**8
    for i in range(size):
        for j in range(i):
            if i != j:
                #coupling.append([i + j * size, j + i * size])
                coupling.append([j, i])
    print("evaluating all possible configurations, this might take a while.")
    node_array_set = adjacencies(size)
    for node_array in node_array_set:
        cost = 0
        for i in range(0, size):
            for j in range(0, size):
                #cost += D[i + size * j] * node_array[i + size * j]
                cost += 0.5*D[i*size + j] * node_array[i][j]
        for j in coupling:
            cost += -5 * (1 - 2 * node_array[j[0]][j[1]]) * (1 - 2 * node_array[j[1]][j[0]])
        if cost <= result:
            result = cost
    return result

def tsp_batch_solver(max_size):
    opt_results = [0]*(max_size-4)
    for n in range(5, max_size+1):
        opt_results[n-5] = tsp_solver(n)
    return opt_results
