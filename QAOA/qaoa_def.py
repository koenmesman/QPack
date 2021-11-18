#########################################################################
# QPack                                                                 #
# Koen Mesman, TU Delft, 2021                                           #
# This file defines the QAOA circuit implementations and cost functions.#
#########################################################################
from math import pi, acos
from matplotlib import cm
import matplotlib.pyplot as plt
import Benchmark as bm
plt.interactive(True)
import time

from numpy import *
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute, Aer, IBMQ
from qiskit.providers.aer import QasmSimulator


qaoa_shots = 100

# additional gates
def cry(theta):
    qc = QuantumCircuit(2)
  #  qc.cnot(0, 1)
  #  qc.ry(-theta/2, 1)
  #  qc.cnot(0, 1)
  #  qc.ry(theta / 2, 1)

    qc.ry((pi/2)-theta/2, 0)
    qc.cnot(1, 0)
    qc.ry(-((pi/2)-theta/2), 0)

    return qc.to_gate()


def ccry(theta):
    qc = QuantumCircuit(3)
    qc.toffoli(0, 1, 2)
    qc.ry(-theta/2, 2)
    qc.toffoli(0, 1, 2)
    qc.ry(theta / 2, 2)
    return qc.to_gate()


def scs(n, k):
    qc = QuantumCircuit(n)
    qc.cnot(n - 2, n-1)
    theta = 2 * (acos(sqrt(1 / n)))
    qc.append(cry(theta), [n-2, n - 1])
    qc.cnot(n - 2, n-1)

    for m in range(k - 1):
        l = 2+m
        control = n-2-m
        qc.cnot(control-1, n-1)
        theta = 2 * (acos(sqrt((n-control) / n)))
        qc.append(ccry(theta), [n-1, control, control-1])

        qc.cnot(control-1, n-1)
    return qc.to_gate()


def dicke_init(n, k):
    #deterministic  Dicke state preparation (Bärtschi & Eidenbenz, 2019)
    #unoptimized version
    qc = QuantumCircuit(n)
    qc.x(range(n-k, n))
    for i in range(n, k, -1):
        qc.append(scs(i, k), range(0, i))

    for i in range(k, 1, -1):
        qc.append(scs(i, i-1), range(0, i))

    return qc.to_gate()


def OR_2q():
    qc = QuantumCircuit(3)
    qc.toffoli(0, 1, 2)
    qc.cnot(0, 1)
    qc.cnot(0, 2)
    return qc.to_gate()


def OR_nrz(n, gamma):
    ORGate = OR_2q()
    qc = QuantumCircuit(2*n)
    qc.append(ORGate, [0, 1, n])
    for i in range(2, n):
        qc.append(ORGate, [i, n+i-2, n+i-1])
    qc.crz(gamma, 2*n-2, 2*n-1)
    for i in range(n, 2, -1):
        qc.append(ORGate, [n-i-1, 2*n-2-i, 2*n-1-i])
    qc.append(ORGate, [0, 1, n])
    return qc.to_gate()


# max-cut
def eval_cost(out_state, graph):
    # evaluate Max-Cut
    v = graph[0]
    edges = graph[1]
    c = 0
    bin_len = "{0:0" + str(v) + "b}"  # string required for binary formatting
    bin_val = [int(i) for i in list(out_state)]
    bin_val = [-1 if x == 0 else 1 for x in bin_val]
    for e in edges:
        c += 0.5 * (1 - int(bin_val[e[0]]) * int(bin_val[e[1]]))
    return c


def max_cut_circ(params, graph, p, backend):
    beta = params[0:p]     # !!!
    gamma = params[p:2*p]  #!!!
    #[beta, gamma] = params           #!!!
    #beta = [beta]
    #gamma = [gamma]
    #gamma = []
    #for b in beta:          #!!!
    #    gamma.append(pi-2*b)

    v, edge_list = graph
    vertice_list = list(range(0, v, 1))
    for pb in range(len(beta)):
        if beta[pb] < 0 or beta[pb] > (2 * pi) or gamma[pb] < 0 or gamma[pb] > (2 * pi):
            return 0
    else:
        c = ClassicalRegister(v)
        q = QuantumRegister(v)
        qc = QuantumCircuit(q, c)
        for qubit in range(v):
            qc.h(qubit)
        for iteration in range(p):
            for e in edge_list:                     # TODO: fix for unordered edges e.g. (2,0)
                qc.cnot(e[0], e[1])
                qc.rz(-gamma[p-1], e[1])
                qc.cnot(e[0], e[1])
            for qb in vertice_list:
                qc.rx(2*beta[p-1], qb)
        qc.measure(q, c)
        start = time.time()
        job = execute(qc, backend, shots=qaoa_shots)
        end = time.time()
        wall = end-start
        result = job.result()
        out_state = result.get_counts()
       # print(out_state)
        #print(qc)
    prob = list(out_state.values())
    states = list(out_state.keys())
    exp = 0
    for k in range(len(states)):
        exp += eval_cost(states[k], graph)*prob[k]
       # print(exp)
    try:
        time_res = job.time_per_step()
        time_dict = bm.get_times(time_res, wall)
        bm.BENCHMARK_TIMES = bm.merge_times(bm.BENCHMARK_TIMES, time_dict)
    except:
        time_res = 0
    return [exp/qaoa_shots]


# traveling salesman
def cost_tsp(params, graph, p, rep):
    v, A, D = graph
    out_state = tsp_circ(params, graph, p, rep)
    total_cost = 0
    for t in out_state:
        count = int(out_state.get(t))
        bin_len = "{0:0" + str(v) + "b}"  # string required for binary formatting
        bin_val = t.format(bin_len)
        bin_val = [int(i) for i in bin_val]
        cost = 0
        coupling = []
        for i in range(v):
            for j in range(i):
                if i != j:
                    coupling.append([i+j*v, j+i*v])

        for i in range(0, v):
            for j in range(i, v):
                cost += D[i + v*j]*bin_val[i + v*j]
        for j in coupling:
            cost += -5*(1 - 2*bin_val[j[0]])*(1 - 2*bin_val[j[1]])
        total_cost += cost*count/rep

    return total_cost


def tsp_circ(params, graph, p, rep):
    beta = params[0:p]
    gamma = params[p:2*p]
    v, A, D = graph

    vertice_list = list(range(0, v, 1))
    for p in range(len(beta)):
        if beta[p] < 0 or beta[p] > (2 * pi) or gamma[p] < 0 or gamma[p] > (2 * pi):
            return 0
    else:
        n = v**2
        qc = QuantumCircuit(n, n)
        # INIT  :    This initiation can and should be reused
        for q in range(v):
            q_range = range(q*v, (q+1)*v)
            DickeGate = dicke_init(v, 2)
            qc.append(DickeGate, q_range)
        # QAOA cycles
        #for 1 to p:

        #cost unitary
        for iteration in range(p):
            for i in range(n):
                qc.rz(gamma[p]*D[i]/(2*pi), i)

            for i in range(v):
                for j in range(i):
                    if i != j:
                        qc.rzz(20*gamma[p]/pi, (j+i*v), i+j*v)   #RZZ on reflection over the diagonal

            #mixer unitary
            for i in range(0, v):
                qc.rxx(-beta[p], i*v, (i*v+1))
                qc.rxx(-beta[p], (i*v+1), (i*v+2))

                qc.ryy(-beta[p], i * v, (i * v + 1))
                qc.ryy(-beta[p], (i * v + 1), (i * v + 2))


        qc.measure(range(n), range(n))
        #qc.draw(output='mpl', filename='my_circuit.png')
        simulator = QasmSimulator(configuration='automatic')
        result = execute(qc, simulator, shots=rep).result()
        #out_state = result.get_counts()
        out_state = result.get_counts()
        #plot_histogram(out_state, figsize=(7, 5))

        #eval cost

    return out_state


# dominating set problem
def dsp_circuit(params, graph, p):
    beta = params[0:p]
    gamma = params[p:2*p]
    v, edge_list = graph
    vertice_list = list(range(0, v, 1))

    connections = []
    for i in range(v):
        connections.append([i])
    for t in edge_list:
        connections[t[0]].append(t[1])
        connections[t[1]].append(t[0])
    ancillas = 0
    for con in connections:
        if len(con) > ancillas:
            ancillas = len(con)
    n = v+ancillas         # add ancillas
    qc = QuantumCircuit(n, v)

    for qubit in range(v):
        qc.h(qubit)
        #inverted  crz gate
        qc.x(qubit)
        qc.crz(-gamma[0], qubit, n-1)
        qc.x(qubit)
    for iteration in range(p):
        f = 0
        f_anc = v
        for con in connections:  # TODO: fix for unordered edges e.g. (2,0)
            c_len = len(con)
            OR_range = con
            for k in range(c_len-1):
                OR_range.append(v+k)
            cOR_rz = OR_nrz(c_len, gamma[p-1])
            OR_range.append(n-1)
            qc.append(cOR_rz, OR_range)

        for qb in vertice_list:
            qc.rx(-2*beta[p-1], qb)
    qc.measure(range(v), range(v))
    res = execute(qc, backend, shots=qaoa_shots).result()
    out_state = res.get_counts()
    return out_state


def dsp_cost(params, v, e, p):
    graph = [v, e]
    beta = params[0:p]
    gamma = params[p:2*p]

    for pb in range(len(beta)):
        if beta[pb] < 0 or beta[pb] > (2 * pi) or gamma[pb] < 0 or gamma[pb] > (2 * pi):
            return 0
    else:
        out_state = dsp_circuit(params, graph, p)

    edge_list = e
    vertice_list = list(range(0, v, 1))
    connections = []
    for i in range(v):
        connections.append([i])
    for t in edge_list:
        connections[t[0]].append(t[1])
        connections[t[1]].append(t[0])
    total_count = 0
    total_cost = 0
    for p_it in out_state:
        for t in out_state:
            count = int(out_state.get(t))
            total_count += count
            bin_len = "{0:0" + str(v) + "b}"  # string required for binary formatting
            bin_val = t.format(bin_len)
            bin_val = [int(i) for i in bin_val]
            T = 0
            for con in connections:
                tmp = 0
                for k in con:
                    tmp = tmp or bin_val[k]
                    if tmp:
                        T += 1
                        break
            D = 0
            for i in range(v):
                D += 1 - bin_val[i]
            total_cost += (T+D)*count
    total_cost = total_cost/total_count
    return total_cost


# plot parameter landscape for mcp
def qaoa_landscape(g, n, p, bool_show):
    betas = np.arange(0, 2 * pi, 1/n)
    gammas = np.arange(0, 2 * pi, 1/n)
    val = [[0]*(n) for _ in range(n)]
    for b in range(n):
        beta = betas[b]
        for c in range(n):
            gamma = gammas[c]
            val[b][c] = max_cut_circ([beta, gamma], g, p)

    X, Y = np.meshgrid(betas, gammas)
    if bool_show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, np.array(val), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
        plt.show()

    return X, Y, val

