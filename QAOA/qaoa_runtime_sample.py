#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 19:42:05 2021

@author: koen
"""

import scipy.optimize as opt
import numpy as np
from numpy import pi
from qiskit import transpile, ClassicalRegister, QuantumRegister, QuantumCircuit
import time
global qtime
global opttime
global queue_time

def main(backend,
         user_messenger,
         #graph=[],
         p=1,
         size=5,
         shots=100,
         optimizer_config={'maxiter': 10}
         ):
    """
    Main entry point of the program.

    Parameters:
        backend (ProgramBackend): Backend to submit the circuits to.
        user_messenge (ProgramBackend): Used to communicate with the program consumer.
        p (int): QAOA layers.
        shots (int): Optional, number of shots to take per circuit.
        optimizer_config (dict): Optional, configuration parameters for the
                                optimizer.
    Returns:
        OptimizeResult: The result in SciPy optimization format.
    """
    global qtime
    global opttime
    qtime = 0
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
    graph = regular_graph(5)
    
    def callback(xk):
        user_messenger.publish(list(xk))
    
    def prepare_circuits(backend, params, graph, p):
        """Generate a QAOA Max-Cut problem circuit.

        Args:
            backend: Backend used for transpilation.
            params: Parameters beta and gamma used in QAOA.
            graph: Graph of the target problem.
            p: Number of QAOA iterations.

        Returns:
            Generated circuit.
        """
        beta = params[0:p]
        gamma = params[p:2*p]

        v, edge_list = graph
        vertice_list = list(range(0, v, 1))

        c = ClassicalRegister(v)
        q = QuantumRegister(v)
        qc = QuantumCircuit(q, c)
        for qubit in range(v):
            qc.h(qubit)
        for iteration in range(p):
            for e in edge_list:
                qc.cnot(e[0], e[1])
                qc.rz(-gamma[p-1], e[1])
                qc.cnot(e[0], e[1])
            for qb in vertice_list:
                qc.rx(2*beta[p-1], qb)
        qc.measure(q, c)

        return transpile(qc, backend)


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

    def mcp_fun(params, size, p, iterations, backend, graph):
        """Run and evaluate the QAOA MaxCut problem.

        Args:
            backend: Backend used for transpilation.
            params: Parameters beta and gamma used in QAOA.
            size: Problem size.
            p: Number of QAOA iterations.

        Returns:
            Generated circuit.
        """

        qc = prepare_circuits(backend, params, graph, p)
        start = time.time()
        job = backend.run(qc, shots=iterations)
        #user_messenger.publish("status:{}".format())
        result = job.result()
        end = time.time()-start

        global qtime
        try: qtime
        except NameError: qtime = time.time()-time.time() #timedelta?
        qtime += end
        #user_messenger.publish("qtime:{}".format(qtime))
        out_state = result.get_counts()
        prob = list(out_state.values())
        states = list(out_state.keys())
        exp = 0
        for k in range(len(states)):
            exp += eval_cost(states[k], graph) * prob[k]

        return -(exp/iterations)
    
    #iterations = kwargs.pop('iterations', 5)
    #p = kwargs.pop('p')
    global opttime
    try: opttime
    except NameError: opttime = time.time()-time.time() #timedelta?
    opt_start = time.time()
    params = np.zeros(2*p)
    bounds = [(0, pi), (0, 2 * pi)]

    result = opt.shgo(mcp_fun, bounds, args=(size, p, shots, backend, graph),
                   options={'ftol': 1e-10, 'maxfev': 100}, callback=callback) 
    #user_messenger.publish({"iteration": it, "results": result, "time": end})
    user_messenger.publish("time = {}".format(qtime))
    opttime += time.time()-opt_start-qtime
    return {"result":result, "opt_time":opttime, "qtime": qtime}


# TODO:
    # convert times to meaningful results

#from qiskit.providers.ibmq.runtime import UserMessenger
#msg = UserMessenger()
# Use the local Aer simulator
#from qiskit import Aer
#backend = Aer.get_backend('qasm_simulator')

#provider = IBMQ.get_provider(hub='strangeworks-hub', group='science-team', project='science-test')
#backend = provider.get_backend("ibm_nairobi") #, account_id="koenmesman")


# Execute the main routine for our simple two-qubit Hamiltonian H, and perform 5 iterations of the SPSA solver.
#main(backend, msg, graph)