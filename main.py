#!/usr/bin/env python3
from Benchmark import Benchmark
import utils
from qiskit import IBMQ, Aer

IBMQ.load_account()
#provider = IBMQ.get_provider(hub='strangeworks-hub')
provider = IBMQ.get_provider(hub='ibm-q')

#backend_tag = 'aer_simulator'
#backend_tag = 'ibm_nairobi'
backend_tag = 'ibmq_qasm_simulator'
backend = provider.get_backend(backend_tag)

#MCP benchmark
mcp = Benchmark('mcp')
mcp.set_backend(backend, backend_tag)
mcp.set_lim(15)
mcp.set_iter(100)
for i in range(10):
    mcp.run()
#print(mcp.stream)

#DSP benchmark
#dsp = Benchmark('dsp')
#dsp.set_lim(20)
#dsp.run()

#TSP benchmark
#tsp = Benchmark('tsp')
#tsp.run()
#print(tsp.max_size)



# ToDo
#   verify results --> score
#   set result threshold / measure fidelity (quantum simulator?)
#   measure time
#   measure memory usage
#   define output format (JSON)
#       {problem, best, {size, p, score, time, memory usage}}
#   create output file