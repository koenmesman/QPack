# QPack
In this repository,the QPack Benchmark for quantum computing is presented. QPack aims to set the standard for evaluating quantum- hardware and simulators, using meaningful metrics and practical applications. The QPack benchmark makes use of quantum algorithms, applied to different applications, to evaluate quantum computers and quantum simulators. The quantum applications are implemented as they would be in practice, using minimal (quantum) resources and aiming for an optimal performance. As more quantum applications become practical on quantum hardware, these applications will be included in the QPack benchmark.

The schematic outline for the QPack benchmark is shown below ![here](https://github.com/koenmesman/QPack/blob/main/Benchmark_schematic.png?raw=true).
This implementation is aimed at the IBM systems, and supports the IBM runtime environment. A XACC implementation of QPack can be found [here](https://github.com/huub-d96/xacc_qaoa_benchmarks), and can be used for XACC supported backends.

The QPack benchmark currently includes the following benchmarks:
QAOA for:
 * MaxCut problem (MCP)
 * Dominating set problem (DSP)
 * Traveling salesman problem (TSP)

To run the benchmark simply download the repo and edit the main file with the system you want to benchmark. Make sure to have your IBMQ credentials set up.
The results are saved as a JSON file in the 'logs' directory.



Further information about QPACK can be found on [Arxiv.org](https://arxiv.org/abs/2103.17193).
