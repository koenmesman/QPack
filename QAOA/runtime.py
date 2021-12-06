#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Created on Mon Nov 29 20:54:07 2021

@author: koen
"""
import time
from qiskit import IBMQ
import custom_tool
from qiskit.tools import job_monitor

def runtime_mcp(provider, backend, max_iter, shots):
    time_res = {"START":time.time()}
    
    #IBMQ.load_account()
    #provider = IBMQ.get_provider(hub='strangeworks-hub', group='science-team', project='science-test')
    
    meta = {
      "name": "sample-qaoa",
      "description": "A sample QAOA program.",
      "max_execution_time": 1000000,
      "spec": {}
    }
    
    meta["spec"]["interim_results"] = {
      "$schema": "https://json-schema.org/draft/2019-09/schema",
      "description": "Parameter vector at current optimization step. This is a numpy array.",
      "type": "array"
    }
    
    
    program_id = provider.runtime.upload_program(data='qaoa_runtime_sample.py', metadata=meta)
    program_id
    
    prog = provider.runtime.program(program_id)
    #print(prog)
    
    interm_results = []
    #def qaoa_callback(job_id, data):
    #    interm_results.append(data)
    queue_times = []
    
    def interim_result_callback(job_id, interim_result):
        print(f"interim result: {interim_result}")
        #print()
        
        
    #backend = provider.get_backend("ibm_nairobi")
    #backend = provider.get_backend("ibmq_qasm_simulator")
    inputs = {"shots":shots}
    options = {'backend_name': backend.name()}
    
    #job = provider.runtime.run(program_id, options=options, inputs=inputs)
    #job.result()
    start = time.time()
    time_res["JOB_START"] = start
    job2 = provider.runtime.run(program_id, options=options, inputs=inputs, callback=interim_result_callback)
    #tally(job_monitor(job2))
    #job_monitor(job2)
    custom_tool.custom_job_monitor(job2, provider, output=time_res)
    result = job2.result()
    time_res["JOB_END"] = time.time()
    time_res["QTIME"] = result["qtime"]
    time_res["OPT_TIME"] = result["opt_time"]

    
    
    keys = ["size", "shots", "layers", "iter", "eval", "score", "acc", "times", "datetime"]
    return[-result["result"]["fun"], time_res, result["result"]["nfev"], result["result"]["nit"]]
