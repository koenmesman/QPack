import generate_graph as gg
import Classic_opt as opt
import time
import json
import datetime
import traceback
from utils import get_times, merge_times, format_runtime
global BENCHMARK_TIMES
import exact_solver as exact
import os
import stat
import runtime as runtime_lib

#Workflow qaoa instance:
#   define problem size (qubits), optionally adjust for TSP (though strangeworks does not support >9 qubits)
#   create graph instance from 'generate_graph.py'
#   you can change the initial parameters for [beta, gamma], iterations p and the nr. of quantum circuit repetitions
#   define [beta, gamma] as [beta[0], beta[1], ..., beta[p-1], gamma[0], ... , gamma[p-1]]
#   the qaoa optimizer reads this as beta =  param[0:p], gamma = [p:2p]
#   (rep = 100 is more realistic, but for testing purposes use a small number)
#   define q_func as either "mcp" (max-cut), "tsp" (traveling salesman) or "dsp" (dominating set)
#   result format is: [best_solution_result, [best_beta, best_gamma]]

#   data_stream format: {"func":str, "backend_tag":str, "data":{"size":int, "shots":int, "layers":int, "iter":"int,  "score":{float}, "acc":{float} "times":{float}}}

class Benchmark:
    def __init__(self, func):
        self.init_param = [0,0]
        self.p = 1
        self.rep = 10
        self.q_func = func
        self.max_size = []
        self.qvm = ""
        self.provider = ""
        self.lim = 10
        self.backend = ""
        self.max_iter = 1000
        self.score=0
        self.shots=100
        self.runtime = False
        date = datetime.date.today()
        time = datetime.datetime.now()
        self.datetime = (str(date)+"_" + str(time.hour)+"-"+str(time.minute))
        self.stream = {"size":0, "shots":100, "layers":self.p, "iter":1, "eval":0, "score":0, "acc":0, "times":{}, "datetime":self.datetime}
        global BENCHMARK_TIMES
        BENCHMARK_TIMES = {'CREATING': 0, 'VALIDATING': 0, 'QUEUED': 0, 'RUNNING': 0,
                           'OTHER': 0, 'CONNECTION': 0}

    def __qubit_select(self):
        self._qbits = {
            'mcp': self._problem_size,
            'dsp': self._problem_size + 10,
            'tsp': self._problem_size**2
        }
        return self._qbits.get(self.q_func)
    
    def __update_data(self):
        work_dir = os.path.dirname(os.path.realpath(__file__))
        if self.runtime:
            path = work_dir + "/logs/log_{}_{}_runtime.json".format(self.q_func, self.backend_tag)
        else:
            path = work_dir + "/logs/log_{}_{}.json".format(self.q_func, self.backend_tag)
        try:
            with open(path,'r+') as file:
                file_data = json.load(file)
                
                old_data = file_data["data"] 
                data = old_data.copy()
                data.append(self.stream)
                file_data = {"qfunc" :"mcp", "backend_tag":self.backend_tag, "data":data}
                file.seek(0)
                json.dump(file_data, file, indent = 4)
        except:
            with open(path, 'w') as file:
                file_data = {"qfunc" :self.q_func, "backend_tag":self.backend_tag, "data":[self.stream]}
                file.seek(0)
                json.dump(file_data, file, indent = 4)
        finally:
            print("data saved")

        

    def update_p(self, new_p):
        self.p = new_p

    def set_lim(self, lim):
        self.lim = lim+1

    def set_iter(self, max_iter):
        # set max number of iterations for optimizer
        self.max_iter = max_iter

    def set_backend(self, provider, tag):
        self.backend = provider.get_backend(tag)
        self.provider = provider
        self.backend_tag = tag
    
        
    def accuracy(self, size):
        optimal = 0.001 
        solver = {
            "mcp":exact.mcp_solver,
            "dsp":exact.dsp_solver,
            "tsp":exact.tsp_solver
            } 
        
        xsolver = solver.get(self.q_func)
        optimal = xsolver(size)
        return self.score/(optimal)

    def run(self, runtime=False):
        self.runtime = runtime
        self._problem_size = 5
        out = []
        keys = ["size", "shots", "layers", "iter", "eval", "score", "acc", "times", "datetime"]
        global BENCHMARK_TIMES
        while self._problem_size < self.lim:
            self.qubits = self.__qubit_select()

            try:
                self.graph = gg.regular_graph(self._problem_size)
                self.results = 0
                print('start', self.q_func, self._problem_size)
                
                if not runtime:
                    self._start = time.time()
                    #self._results = opt.nm(self.init_param, self.graph, self.p, self.q_func)
                    # TODO: enable max_iter
                    self._results = opt.shgo_fun(self.init_param, self.graph, self.p, self.q_func, self.backend)
                    
                    self._time = time.time() - self._start
                    self._time_dict = BENCHMARK_TIMES
                else:
                    self._start = time.time()
                    # enable options
                    self._results = runtime_lib.runtime_mcp(self.provider, self.backend, self.max_iter, self.shots)
                    self._time_dict = self._results[1]
                    self._time = time.time()


                # TODO: specify score
                self.score = self._results
                self._time_dict['WALLTIME'] = self._time
                self.stream["size"] = self._problem_size
                self.stream["score"] = self._results[0]
                self.stream["eval"] = self._results[2]
                self.stream["iter"] = self._results[3]
                self.score = self._results[0]
                self.stream["acc"] = self.accuracy(self._problem_size)
                self._formatted_times = format_runtime(self._time_dict)
                
                self.stream["times"] = self._formatted_times
                
                self.__update_data()
                
                self._problem_size += 1
            except Exception as e:
                print(e)
                traceback.print_exc()
                self._problem_size -= 1
                break
        print('finished!')
        status = "completed"
        return status
    

        
        #try:
        #    #outfile = open('./data/log_'+ stream["qfunc"] + "_" + self.backend_tag + '.json', 'r')
        #    #data = json.load(outfile)

            
        #except:
        #    outfile = open('./data/log_'+ stream["qfunc"] + "_" + self.backend_tag + '.json', 'w')
        #    json.dump(stream, outfile, indent=2)
        #finally:
        #    outfile.close()

