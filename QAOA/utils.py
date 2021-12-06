#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:02:15 2021

@author: koen
"""
import datetime
import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def plot_json(q_func, backend_tag):
    with open('./data/log_'+ q_func + "_" + backend_tag + '.json','r+') as file:
        file_data = json.load(file)
        data = file_data["data"].copy()
        
        size = []
        acc = []
        iterat = []
        shots = []
        evalu = []
        score = []
        times = []
        
        
        for stream in data:
           size.append(stream['size'])
           score.append(stream['score'])
           acc.append(stream['acc'])
           iterat.append(stream['iter'])
           evalu.append(stream['eval'])
           times.append(stream['times'])
           shots.append(stream['shots'])
           
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.plot(size, evalu, '.')
        ax2.plot(size, score, '.')
        ax3.plot(size, acc, '.')
        ax4.plot(size, iterat, '.')
        
        #fig2, ax = plt.plot()
        #ax.plot()
    
    
def get_times(job, wall):
    
    create = job.get('CREATED')-job.get('CREATING')
    validate = job.get('VALIDATED') - job.get('VALIDATING')
    if job.get('RUNNING'): # and job.get('QUEUED'):
        queue = job.get('RUNNING') - job.get('QUEUED')
    else:
        queue = datetime.timedelta()    # microsecond resolution

    if job.get('RUNNING'):
        runtime = job.get('COMPLETED') - job.get('RUNNING')
    else:
        runtime = datetime.timedelta()

    other = job.get('COMPLETED') - job.get('CREATING') - create - validate - queue - runtime
    total = job.get('COMPLETED') - job.get('CREATING')
    out_dict = {'CREATING': create, 'VALIDATING': validate, 'QUEUED': queue, 'RUNNING': runtime, 'OTHER': other, 'CONNECTION': total}
    for key, val in out_dict.items():
        out_dict[key] = val.total_seconds()
    out_dict['CONNECTION'] -= wall
    return out_dict     # seconds


def merge_times(old_times, new_times):
    for key, old_val in old_times.items():
        old_times[key] = old_val + new_times[key]
    return old_times


def format_runtime(times):
        formatted = {}
        formatted["UPLOAD"] = times["JOB_START"] - times["START"]           # upload job to remote CPU
        formatted["COMMUNICATION"] = times["QUEUE"] - times["JOB_START"]+(times["JOB_END"]-times["DONE"])    # Verficationd and ethernet
        formatted["QUEUE"] = times["RUNNING"] - times["QUEUE"]              # Time in queue
        formatted["QTIME"] = times["QTIME"]
        formatted["OPT_TIME"] = times["OPT_TIME"]
        formatted["LATENCY"] = times["DONE"]-times["RUNNING"]-times["OPT_TIME"]-times["QTIME"]
        formatted["POST"] = times["WALLTIME"] - times["JOB_END"]              # Post processing
        return formatted

def parse_results(backends, runtime=False, q_func="mcp", size=0):
    if not runtime:
        for backend_tag in backends:
            with open('./logs/log_{}_{}.json'.format(q_func, backend_tag),'r') as file:
                file_data = json.load(file)
                data = file_data["data"]
                s = []
                t = {}
                if size ==0:
                    for d in data:
                        s.append(d["size"])
                        d["CLASSICAL"] = d.pop("WALLTIME")
                        for key in d["times"].keys():
                            try:  t[key].append(d["times"][key])
                            except:
                                t[key] = []
                                t[key].append(d["times"][key])
                else:
                    for d in data:
                        if d["size"]==size:
                            s.append(d["size"])
                            d["times"]["CLASSICAL"] = d["times"].pop("WALLTIME")
                            for key in d["times"].keys():
                                try:  t[key].append(d["times"][key])
                                except:
                                    t[key] = []
                                    t[key].append(d["times"][key])
                
                if len(Counter(s).keys())>1:
                    for key in t.keys():
                        #plt.plot(s, t[key], '.')
                        print()
                else:
                    total = []
                    mean_times = []
                    std_times = []
                    for i in  range(len(t["CREATING"])):   # any key will do
                        total.append(0)
                        for key in t.keys():
                            total[i] += t[key][i]
                    perc = [0]*len(t["CREATING"])
                    for key in t.keys():
                            mean_times.append(np.mean(t[key]))
                            std_times.append(np.std(t[key]))
                            for i in  range(len(t[key])):
                                perc[i] += t[key][i]/total[i]
                    fig, ax = plt.subplots()
                    x_pos =  np.arange(len(t.keys()))
                    ax.bar(x_pos, mean_times, yerr=std_times, align='center', alpha=0.5, ecolor='black', capsize=10)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(t.keys())
                    plt.xticks(rotation=45)
                    plt.title("{}: {}, size:{}, shots={}".format(backend_tag, q_func, s[0], data[0]["shots"]))