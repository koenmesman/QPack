#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:02:15 2021

@author: koen
"""
import datetime
import json
import matplotlib.pyplot as plt

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
