#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:50:16 2021

@author: koen
"""

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""A module for monitoring various qiskit functionality"""

import sys
import time
global queue_time
import os
import json

def _text_checker(
    job, interval, provider, _interval_set=False, quiet=False, output=sys.stdout, line_discipline=""#"\r"
):
    """A text-based job status checker

    Args:
        job (BaseJob): The job to check.
        interval (int): The interval at which to check.
        _interval_set (bool): Was interval time set by user?
        quiet (bool): If True, do not print status messages.
        output (file): The file like object to write status messages to.
        By default this is sys.stdout.
        line_discipline (string): character emitted at start of a line of job monitor output,
        This defaults to \\r.

    """
    job_id = job.job_id
    retrieved_job = provider.runtime.job(job.job_id())
    status = retrieved_job.status()

    #status = job.status()
    msg = status.value
    prev_msg = msg
    msg_len = len(msg)

    #if not quiet:
        #print("{}{}: {}".format(line_discipline, "Job Status", msg), end="")#, file=output)
    while status.name not in ["DONE"]:#, "CANCELLED", "ERROR"]:
        time.sleep(1)
        status = job.status()
        #print("status: {}".format(status))
        msg = status.value
        if status.name == "VALIDATING":
            #print("interim time: {}".format(time.time()), file=sys.stdout)
            if "VALIDATING" not in output.keys():
                _intermediate_log("VALIDATING", time.time())
                output["VALIDATING"] = time.time()
        elif status.name == "QUEUED":
            #print("interim time: {}".format(time.time()), file=sys.stdout)
            if "QUEUE" not in output.keys():
                output["QUEUE"] = time.time()
            #print(output)
            #msg += " (%s)" % job.queue_position()
            #if job.queue_position() is None:
            #    interval = 2
            #elif not _interval_set:
            #    interval = max(job.queue_position(), 2)
        elif status.name == "RUNNING":
            if "RUNNING" not in output.keys():
                output["RUNNING"] = time.time()
                _intermediate_log("VALIDATING", time.time())

        elif status.name == "DONE":
            if "DONE" not in output.keys():
                output["DONE"] = time.time()
                _intermediate_log("VALIDATING", time.time())
        else:
            if not _interval_set:
                interval = 2

        # Adjust length of message so there are no artifacts
        if len(msg) < msg_len:
            msg += " " * (msg_len - len(msg))
        elif len(msg) > msg_len:
            msg_len = len(msg)

        if msg != prev_msg and not quiet:
            #print("{}{}: {}{}".format(line_discipline, "Job Status", msg), end="")#, file=output)
            prev_msg = msg
    if not quiet:
        print("")#, file=output)
    return output

def _intermediate_log(key, output):
        work_dir = os.path.dirname(os.path.realpath(__file__))
        path = work_dir + "/logs/inter_log_mcp_runtime.json"

        try:
            with open(path,'r+') as file:
                file_data = json.load(file)
                file_data[key] = output
                file.seek(0)
                json.dump(file_data, file, indent = 4)
        except:
            with open(path, 'w') as file:
                #file_data = json.load(file)
                file_data = {key: output}
                #file_data[key] = output
                file.seek(0)
                json.dump(file_data, file, indent = 4)
        finally:
            print("intermediate data saved")


def custom_job_monitor(job, provider, interval=None, quiet=False, output=sys.stdout, line_discipline="\r"):
    """Monitor the status of a IBMQJob instance.

    Args:
        job (BaseJob): Job to monitor.
        interval (int): Time interval between status queries.
        quiet (bool): If True, do not print status messages.
        output (file): The file like object to write status messages to.
        By default this is sys.stdout.
        line_discipline (string): character emitted at start of a line of job monitor output,
        This defaults to \\r.
    """
    if interval is None:
        _interval_set = False
        interval = 5
    else:
        _interval_set = True

    step_times = _text_checker(
        job, interval, provider, _interval_set, quiet=quiet, output=output, line_discipline=line_discipline
    )
    return step_times