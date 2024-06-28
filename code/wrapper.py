#!/usr/bin/env python2.7
# encoding: utf-8

import sys, os, time, re
import main
from subprocess import Popen, PIPE
from multiprocessing import Process
from threading import Timer
import logging 

def run(args, timeout_sec):
    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    timer = Timer(timeout_sec, proc.kill)
    try:
        timer.start()
        stdout, stderr = proc.communicate()
        return stdout
    finally:
        timer.cancel()

def popen_timeout(args, timeout):
    p = Popen(args, stdout=PIPE, stderr=PIPE)
    for t in xrange(timeout):
        time.sleep(1)
        if p.poll() is not None:
            return p.communicate()
    p.kill()
    return False

#logging.basicConfig(filename='logz.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')
#logging.warning(sys.argv[1::])

# Read in first 5 arguments.
instance = sys.argv[1]
specifics = sys.argv[2]
cutoff = int(float(sys.argv[3]) + 1)
runlength = int(sys.argv[4])
seed = int(sys.argv[5])

runtime = cutoff


# Read in parameter setting and build a dictionary mapping param_name to param_value.
params = sys.argv[6:]
configMap = dict((name, value) for name, value in zip(params[::2], params[1::2]))

#wce_path = os.path.dirname(os.path.realpath(__file__))+"/main.py"
#wce_path = "./main.py"
#cmd = "python2.7 %s" %(wce_path)       
#for name, value in configMap.items():
#    cmd += " -%s %s" %(name,  value)
args = [instance]
for name, value in configMap.items():
    args.append('%s' %(name))
    args.append('%s' %(value))

# Execute the call and track its runtime.
#start_time = time.time()
#runtime = run(args, cutoff)
#print(runtime)
#proc = Popen(args, stdout=PIPE, stderr=PIPE)
#stdout, stderr = proc.communicate()
#print("Result: {} Error: {}".format(stdout, stderr))
#runtime = time.time() - start_time

start_time = time.time()
p = Process(target=main.main, args=[args])
p.start()
p.join(cutoff)
p.terminate()
runtime = time.time() - start_time

#TODO: Check if result is correct

status = "CRASHED"
if (runtime >= cutoff):
    status = 'TIMEOUT'
if (runtime < cutoff):
    status = 'SUCCESS'

#print("runtime: {}".format(runtime))
#print("cutoff: {}".format(cutoff))
#print("status: {}".format(status))
# Output result for SMAC.
print("Result for SMAC: %s, %s, 0, 0, %s" % (status, str(runtime), str(seed)))

#./smac --scenario-file ./example_scenarios/wce/wce-scenario.txt --seed 1234 --num-validation-runs 10
# to execute rapper only
#python2.7 wrapper.py ./2-real-world/w001.dimacs 0 5 0 1234 lb2 1 heavy_none_edge 10 greedy 5