import string
import os

settings = ['exposure=True', 'use_sideinfo=True', 'K=16'] 

name = ''
ss = ''
for setting in settings:
    k, v = string.split(setting, '=') 
    name = name +  k[0] + v + '_'
    ss = ss + setting + ' '

job_str = """#!/bin/sh

# Directives
#PBS -N w2v%s
#PBS -W group_list=yetidsi
#PBS -l nodes=1,walltime=05:00:00,mem=1000mb
#PBS -M ll3105@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/u/4/l/ll3105/sentiment-analysis/experiment/log
#PBS -e localhost:/u/4/l/ll3105/sentiment-analysis/experiment/log

tfpy restaurant_experiment.py %s

# End of script""" % (name, ss)


with open('job.sh', 'w') as jfile:
    jfile.write(job_str)

os.system('qsub job.sh')
