import string
import os

def yeti_script(name, command, hours=5, memory=1):

    yeti_str = """#!/bin/sh

# Directives
#PBS -N %s
#PBS -W group_list=yetidsi
<<<<<<< HEAD
#PBS -l nodes=1,walltime=0%s:00:00,mem=%sgb
=======
#PBS -l nodes=1,walltime=10:00:00,mem=2000mb
>>>>>>> da936040b9ff2f2325b6b0555162a2a0c80e0f9f
#PBS -M ll3105@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/u/4/l/ll3105/sentiment-analysis/experiment/log
#PBS -e localhost:/u/4/l/ll3105/sentiment-analysis/experiment/log

%s

# End of script""" % (name, hours, memory, command)

    return yeti_str



def habanero_script(name, command, hours=5, memory=1):

    job_str = """#!/bin/sh
#
#SBATCH --account=dsi            # The account name for the job.
#SBATCH --job-name=%s         # The job name.
#SBATCH --error=log/%s.e
#SBATCH --output=log/%s.o
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=%s:00:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=%sgb        # The memory the job will use per cpu core.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liping.liulp@gmail.com

%s

# End of script""" % (name, name, name, hours, memory, command)
    return job_str


settings = ['exposure=False', 'use_sideinfo=False', 'K=512'] 

name = 'w2v'
ss = 'python run_experiment.py '
for setting in settings:
    k, v = string.split(setting, '=') 
    name = name +  k[0] + v + '_'
    ss = ss + setting + ' '

script = habanero_script(name, ss, hours=10, memory=2)

print(script)
with open('job.sh', 'w') as jfile:
    jfile.write(script)

os.system('sbatch job.sh')


