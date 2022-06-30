#!/usr/bin/env python

import os
import numpy as np

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir): os.mkdir(dir)

job_directory = "%s" %os.getcwd()

# Make top level directories
# mkdir_p(job_directory)
mkdir_p(job_directory+'/job')

# list of AOD/nPCA
# tau = np.logspace(np.log10(0.01), np.log10(2.1), 1)
tau = range(46,108) # max is 107
# Solar Zenith angle (used if no real sun-satelliote geometry is used)
SZA = 30
# realGeometry: True if using the real geometry provided in the .mat file
useRealGeometry = 1
# Job name
jobName = 'Y' # 'A' for 2modes, 'Z' for realGeometry
if not useRealGeometry: jobName = jobName + str(SZA); varStr = 'aod'
else: varStr = 'nPCA'

# Instrment name
instrument = 'megaharp01'

# looping through the var string
for aod in tau:
    if useRealGeometry: aod_ = aod; fileName = '%04d' %(aod_)
    else: aod_ = aod*1000; fileName = '%.4d' %(aod_)
    
    job_file = os.path.join(job_directory,
                            "j_%s_%s_%s.slurm" %(jobName, varStr, fileName))

    # Create directories
    mkdir_p(job_directory)

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n\n")
        fh.writelines("#SBATCH --job-name=%s%.4d\n" % (jobName, aod_))
        fh.writelines("#SBATCH --output=./job/%s_%.4d.out\n" % (jobName, aod_))
        fh.writelines("#SBATCH --error=./job/%s_%.4d.err\n" % (jobName, aod_))
        fh.writelines("#SBATCH --time=11:29:59\n")
        #fh.writelines("#SBATCH --partition=LocalQ\n")
        # fh.writelines("#SBATCH --qos=short\n")
        fh.writelines('#SBATCH --nodes=1 --constraint="sky|cas"\n')
        fh.writelines("#SBATCH --array=0\n")
        #fh.writelines("#SBATCH --nodes=1\n")
        #fh.writelines("#SBATCH --ntasks-per-node=1\n\n")
        # fh.writelines("#SBATCH --mail-type=ALL\n")
        # fh.writelines("#SBATCH --mail-user=$USER@umbc.edu\n")
        fh.writelines("\necho Start: \n")
        fh.writelines("date\n")
        fh.writelines("python runRetrievalSimulationSlurm.py %.4f %s %s %s\n" %(aod, instrument, SZA, useRealGeometry))
        fh.writelines("echo End: \n")
        fh.writelines("date")
        # fh.writelines("python hello.py %.3d" % aod_)
    fh.close()
    os.system("sbatch %s" %job_file)
