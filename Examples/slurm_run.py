#!/usr/bin/env python

import os
import numpy as np

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = "%s" %os.getcwd()

# Make top level directories
# mkdir_p(job_directory)
# mkdir_p(data_dir)

tau = np.logspace(np.log10(1.1), np.log10(2), 2)
jobName = 'A'
SZA = 30
jobName = jobName + str(SZA)

instrument = 'megaharp01'



for aod in tau:
    aod_ = aod*1000
    fileName = '%.4d' %(aod_)
    job_file = os.path.join(job_directory,
                            "job_%s_aod_%s.slurm" %(jobName, fileName))

    # Create directories
    mkdir_p(job_directory)

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n\n")
        fh.writelines("#SBATCH --job-name=%s%.4d\n" % (jobName, aod_))
        fh.writelines("#SBATCH --output=%s_%.4d.out\n" % (jobName, aod_))
        fh.writelines("#SBATCH --error=%s_%.4d.err\n" % (jobName, aod_))
        fh.writelines("#SBATCH --time=11:29:59\n")
        fh.writelines("#SBATCH --partition=LocalQ\n")
        # fh.writelines("#SBATCH --qos=short\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --ntasks-per-node=10\n\n")
        fh.writelines("#SBATCH --mail-type=ALL\n")
        fh.writelines("#SBATCH --mail-user=$USER@umbc.edu\n")
        fh.writelines("python runRetrievalSimulationSlurm.py %.4f %s %s" %(aod, instrument, SZA))
        # fh.writelines("echo %s" %jobName)
        # fh.writelines("python hello.py %.3d" % aod_)
    fh.close()
    os.system("sbatch %s" %job_file)