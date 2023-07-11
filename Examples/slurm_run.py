#!/usr/bin/env python

import os
import numpy as np
import sys
import itertools

hostname = os.uname()[1]


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir): os.mkdir(dir)


job_directory = "%s" %os.getcwd()

# Make top level directories
# mkdir_p(job_directory)
mkdir_p(job_directory+'/job')

# list of AOD/nPCA
tau = -np.logspace(np.log10(0.01), np.log10(2.0), 20)
# splitting into chunks to make the code efficient and run easily in DISCOVER
try:
    arrayNum= int(sys.argv[1])
except:
    print('No array number is given: it should be 0,1 or 2 \n Using default value 0')
    arrayNum = 0

npca_ = [range(0,36), range(36,72), range(72, 107)]
npca = npca_[arrayNum] # max is 107

# Solar Zenith angle (used if no real sun-satelliote geometry is used)
SZA = 30
sza_ = [0, 30, 60]# For running multible simulations in DISCOVER
sza = list(itertools.chain.from_iterable(itertools.repeat(x, 12) for x in sza_))
# realGeometry: True if using the real geometry provided in the .mat file
useRealGeometry = 1
# Job name
jobName = 'T%d_' %arrayNum # 'A' for 2modes, 'Z' for realGeometry
if not useRealGeometry: jobName = jobName + str(SZA); varStr = 'aod'
else: varStr = 'nPCA'

# Instrment name
instrument = 'megaharp01'

# looping through the var string
for aod in tau:
    if useRealGeometry: aod_ = aod*1000; fileName = '%04d' %(aod_)
    else: aod_ = aod*1000; fileName = '%.4d' %(aod_)
    
    job_file = os.path.join(job_directory,
                            "j_%s_%s_%s.slurm" %(jobName, varStr, fileName))

    # Create directories
    mkdir_p(job_directory)

    with open(job_file, 'w') as fh:
        fh.writelines("#!/usr/local/bin/bash\n\n")
        fh.writelines("#SBATCH --job-name=%s%.4d\n" % (jobName, aod_))
        fh.writelines("#SBATCH --output=./job/%s_%.4d.out.%s\n" % (jobName, aod_, '%A'))
        fh.writelines("#SBATCH --error=./job/%s_%.4d.err.%s\n" % (jobName, aod_, '%A'))
        fh.writelines("#SBATCH --time=01:59:59\n")
        # In Discover
        if 'discover' in hostname:
            fh.writelines('#SBATCH --constraint="sky|cas"\n')
            fh.writelines("#SBATCH --ntasks=36\n")
            # fh.writelines("#SBATCH --array=0\n")
        # In Uranus
        elif 'uranus' in hostname:
            fh.writelines("#SBATCH --partition=LocalQ\n")
        # fh.writelines("#SBATCH --qos=short\n")
        # fh.writelines("#SBATCH --nodes=1\n")
        # fh.writelines("#SBATCH --ntasks-per-node=1\n\n")
        # fh.writelines("#SBATCH --mail-type=ALL\n")
        # fh.writelines("#SBATCH --mail-user=$USER@umbc.edu\n")
        fh.writelines("date\n")
        fh.writelines("hostname\n")
        fh.writelines('echo "---Running Simulation---"\n')
        fh.writelines("date\n")
        if useRealGeometry:
            if 'discover' in hostname:
                for i in npca:
                    fh.writelines("python runRetrievalSimulationSlurm.py %d %s %s %s %2.3f &\n" %(int(i), instrument,
                                                                                           SZA, useRealGeometry, aod))
            else:
                fh.writelines("python runRetrievalSimulationSlurm.py %.4f %s %s %s\n" %(aod, instrument, SZA, useRealGeometry))
        else:
            if 'discover' in hostname:
                temp_num = 1
                for i in sza:
                    fh.writelines("python runRetrievalSimulationSlurm.py %d %s %s %s %2.3f &\n" %(int(i), instrument,
                                                                                           (i+((arrayNum*1.3)/10+temp_num/100)), useRealGeometry, aod))
                    temp_num += 1
            else:
                fh.writelines("python runRetrievalSimulationSlurm.py %.4f %s %s %s\n" %(aod, instrument, SZA, useRealGeometry))
        fh.writelines("wait\n")
        fh.writelines("echo 0\n")
        fh.writelines("echo End: \n")
        fh.writelines("rm -rf ${TMPDIR}\n")
        fh.writelines("date")
        # fh.writelines("python hello.py %.3d" % aod_)
    fh.close()
    os.system("sbatch %s" %job_file)
print('Jobs sumitted succesfully check the ./job/ folder for output/error')
