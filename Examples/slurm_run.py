#!/usr/bin/env python
'''
This code is used to run the retrieval simulation code in parallel using slurm
The code is split into 5 chunks to run in parallel in NyX
The code is split into 3 chunks to run in parallel in Discover

'''
#--------------------------------------------#
# Importing the required modules
#--------------------------------------------#
import os
import numpy as np
import sys
import itertools
import yaml

#--------------------------------------------#
# Checking the input arguments
#--------------------------------------------#

if sys.argv[1] == '--help':
    print('options: \n --dry-run: to run the code without submitting the jobs \n --help: to print this message')
    print('Usage: python slurm_run.py --dry-run')
    print('Usage: python slurm_run.py 0 for running the first chunk of simulations')
    print('Usage: python slurm_run.py 0 Triangular for running simulations with triangular bin distribution\nDefault is BiModal')
    print('The code is split into 5 chunks to run in parallel in NyX')
    print('<><><><> Printing help message <><><><>')
    sys.exit()
hostname = os.uname()[1]

#--------------------------------------------#
# Creating the directories
#--------------------------------------------#
def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir): os.mkdir(dir)

job_directory = "%s" %os.getcwd()

# Make top level directories
mkdir_p(job_directory+'/job')

# --------------------------------------------------------------------------- #
# Load the YAML settings files for the retrieval simulation configuration
# --------------------------------------------------------------------------- #
# Configuration file YAML

# sys argv is used to define the PSD distribution, default is BiModal
# example: python slurm_run.py 0 Triangular
#          python slurm_run.py 0 BiModal
if len(sys.argv) > 2:
    conf_ = sys.argv[2]
else:
    conf_ = 'BiModal' # or 'Triangular'

# This template is used for the finding the correct retrieval simulation configuration
yamlFile = '../ACCP_ArchitectureAndCanonicalCases/camp2ex-configurations.yml'
with open(yamlFile, 'r') as f:
    ymlData = yaml.load(f, Loader=yaml.FullLoader)

#--------------------------------------------#
# Setting the parameters
#--------------------------------------------#

# Number of nodes
nNode = int(ymlData['default']['run']['nNodes']) # Ideally, a max of 7 is used in NyX because of the 7 ryzen nodes, this has been automated to use the number from yaml file

# list of AOD/nPCA. Even though the code is written to run multiple AODs, it is not used directly when the sign is negative
tau = -np.logspace(np.log10(0.01), np.log10(4.0), nNode)

# splitting into chunks to make the code efficient and run easily in DISCOVER
try:
    arrayNum= int(sys.argv[1])
except:
    print('No array number is given: it should be 0,1 or 2 \n Using default value 0')
    arrayNum = 0
if 'nyx' in hostname:
    npca_ = [range(0,22), range(22,44), range(44, 66), range(66, 88), range(88, 107)] # modified for NyX
# Discover has 36 cores per node
elif 'discover' in hostname:
    npca_ = [range(0,36), range(36,72), range(72, 108)] # modified for Discover
npca = npca_[arrayNum] # max is 107

#--------------------------------------------#
# Solar Zenith angle (used if no real sun-satellite geometry is used)
#--------------------------------------------#
SZA = 30
sza_ = [0, 30, 60] # For running multiple simulations in DISCOVER
sza = list(itertools.chain.from_iterable(itertools.repeat(x, 12) for x in sza_))

# realGeometry: True if using the real geometry provided in the .mat file
useRealGeometry = 1

# Job name
jobName = 'Y%d' %arrayNum # 'A' for 2modes, 'Z' for realGeometry
if not useRealGeometry: jobName = jobName + str(SZA); varStr = 'aod'
else: varStr = 'nPCA'

# Instrment name
instrument = ymlData['default']['forward']['instrument']

#--------------------------------------------#
# looping through the var string
#--------------------------------------------#

# looping through the AOD to create the slurm files, to distribute the jobs
for aod in tau:
    if useRealGeometry: aod_ = aod*1000; fileName = '%04d' %(aod_)
    else: aod_ = aod*1000; fileName = '%.4d' %(aod_)
    
    job_file = os.path.join(job_directory,
                            "j_%s_%s_%s.slurm" %(jobName, varStr, fileName))

    # Create directories
    mkdir_p(job_directory)

    #--------------------------------------------#
    # Creating the slurm file
    #--------------------------------------------#
    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n\n")
        fh.writelines("#SBATCH --job-name=%s%.4d\n" % (jobName, aod_))
        fh.writelines("#SBATCH --output=./job/%s_%.4d.out.%s\n" % (jobName, aod_, '%A'))
        fh.writelines("#SBATCH --error=./job/%s_%.4d.err.%s\n" % (jobName, aod_, '%A'))
        fh.writelines("#SBATCH --time=2-23:59:59\n")
        # In Discover
        if 'discover' in hostname:
            fh.writelines('#SBATCH --constraint="[sky"\n')
            fh.writelines("#SBATCH --ntasks=40\n")
        # In Uranus
        elif 'uranus' in hostname:
            fh.writelines("#SBATCH --partition=LocalQ\n")
        # in NyX
        elif 'nyx' in hostname:
            # Selecting the partition
            if ymlData['default']['run']['partition'] == 'zen4':
                fh.writelines("#SBATCH --partition=zen4\n")
                fh.writelines("#SBATCH --ntasks=22\n")
                fh.writelines("#SBATCH --mem=100G\n")
            else:
                fh.writelines("#SBATCH --partition=zen3\n")
                fh.writelines("#SBATCH --ntasks=16\n")
                fh.writelines("#SBATCH --mem=26G\n")
        fh.writelines("date\n")
        fh.writelines("hostname\n")
        fh.writelines('echo "---Running Simulation---"\n')
        fh.writelines("date\n")

        #--------------------------------------------#
        # Scanning through the AOD/nPCA
        #--------------------------------------------#
        if useRealGeometry:
            if 'discover' in hostname:
                for i in npca:
                    fh.writelines("python runRetrievalSimulationSlurm.py %2d %s %s %s %2.3f %s &\n" %(int(i), instrument,
                                                                                           SZA, useRealGeometry, aod, conf_))
            elif 'nyx' in hostname:
                for i in npca:
                    fh.writelines("python runRetrievalSimulationSlurm.py %2d %s %s %s %2.3f %s &\n" %(int(i), instrument,
                                                                                           SZA, useRealGeometry, aod, conf_))
            else:
                # dry run option for MBP or other local machines
                if sys.argv[1] == '--dry-run':
                    for i in npca:
                        fh.writelines("python runRetrievalSimulationSlurm.py %2d %s %s %s %2.3f %s &\n" %(int(i), instrument,
                                                                                           SZA, useRealGeometry, aod, conf_))
                else:
                    fh.writelines("python runRetrievalSimulationSlurm.py %.4f %s %s %s %s\n" %(aod, instrument, SZA,
                                                                                               useRealGeometry, conf_))
        else:
            if 'discover' in hostname:
                temp_num = 1
                for i in sza:
                    fh.writelines("python runRetrievalSimulationSlurm.py %2d %s %s %s %2.3f %s &\n" %(int(i), instrument,
                                                                                           (i+((arrayNum*1.3)/10+temp_num/100)),
                                                                                             useRealGeometry, aod, conf_))
                    temp_num += 1
            else:
                fh.writelines("python runRetrievalSimulationSlurm.py %.4f %s %s %s %s\n" %(aod, instrument, SZA,
                                                                                           useRealGeometry, conf_))
        fh.writelines("wait\n")
        fh.writelines("echo 0\n")
        fh.writelines("echo End: \n")
        if 'discover' in hostname:
            fh.writelines("rm -rf ${TMPDIR}\n")
        fh.writelines("date")

    fh.close()
    #--------------------------------------------#
    # dry run option
    #--------------------------------------------#
    try:
        if not sys.argv[1] == '--dry-run':
            os.system("sbatch %s" %job_file)
        else:
            print('<><><><> dry run, check the ./job directory for slurm files <><><><>')
    except IndexError:
        os.system("sbatch %s" %job_file)

print('Jobs submitted successfully check the ./job/ folder for output/error')
#----------------------------------end of file--------------------------------#
