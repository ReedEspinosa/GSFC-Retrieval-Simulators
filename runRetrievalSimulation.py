#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulation using the A-CCP canonical cases and corresponding architectures defined in the ACCP_ArchitectureAndCanonicalCases directory within this repo """

import os
import sys
import itertools
MADCAPparentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GRASP_scripts is in parent of MADCAP_scripts
sys.path.append(os.path.join(MADCAPparentDir, "GRASP_scripts"))
import simulateRetrieval as rs
from miscFunctions import checkDiscover
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ACCP_ArchitectureAndCanonicalCases'))
from architectureMap import returnPixel
from canonicalCaseMap import setupConCaseYAML

#n = int(sys.argv[1]) # (0,1,2,...,N-1)
x = int(sys.argv[1]) # (0,1,2,...,N-1)
missingCases=[28,677,678,679,681,682,684,685,687,689,690,692,693,699,700,702,703,705,758,760,761,762,783,843,847,851,857,858,861,862,863,868,869,872,874,875,924,927,932,934,939,940,941,944,946,949,954,955,1008,1011,1016,1018,1019,1020,1022,1024,1028,1037,1039,1040,1041,1093,1096,1099,1102,1103,1104,1105,1106,1108,1110,1115,1119,1121,1122,1176,1178,1179,1181,1182,1183,1184,1186,1187,1188,1198,1200,1201,1202,1203,1206,1208,1209,1262,1264,1267,1270,1275,1277,1279,1281,1282,1286,1294,1295,1362,1375,1391,1392,1393,1394,1396,1401,1403,1405,1406,1408,1410,1411,1412,1414,1415,1417,1418,1421,1422,1423,1424,1426,1436,1455,1458,1459,1473,1475,1477,1479,1483,1488,1489,1491,1492,1493,1495,1497,1500,1501,1504,1506,1509,1514,1547,1561,1562,1563,1565,1566,1572,1574,1577,1581,1582,1583,1584,1585,1587,1588,1590,1594,1603,1607,1612,1616,1625,1633,1640,1646,1647,1648,1650,1653,1656,1657,1658,1659,1662,1667,1669,1673,1679,1682,1697,1708,1711,1716,1723,1724,1731,1733,1734,1737,1738,1739,1740,1743,1744,1745,1746,1748,1749,1750,1755,1759,1803,1815,1817,1820,1821,1823,1824,1825,1826,1829,1830,1831,1833,1836,1838,1840,1842,1846,1890,1896,1901,1902,1904,1905,1907,1911,1912,1916,1917,1920,1921,1922,1925,1928,1929,1981,1983,1994,1995,1998,1999,2005,2011,2012,2014,2015]
n=missingCases[x]

if checkDiscover(): # DISCOVER
    basePath = os.environ['NOBACKUP']
    saveStart = os.path.join(basePath, 'synced/Working/SIM15_pre613SeminarApr2020/COMBO03_2mode_n%d_' % n)
    ymlDir = os.path.join(basePath, 'MADCAP_scripts/ACCP_ArchitectureAndCanonicalCases/')
    dirGRASP = os.path.join(basePath, 'grasp_open/build/bin/grasp')
    krnlPath = os.path.join(basePath, 'local/share/grasp/kernels')
    Nsims = 40
    maxCPU = 14
else: # MacBook Air
    saveStart = '/Users/wrespino/Desktop/testLIDAR_' # end will be appended
    ymlDir = '/Users/wrespino/Synced/Local_Code_MacBook/MADCAP_Analysis/ACCP_ArchitectureAndCanonicalCases/'
    dirGRASP = '/usr/local/bin/grasp'
    krnlPath = None
    Nsims = 2
    maxCPU = 1
fwdModelYAMLpathLID = os.path.join(ymlDir, 'settings_FWD_POLARandLIDAR_1lambda.yml')
bckYAMLpathLID = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_10Vbins_2modes.yml')
# bckYAMLpathLID = os.path.join(ymlDir, 'settings_BCK_POLARandLIDAR_10Vbins_2modes%d.yml' % x)
fwdModelYAMLpathPOL = os.path.join(ymlDir, 'settings_FWD_IQU_3lambda_POL.yml')
bckYAMLpathPOL = os.path.join(ymlDir, 'settings_BCK_IQU_3lambda_POL.yml')


conCases = ['variableFineLofted+variableCoarse',
            'variableFine+variableCoarseLofted',
            'variableFineLofted+variableCoarseNonsph',
            'variableFine+variableCoarseLoftedNonsph',
            'variableFineLoftedNonsph+variableCoarse',
            'variableFineNonsph+variableCoarseLofted',
#            'variableFineLoftedNonsph',
#            'variableFineNonsph',
#            'variableCoarseLoftedNonsph',
#            'variableCoarseNonsph',
            'variableFineLoftedChl+variableCoarseChl',
            'variableFineChl+variableCoarseLoftedChl',
            ] #12 - 4 = 8
SZAs = [0.1, 5, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60] # 12 (GRASP doesn't seem to be wild about θs=0)
Phis = [0] # 1 
#τFactor = [0.04, 0.08, 0.10, 0.12, 0.14, 0.18, 0.35] #7 
τFactor = [0.02 , 0.04 , 0.05 , 0.06 , 0.07 , 0.09 , 0.175] #7 (cut in half because we generally are using two modes)
instruments = ['misr', 'modisMisr', 'modisMisrPolar'] #3 N=2016
rndIntialGuess = True # randomly vary the intial guess of retrieved parameters

paramTple = list(itertools.product(*[instruments,conCases,SZAs,Phis,τFactor]))[n] 
savePath = saveStart + '%s_case-%s_sza%d_phi%d_tFct%4.2f_V1.pkl' % paramTple
# savePath = saveStart + 'TEST_%s_V2_YAML%d.pkl' % (instruments[n], x)
print('-- Processing ' + os.path.basename(savePath) + ' --')

# RUN SIMULATION
if 'lidar' in paramTple[0].lower():
    fwdModelYAMLpath = fwdModelYAMLpathLID
    bckYAMLpath = bckYAMLpathLID
else:
    fwdModelYAMLpath = fwdModelYAMLpathPOL
    bckYAMLpath = bckYAMLpathPOL
    
nowPix = returnPixel(paramTple[0], sza=paramTple[2], relPhi=paramTple[3], nowPix=None)
cstmFwdYAML, landPrct = setupConCaseYAML(paramTple[1], nowPix, fwdModelYAMLpath, caseLoadFctr=paramTple[4])
nowPix.land_prct = landPrct
print('n= %d, Nλ = %d' % (n,nowPix.nwl))
simA = rs.simulation(nowPix) # defines new instance for architecture described by nowPix
simA.runSim(cstmFwdYAML, bckYAMLpath, Nsims, maxCPU=maxCPU, savePath=savePath, binPathGRASP=dirGRASP, intrnlFileGRASP=krnlPath, releaseYAML=True, lightSave=True, rndIntialGuess=rndIntialGuess)
