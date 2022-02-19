import numpy as np
import sys
import matplotlib.pyplot as plt
import datetime as dt
sys.path.append("/Users/wrespino/Synced/Local_Code_MacBook/GSFC-GRASP-Python-Interface")
import runGRASP as rg
import miscFunctions as mf
from MADCAP_functions import loadVARSnetCDF

YAMLpath = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/newOpticsTables/LUT-DUST/settings_BCK_ExtSca_9lambda.yml'
netCDFpath = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/newOpticsTables/LUT-DUST/optics_DU.v15_6.nc'
savePath = '/Users/wrespino/Synced/Remote_Sensing_Projects/MADCAP_CAPER/newOpticsTables/LUT-DUST/testCase_scatExtFit_BOUNDED_80nmTo20um_RUN2TEST.pkl'
maxCPU = 3
unbound = False # if FALSE: each bin has its own YAML (YAMLpath[:-4:]+'_mode%d' % (bn+1)+'.yml')

wvInds = [2,4,5,6,7,8,9,11,13] #2->354nm, 5->500nm, 7->870nm, 9->1250nm, 13->2500nm
#wvls = [0.354, 0.55, 0.87, 1.23, 1.65] # Nλ=5 # WE NEED TO GET THESE FROM ABOVE
# DUMMY MEASUREMENTS (determined by architecture, should ultimatly move to seperate scripts)
#  For more than one measurement type or viewing geometry pass msTyp, nbvm, thtv, phi and msrments as vectors: \n\
#  len(msrments)=len(thtv)=len(phi)=sum(nbvm); len(msTyp)=len(nbvm) \n\
#  msrments=[meas[msTyp[0],thtv[0],phi[0]], meas[msTyp[0],thtv[1],phi[1]],...,meas[msTyp[0],thtv[nbvm[0]],phi[nbvm[0]]],meas[msTyp[1],thtv[nbvm[0]+1],phi[nbvm[0]+1]],...]'
msTyp = [12] # 12 -> ext, 13 -> abs
nbvm = np.ones(len(msTyp))
sza = 0
thtv = np.zeros(len(msTyp))
phi = np.zeros(len(msTyp)) # currently we assume all observations fall within a plane

varNames = ['qext', 'qsca', 'lambda', 'refimag', 'refreal', 'rLow', 'rUp', 'g', 'area', 'volume', 'rEff']
optTbl = loadVARSnetCDF(netCDFpath, varNames)


#print([-x for x in optTbl['refimag'][4,0,wvInds]])
#print([x for x in optTbl['refreal'][4,0,wvInds]])
#sys.exit()
wvls = optTbl['lambda'][wvInds]*1e6
gspRun = rg.graspRun(YAMLpath) if unbound else []
gspRun = []
for bn in range(5):
    binYAML = YAMLpath if unbound else YAMLpath[:-4:]+'_mode%d' % (bn+1)+'.yml'
    gspRunNow = rg.graspRun(binYAML)   
    dtObj = dt.datetime.now + dt.timedelta(hours=bn)
    nowPix = rg.pixel(dtObj, 1, 1, 0, 0, 0, 100)
    for wvl, wvInd in zip(wvls, wvInds): # This will be expanded for wavelength dependent measurement types/geometry
#        meas = np.r_[optTbl['qext'][bn,0,wvInd], optTbl['qsca'][bn,0,wvInd]]
        meas = np.r_[optTbl['qext'][bn,0,wvInd]]*3/4/optTbl['rEff'][bn,0]/1e6 # should give extinction coef. at volume concentration of unity/1000 (kind of... see note by avRatio defintion below)
        nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv, phi, meas)
    if unbound:
        gspRun.addPix(nowPix)
    else:
        gspRunNow.addPix(nowPix)
        gspRun.append(gspRunNow)
gDB = rg.graspDB(gspRun, maxCPU)
rslt = gDB.processData(savePath=savePath)


plt.figure()
for bn in range(5):
    dvdr, radii = mf.logNormal(rslt[bn]['rv'][0], rslt[bn]['sigma'][0])
    dvdlnr = dvdr*radii
    plt.plot(radii, dvdlnr/dvdlnr.max())
plt.ylabel('dV/dr')
plt.legend(['Mode %d' % int(x+1) for x in range(5)])
plt.plot(np.tile(np.atleast_2d(optTbl['rLow']).T*1e6,2).T, [0, 1], '--k')
plt.gca().set_prop_cycle(None)
plt.plot(np.tile(np.atleast_2d(optTbl['rUp']).T*1e6,2).T, [0, 1], '--k')
plt.xlim([0.08, 20.0])
plt.xlabel('radius (μm)')
plt.xscale('log')
plt.tight_layout()

pltMdInd = np.r_[0:5]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3.5))
#avRatio = np.r_[optTbl['area'][pltMdInd,0]]/np.r_[optTbl['volume'][pltMdInd,0]]/1e6
avRatio = 3/4/optTbl['rEff'][pltMdInd,0]/1e6 # Above is exactly correct & this agrees, except in first bin. There Pete did mass weighted average of several sub-bins to find rEff but since we want to match past results we want to be consistent with Pete's technically incorrect method.
ax[0].plot(optTbl['lambda']*1e6, avRatio*optTbl['qext'][pltMdInd,0,:].T)
ax[1].plot(optTbl['lambda']*1e6, avRatio*optTbl['qsca'][pltMdInd,0,:].T)
ax[0].set_prop_cycle(None)
ax[1].set_prop_cycle(None)
ax[0].plot(rslt[0]['lambda'], np.array([x['aod'] for x in rslt[pltMdInd]]).T, 'x')
ax[1].plot(rslt[1]['lambda'], np.array([x['ssa']*x['aod'] for x in rslt[pltMdInd]]).T, 'x')
for i in range(2):
    ax[i].set_xlim([0.3, 3.0])
    ax[i].set_xlabel('wavelength')   
ax[0].set_ylabel('$q_{ext}$')
ax[1].set_ylabel('$q_{sca}$')
ax[0].legend(['Mode %d' % int(x+1) for x in pltMdInd])
plt.tight_layout()

sig = [r['sigma'][0] for r in rslt]
rv = [r['rv'][0] for r in rslt]
for r,s in zip(rv, sig):
    print('[%6.4f, %6.4f]' % (r,s))
    
    
    