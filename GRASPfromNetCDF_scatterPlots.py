#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib as mpl
import os
os.environ["PROJ_LIB"] = "/Users/wrespino/anaconda3/share/proj" # fix for "KeyError: 'PROJ_LIB'" bug
from mpl_toolkits.basemap import Basemap

maxAOD = 2
logLogAOD = True

#i = 3
#wvl = 0.67
i = 1
wvl = wvls[i]

# we should plot DoLP error VS aodSum/albSum
# may want to change AOD sum to use trueData[0]['tau'] 
#aodSum = np.array([np.sum(rslt['aod'][:] if (rslt['aod'][:]<2).all() else 0) for rslt in rslts])/6
#albSum = np.array([np.sum(rslt['brdf'][0,4]) for rslt in rslts])/6
aodSum = trueData[0]['tau']
albSum = np.mean(measData[i]['surf_reflectance'][0:120,:],axis=1)
albSum[albSum > np.percentile(albSum, 95)] = np.percentile(albSum, 95)
albSum[albSum < np.percentile(albSum, 5)] = np.percentile(albSum, 5)
#DoLPres = np.array([np.sum(rslt['meas_PoI'][:,i]-rslt['fit_PoI'][:,i])/np.sum(rslt['meas_PoI'][:,i]+rslt['fit_PoI'][:,i]) for rslt in rslts])
#DoLPres = np.array([np.mean(np.abs(rslt['meas_PoI'][:,i]*rslt['meas_I'][:,i]-rslt['fit_PoI'][:,i]*rslt['fit_I'][:,i])) for rslt in rslts])
DoLPres = np.array([np.mean(np.abs(rslt['meas_PoI'][:,i]-rslt['fit_PoI'][:,i])) for rslt in rslts])
Ires = 2*np.array([np.sum(np.abs(rslt['meas_I'][:,i]-rslt['fit_I'][:,i]))/np.sum(rslt['meas_I'][:,i]+rslt['fit_I'][:,i]) for rslt in rslts])
#DoLPres = np.array([np.mean(np.abs(rslt['meas_PoI'][:,i])) for rslt in rslts])

phi = np.reshape(measData[i]['solar_azimuth'], (-1,1)) - measData[i]['sensor_azimuth'][:,0]
#phi = np.array([rslt['sca_ang'][:,0].max() for rslt in rslts])
#plt.figure()
#plt.scatter(aodSum, albSum, c=DoLPres)
#plt.yscale('log')
#plt.xscale('log')
#plt.xlabel("AOD (532nm)")
#plt.ylabel("Mean Surface Reflectance (532nm)")
#cbar = plt.colorbar()
#cbar.set_label('%f μm DoLP Residual (%%)' % wvl)

#plt.figure(); dolpAx = plt.gca()
plt.sca(dolpAx)
values, base = np.histogram(DoLPres, bins=40)
cumulative = np.cumsum(values)
plt.plot(base[:-1], 100*cumulative/DoLPres.shape[0])
plt.xlabel('DoLP Absolute Residual')
plt.ylabel('Fraction Below (%)')
plt.plot([0.005, 0.005], [0, 100], '--', Color=[0.5,0.5,0.5])
plt.ylim([-1,101])

#plt.figure(); IAx = plt.gca()
plt.sca(IAx)
values, base = np.histogram(Ires, bins=40)
cumulative = np.cumsum(values)
plt.plot(base[:-1], 100*cumulative/Ires.shape[0])
plt.xlabel('Intensity Relative Residual')
plt.ylabel('Fraction Below (%)')
plt.ylim([-1,101])


FS=16
plt.rcParams.update({'font.size': FS})
grnAOD = np.array([rslt['aod'][i] for rslt in rslts])
lidAOD = trueData[0]['tau']
grnAlb = np.array([rslt['brdf'][0,i] for rslt in rslts])
plt.figure()
#plt.scatter(lidAOD, grnAOD, c=grnAlb)
plt.plot(np.r_[0, maxAOD], np.r_[0, maxAOD], 'k')
plt.scatter(lidAOD, grnAOD, c='r')
plt.xlabel("AOD Simulated (0.532 μm)", fontsize=FS)
plt.ylabel("AOD GRASP (%.3f μm)"  % wvl, fontsize=FS)
plt.xlim([0.005, maxAOD])
plt.ylim([0.005, maxAOD])
Rcoef = np.corrcoef(lidAOD, grnAOD)[0,1]
RMSE = np.sqrt(np.mean((lidAOD - grnAOD)**2))
bias = np.mean(np.abs(lidAOD - grnAOD))
textstr = 'N=%d\nR=%.3f\nRMS=%.3f\nbias=%.3f\n'%(len(lidAOD), Rcoef, RMSE, bias)
if logLogAOD:
    plt.text(0.13*maxAOD, 0.004, textstr, fontsize=FS)
    plt.yscale('log')
    plt.xscale('log')
else:
    plt.text(0.7*maxAOD, 0.03, textstr, fontsize=FS)
    
#plt.title('Sept 1, No BPDF')
plt.tight_layout()

if 'rEff' in rslts[0]: # GRASP doesn't report reff with log normals
    maxReff = 3
    plt.figure()
    reffGRASP = [rslt['rEff'] for rslt in rslts]
    plt.plot(np.r_[0, maxReff], np.r_[0, maxReff], 'k')
    plt.scatter(trueData[0]['reff']*1e6, reffGRASP, c='r')
    plt.xlabel('Reff Simulated (μm)')
    plt.ylabel('Reff GRASP (μm)')
    plt.xlim([0,maxReff])
    plt.ylim([0,maxReff])
    plt.tight_layout()


plt.figure(figsize=(12, 6)) 
m = Basemap(projection='robin', resolution='l', lat_0=0, lon_0=0)
#m.bluemarble(scale=1);
m.shadedrelief(scale=0.2)
lon = measData[0]['trjLon']
lat = measData[0]['trjLat']
x, y = m(lon, lat)
plt.scatter(x, y, c=lidAOD, s=8, cmap='jet')
#plt.scatter(x, y, c=100*DoLPres, s=8, cmap='jet')
#plt.scatter(x, y, c=100*Ires, s=8, cmap='jet')
#plt.title('DoLP Residaul (%4.2f μm), No BPDF, Fitting I and DoLP' % wvl)
#cbar = plt.colorbar()
#cbar.set_label("(OSSE-GRASP)/(OSSE+GRASP)")
#plt.scatter(x, y, c=2*(grnAOD-lidAOD)/(grnAOD+lidAOD), s=8, cmap='seismic')
#plt.scatter(x, y, c=lidAOD, s=4, cmap='plasma')
#plt.title('OSSE AOD, No BPDF, Fitting I and DoLP')
#plt.title('AOD residual, No BPDF, Fitting I and DoLP')
cbar = plt.colorbar()
plt.clim(-1.5,1.5)
#cbar.set_label("$τ_{OSSE}$", FontSize=16)
cbar.set_label("$2(τ_{GRASP} - τ_{OSSE})/(τ_{GRASP} + τ_{OSSE})$", FontSize=18)
#cbar.set_clim([-1.5,1.5])

sys.exit()

#plt.figure()
##plt.scatter(albSum, DoLPres, c=aodSum)
##plt.scatter(aodSum, DoLPres, c=phi[0:120, -1])
##plt.scatter(phi[0:120, -1], DoLPres, c=aodSum)
##plt.scatter(measData[i]['solar_azimuth']-measData[i]['sensor_azimuth'][0:120,0], DoLPres, c=aodSum)
#plt.scatter(measData[i]['sensor_azimuth'][0:120,0], DoLPres, c=aodSum)
##plt.scatter(measData[i]['solar_zenith'], DoLPres, c=aodSum)
##plt.yscale('log')
##plt.xscale('log')
##plt.ylim([-0.45, 0.65])
##plt.xlim([0.04, 0.5])
##plt.xlabel("OSSE Mean Surface Reflectance (%.3f um)" % wvl)
#plt.xlabel("Relative Azimuth")
#plt.ylabel('DoLP Fractional Residual (%.3f μm)' % wvl)
#plt.title('OSSE - GRASP')
#cbar = plt.colorbar()
#cbar.set_label("OSSE AOD (532nm)")
##cbar.set_label("Relative Azimuth")



# look at dark atmosphere pixels
surfDmInd = trueData[1]['tau'] < 0.015
#surfDmInd = trueData[1]['tau'] == trueData[1]['tau'].min()
spctDOLP = np.array([np.mean(100*MD['DOLP'][surfDmInd,:], axis=0) for MD in measData])
#spctDOLP = np.array([np.mean(100*MD['toa_reflectance'][surfDmInd,:]*MD['DOLP'][surfDmInd,:], axis=0) for MD in measData])
plt.figure()
[plt.plot(wvls, DOLP,  c=plt.cm.jet(20*i)) for i,DOLP in enumerate(spctDOLP.T)]
plt.legend(['θ=%d' % val for val in measData[0]['sensor_zenith']], ncol=3)
#plt.plot(wvls, (0.27*np.array(wvls)**-4)+0., c='k')
#plt.ylim([0,10])


#rWvInd = 1;
##for i in range(np.sum(surfDmInd)):
#for i in range(1):
#    ind = np.nonzero(surfDmInd)[0][i]
#    phiN = phi[ind, :]
#    fig = plt.figure()
#    ax1 = fig.add_subplot(111, projection='polar')
#    c = ax1.scatter(phiN*np.pi/180, np.abs(measData[0]['sensor_zenith']), c=100*measData[rWvInd]['toa_reflectance'][ind,:]*measData[rWvInd]['DOLP'][ind,:])
##    c = ax1.scatter(phiN*np.pi/180, np.abs(measData[0]['sensor_zenith']), c=100*measData[rWvInd]['DOLP'][ind,:])
#    ax1.scatter(0, np.abs(measData[0]['solar_zenith'][ind]), s=100, facecolors='none', edgecolors='r')
#    vmin,vmax = c.get_clim() #-- obtaining the colormap limits
#    cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax) #-- Defining a normalised scale
#    ax3 = fig.add_axes([0.8, 0.1, 0.03, 0.8]) #-- Creating a new axes at the right side
#    cb1 = mpl.colorbar.ColorbarBase(ax3, norm=cNorm) #-- Plotting the colormap in the created axes
#    fig.subplots_adjust(left=0.05,right=0.85)
#    plt.ylabel("100 x Polarized Reflectance (%5.3f um)" % wvls[rWvInd])
#    plt.title('T = %d sec' % measData[0]['time'][ind])
    