import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mplb
from simulateRetrieval import simulation
from mpl_toolkits.basemap import Basemap

# mergePATH = '/discover/nobackup/wrespino/OSSE_results_working/MERGED_ss450-g5nr.leV210.GRASP.example.polarimeter07.200608ALL_1000z.pkl'
mergePATH = '/discover/nobackup/wrespino/OSSE_results_working/MERGED_ss450-g5nr.leV210.GRASP.example.polarimeter07.200608ALL_ALLz.pkl'
simBase = simulation(picklePath=mergePATH)
print('Loading from %s - %d' % (mergePATH, len(simBase.rsltBck)))

plt.figure(figsize=(12, 6.5))
m = Basemap(projection='merc', resolution='c', llcrnrlon=-180, llcrnrlat=-58, urcrnrlon=180, urcrnrlat=75)
m.bluemarble(scale=1);
#m.shadedrelief(scale=0.2)
#m.fillcontinents(color=np.r_[176,204,180]/255,lake_color='white')
lon = [rb['longitude'] for rb in simBase.rsltBck]
lat = [rb['latitude'] for rb in simBase.rsltBck]
# vals = [rb['sph'] for rb in simBase.rsltFwd]
# label = 'SPH'
# vals = np.log10([rb['costVal'] for rb in simBase.rsltBck])
# label = 'log10(Cost Value)'
# vals = [rb['vol'][1]/rb['vol'].sum() for rb in simBase.rsltBck]
# label = 'SS Fraction'
# valsT = np.asarray([rf['aod'][3] for rf in simBase.rsltFwd])
# EE = 0.02+0.05*valsT
# valsR = np.asarray([rb['aod'][3] for rb in simBase.rsltBck])
# vals = np.abs((valsR-valsT)/EE)
# vals[vals>3] = 3 # NOTE: top of color bar is really ≥3
# label = 'AOD Error |(AOD_R-AOD_T)/(0.05 + 10%xAOD_T)|'
# vals = np.log10([rb['vol'][2] for rb in simBase.rsltBck])
# label = 'log10(Dust Vol)'
#vals = np.sqrt([rb['rEffMode'][1]/rf['rEffMode'][1]-1 for rb,rf in zip(simBase.rsltBck, simBase.rsltFwd)])**0.6
# vals = np.array([rb['rEffMode'][1]-rf['rEffMode'][1] for rb,rf in zip(simBase.rsltBck, simBase.rsltFwd)])
# label = 'Reff Error'
# vals = np.array([np.log(rf['aod'][2]/rf['aod'][5])/np.log(rf['lambda'][5]/rf['lambda'][2]) for rf in simBase.rsltFwd])
# label = 'True EAE'
vals = [rf['aod'][3] for rf in simBase.rsltFwd]
label = 'G5NR 0.55μm AOD'
# vals = [rf['n'][4] for rf in simBase.rsltFwd]
# label = 'True RRI'

x, y = m(lon, lat)
plt.scatter(x, y, c=vals, s=3, cmap='YlOrRd', alpha=0.05, norm=mplb.colors.LogNorm()) # 'YlOrRd', 'seismic'
cbar = plt.colorbar()
cbar.set_label(label, fontsize=14)
cbar.solids.set(alpha=1)
plt.tight_layout()

figSaveName = mergePATH.replace('.pkl', '_V2_MAP.png')
print('Saving map to: %s' % figSaveName)
plt.savefig(figSaveName)
plt.ion()
plt.show()

