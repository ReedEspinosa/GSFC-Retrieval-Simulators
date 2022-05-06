from simulateRetrieval import simulation
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap

# savePATH = '/discover/nobackup/wrespino/OSSE_results_working/MERGED_ss450-g5nr.leV210.GRASP.example.polarimeter07.200608ALL_ALLz.pkl'
# simBase = simulation(picklePath=savePATH)
# simBase.classifyAerosolType(verbose=True)

fig = plt.figure(figsize=(10, 5.5))
m = Basemap(projection='merc', resolution='c', lon_0=0, llcrnrlon=-180, llcrnrlat=-40, urcrnrlon=180, urcrnrlat=80)
# m.bluemarble(scale=1, alpha=0.5);
# m.shadedrelief(scale=0.2)
m.fillcontinents(color=[0.85,0.85,0.85],lake_color='white')
m.drawcoastlines()
# m.drawlsmask(land_color='none', ocean_color=[0.01,0.01,0.2], zorder=2)
parallels = np.arange(-40.,81,20.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[True,False,False,True], fontsize=14)
meridians = np.arange(0.,351.,60.)
m.drawmeridians(meridians,labels=[True,False,False,True], fontsize=14)

lon = [rb['longitude'] for rb in simBase.rsltBck]
lat = [rb['latitude'] for rb in simBase.rsltBck]
vals = [rf['aeroType'] for rf in simBase.rsltFwd]
label = 'Aerosol Type'
x, y = m(lon, lat)

cmapOrg = plt.get_cmap('Set1', 9)
cmap = plt.get_cmap('Set1', 6)
cmap.colors = cmapOrg.colors[[0,4,1,7,8,2],:]

cmap.colors[1,1] = cmap.colors[1,1] - 0.05
cmap.colors[1,2] = cmap.colors[1,2] + 0.05
cmap.colors[1,3] = 0.7
cmap.colors[4,0] = cmap.colors[4,0] - 0.0005
cmap.colors[4,1] = cmap.colors[4,1] + 0.18
cmap.colors[4,2] = cmap.colors[4,2] - 0.0005
cmap.colors[5,0] = 0.35
cmap.colors[5,1] = 0.42
cmap.colors[5,2] = 0.2
cmap.colors = cmap.colors**1.3
cmap.colors[3] = np.sqrt(cmap.colors[3])

plt.scatter(x, y, c=vals, s=35, cmap=cmap, vmin=-0.5, vmax=5.5, marker=".", alpha=0.05, zorder=4) # 'YlOrRd', 'seismic'
cbar = plt.colorbar(ticks=np.arange(0, 6), fraction=0.023, pad=0.02)
# cbar.set_label(label, fontsize=14)
cbar.ax.tick_params(size=0)
cbar.solids.set(alpha=1)
plt.tight_layout()


# plt.figure(figsize=(12, 6.5))
# m = Basemap(projection='merc', resolution='c', llcrnrlon=-180, llcrnrlat=-58, urcrnrlon=180, urcrnrlat=75)
# m.bluemarble(scale=1);
# vals = [rf['ssa'][2] for rf in simBase.rsltFwd]
# label = 'True SSA'
# x, y = m(lon, lat)
# plt.scatter(x, y, c=vals, s=3, cmap='YlOrRd', alpha=0.05) # 'YlOrRd', 'seismic'
# cbar = plt.colorbar()
# cbar.set_label(label, fontsize=14)
# cbar.solids.set(alpha=1)
# plt.tight_layout()
# 
# plt.figure(figsize=(12, 6.5))
# m = Basemap(projection='merc', resolution='c', llcrnrlon=-180, llcrnrlat=-58, urcrnrlon=180, urcrnrlat=75)
# m.bluemarble(scale=1);
# vals = np.array([np.log(rf['aod'][2]/rf['aod'][5])/np.log(rf['lambda'][5]/rf['lambda'][2]) for rf in simBase.rsltFwd])
# label = 'True EAE'
# x, y = m(lon, lat)
# plt.scatter(x, y, c=vals, s=3, cmap='YlOrRd', alpha=0.05) # 'YlOrRd', 'seismic'
# cbar = plt.colorbar()
# cbar.set_label(label, fontsize=14)
# cbar.solids.set(alpha=1)
# plt.tight_layout()
# 
# plt.figure(figsize=(12, 6.5))
# m = Basemap(projection='merc', resolution='c', llcrnrlon=-180, llcrnrlat=-58, urcrnrlon=180, urcrnrlat=75)
# m.bluemarble(scale=1);
# vals = [rf['n'][4] for rf in simBase.rsltFwd]
# label = 'True RRI'
# x, y = m(lon, lat)
# plt.scatter(x, y, c=vals, s=3, cmap='YlOrRd', alpha=0.05) # 'YlOrRd', 'seismic'
# cbar = plt.colorbar()
# cbar.set_label(label, fontsize=14)
# cbar.solids.set(alpha=1)
# plt.tight_layout()
# 

fn = 'Aerosol_type_map_globe_V1.pdf'
fig.savefig('/discover/nobackup/wrespino/synced/Working/AGU2021_Plots/%s' % fn)
print('Plot saved as %s' % fn)

plt.ion()
plt.show()