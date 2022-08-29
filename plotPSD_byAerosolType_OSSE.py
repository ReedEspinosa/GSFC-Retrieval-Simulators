import numpy as np
from matplotlib import pyplot as plt
from simulateRetrieval import simulation


# mergePATH = '/Users/wrespino/Synced/Working/OSSE_Test_Run/MERGED_ss450-g5nr.leV202A.GRASP.example.polarimeter07.20060802_ALLz.pkl'
mergePATH = '/Users/wrespino/Synced/Working/OSSE_Test_Run/MERGED_ss450-g5nr.leV210.GRASP.example.polarimeter07.200608ALL_ALLz.pkl'
surf2plot = 'ocean'
aodThresh = 0.01 # applies to lambda[2] (410nm is OSSE)
version = 'FULL_V3' # for FN tag

# simBase = simulation(picklePath=mergePATH)
# print('Loading from %s - %d' % (mergePATH, len(simBase.rsltBck)))
# simBase.classifyAerosolType(verbose=True)

if 'land_prct' not in simBase.rsltFwd[0] or surf2plot=='both':
    keepInd = np.ones(len(simBase.rsltFwd))
    if not surf2plot=='ocean': print('NO land_prct KEY! Including all surface types...')
else:
    lp = np.array([rf['land_prct'] for rf in simBase.rsltFwd])
    keepInd = lp>99 if surf2plot=='land' else lp<1 if surf2plot=='ocean' else lp>-1

# apply convergence filter
# simBase.conerganceFilter(forceχ2Calc=True) # ours looks more normal, but GRASP's produces slightly lower RMSE
# costThresh = np.percentile([rb['costVal'] for rb in simBase.rsltBck[keepInd]], 95)
# keepInd = np.logical_and(keepInd, [rb['costVal']<costThresh for rb in simBase.rsltBck])

# apply minimum AOD filter (AOD_550nm>0.2 for G5NR lambda)
keepInd = np.logical_and(keepInd, [rf['aod'][2]>aodThresh for rf in simBase.rsltFwd])
keepInd = np.logical_and(keepInd, [rf['aod'][2]<2.0 for rf in simBase.rsltFwd])


aeroType = np.array([rf['aeroType'] for rf in simBase.rsltFwd])

labels = [
    'Dust   ',
    'PolDust',
    'Marine',
    'Urb/Ind',
    'BB-White',
    'BB-Dark']

colors = np.array(
      [[0.86459569, 0.0514    , 0.05659828, 1.        ],
       [1.        , 0.35213633, 0.02035453, 1.        ],
       [0.13613483, 0.39992585, 0.65427592, 1.        ],
       [0.850949428, 0.50214365, 0.65874433, 1.        ],
       [0.30,      0.72, 0.30, 1.        ],
       [0.35    , 0.42 , 0.2, 1.        ]])
       

keepTypes = [0,2,3,5]

fig, ax1 = plt.subplots(figsize=(5, 3))
Ntype = colors.shape[0]
radii = simBase.rsltFwd[0]['r'][0]
for t_i in keepTypes:
    indKeep = np.logical_and(aeroType==t_i, keepInd)
    NpixNow = indKeep.sum()
    typeColor = colors[t_i]
#     typeColor = (typeColor + typeColor.mean())/2
    typeColor[-1] = 1
    typeColor = typeColor**1
    if NpixNow>0:
#         dvdr = np.mean([rf['dVdlnr'][0] for rf in simBase.rsltFwd[indKeep]], axis=0)
        dvdr = np.mean([rf['dVdlnr'][0]/radii for rf in simBase.rsltFwd[indKeep]], axis=0)
        ax1.plot(radii, dvdr, color=typeColor) 
    print('Type: %s – N=%7d' % (labels[t_i], NpixNow))

ax1.set_xlim([0.00098,10.1])
ax1.set_xscale('log')
ax1.legend(np.asarray(labels)[keepTypes], loc='upper left')
ax1.set_ylabel('dv/dr  ($μm^{3} \cdot μm^{-2} \cdot μm^{-1}$)')
ax1.set_xlabel('Radius  ($μm$)')
plt.tight_layout()

fn = 'PSD_Plots_AerosolType_gt%05.3f_%s.pdf' % (aodThresh, version)
fig.savefig('/Users/wrespino/Synced/Presentations/AGU_Fall_2022/Figures/%s' % fn)
print('Plot saved as %s' % fn)
plt.ion()
plt.show()
