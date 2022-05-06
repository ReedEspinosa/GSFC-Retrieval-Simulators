import numpy as np
from matplotlib import pyplot as plt
from simulateRetrieval import simulation
from mpl_toolkits.basemap import Basemap

# mergePATH = '/discover/nobackup/wrespino/OSSE_results_working/MERGED_ss450-g5nr.leV210.GRASP.example.polarimeter07.200608ALL_1000z.pkl'
# mergePATH = '/discover/nobackup/wrespino/OSSE_results_working/MERGED_ss450-g5nr.leV210.GRASP.example.polarimeter07.200608ALL_ALLz.pkl'
# simBase = simulation(picklePath=mergePATH)
# print('Loading from %s - %d' % (mergePATH, len(simBase.rsltBck)))
simBase.classifyAerosolType(verbose=True)

for waveInd in range(8):
    waveIndAOD = 3
    aodThresh = 0.0
    logY = False
    relativeErr = False

    costThresh = np.percentile([rb['costVal'] for rb in simBase.rsltBck], 95)
    keepInd = [rb['costVal']<costThresh for rb in simBase.rsltBck]

    trueAOD = np.array([rb['aod'][waveIndAOD] for rb in simBase.rsltBck])[keepInd]
    wavelng = simBase.rsltFwd[0]['lambda'][waveInd]
    # AOD
    true = np.array([rb['aod'][waveInd] for rb in simBase.rsltBck])[keepInd]
    rtrv = np.array([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
    label = 'AOD (λ=%4.2fμm)' % simBase.rsltFwd[0]['lambda'][waveInd]
    relativeErr = False
    # 1-SSA
#     true = 1-np.array([rb['ssa'][waveInd] for rb in simBase.rsltBck])[keepInd]
#     rtrv = 1-np.array([rf['ssa'][waveInd] for rf in simBase.rsltFwd])[keepInd]
#     label = 'Coalbedo (λ=%4.2fμm)' % simBase.rsltFwd[0]['lambda'][waveInd]
#     aodThresh = 0.3
    # AAOD
    # true = np.asarray([(1-rf['ssa'][waveInd])*rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
    # rtrv = np.asarray([(1-rb['ssa'][waveInd])*rb['aod'][waveInd] for rb in simBase.rsltBck])[keepInd]
    # label = 'AAOD (λ=%4.2fμm)' % simBase.rsltFwd[0]['lambda'][waveInd]
    # ANGSTROM
#     label = 'AE (%4.2f/%4.2f μm)' % (wavelng, simBase.rsltFwd[0]['lambda'][waveInd2])
#     aod1 = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltFwd])[keepInd]
#     aod2 = np.asarray([rf['aod'][waveInd2] for rf in simBase.rsltFwd])[keepInd]
#     logLamdRatio = np.log(simBase.rsltFwd[0]['lambda'][waveInd]/simBase.rsltFwd[0]['lambda'][waveInd2])
#     true = -np.log(aod1/aod2)/logLamdRatio
#     aod1 = np.asarray([rf['aod'][waveInd] for rf in simBase.rsltBck])[keepInd]
#     aod2 = np.asarray([rf['aod'][waveInd2] for rf in simBase.rsltBck])[keepInd]
#     rtrv = -np.log(aod1/aod2)/logLamdRatio
#     maxVar = 2.25
#     aodThresh = 0.05 # does not apply to AOD plot
    # g
    # label = 'Asym. Param. (λ=%4.2fμm)' % wavelng
    # true = np.asarray([(rf['g'][waveInd]) for rf in simBase.rsltFwd])[keepInd]
    # rtrv = np.asarray([(rb['g'][waveInd]) for rb in simBase.rsltBck])[keepInd]
    # maxVar = None
    # aodThresh = 0.15 # does not apply to AOD plot
    # vol FMF
    # label = 'Submicron Volum. Frac.'
    # def fmfCalc(r,dvdlnr):
    #     cutRadius = 0.5
    #     fInd = r<=cutRadius
    #     logr = np.log(r)
    #     return np.trapz(dvdlnr[fInd],logr[fInd])/np.trapz(dvdlnr,logr)
    # true = np.asarray([fmfCalc(rf['r'], rf['dVdlnr']) for rf in simBase.rsltFwd])[keepInd]
    # rtrv = np.asarray([fmfCalc(rb['r'][0,:], rb['dVdlnr'].sum(axis=0)) for rb in simBase.rsltBck])[keepInd]
    # maxVar = None
    # aodThresh = 0.1 # does not apply to AOD plot
    # SPH
    # label = 'Volum. Frac. Spherical'
    # true = np.asarray([rf['sph'] for rf in simBase.rsltFwd])[keepInd]
    # rtrv = np.asarray([aodWght(rf['sph'], rf['vol']) for rf in simBase.rsltBck])[keepInd]
    # maxVar = None
    # aodThresh = 0.1 # does not apply to AOD plot
    # k
#     label = 'AOD Wghtd. IRI (λ=%4.2fμm)' % wavelng
#     true = np.asarray([rf['k'][waveInd] for rf in simBase.rsltFwd])[keepInd]
#     rtrv = np.asarray([aodWght(rf['k'][:,waveInd], rf['aodMode'][:,waveInd]) for rf in simBase.rsltBck])[keepInd]
#     maxVar = None
#     aodThresh = 0.2 # does not apply to AOD plot
    # n
#     label = 'RRI-1.33 (λ=%4.2fμm)' % wavelng
#     true = np.asarray([rf['n'][waveInd] for rf in simBase.rsltFwd])[keepInd]-1.33
#     rtrv = np.asarray([aodWght(rf['n'][:,waveInd], rf['aodMode'][:,waveInd]) for rf in simBase.rsltBck])[keepInd]-1.33
#     maxVar = None
#     aodThresh = 0.1 # does not apply to AOD plot
    # reff
    # label = 'Fine Mode Effective Radius'
    # # simBase._addReffMode(0.008, True) # reframe so pretty much all of the PSD is in the second "coarse" mode
    # true = np.asarray([rf['rEffMode'][1] for rf in simBase.rsltFwd])[keepInd]
    # rtrv = np.asarray([rf['rEffMode'][1] for rf in simBase.rsltBck])[keepInd]
    # maxVar = 4.0
    # aodThresh = 0.25 # does not apply to AOD plot

    errFun = lambda t,r : np.sqrt(np.mean((t-r)**2))
    aeroType = np.array([rf['aeroType'] for rf in simBase.rsltFwd])[keepInd]
    version = 'V4'

    labels = [
        'Dust   ',
        'PolDust',
        'Marine',
        'Urb/Ind',
        'BB-White',
        'BB-Dark',
        'All Types']

    colors = np.array(
          [[0.86459569, 0.0514    , 0.05659828, 1.        ],
           [1.        , 0.35213633, 0.02035453, 1.        ],
           [0.13613483, 0.39992585, 0.65427592, 1.        ],
           [0.850949428, 0.50214365, 0.65874433, 1.        ],
           [0.30,      0.72, 0.30, 1.        ],
           [0.35    , 0.42 , 0.2, 1.        ],
           [0.3    , 0.3 , 0.3, 1.        ]]) # all
       
    dxPerType = 6

    fig, ax1 = plt.subplots(figsize=(7, 2.6))
    Ntype = colors.shape[0]
    Npix = []
    for i in np.r_[0:(Ntype*dxPerType):dxPerType]:
        if i/dxPerType==(Ntype-1):
            indKeep = trueAOD > aodThresh
            NpixNow = indKeep.sum()
        else:
            indKeep = np.logical_and(aeroType==round(i/dxPerType), trueAOD > aodThresh)
            NpixNow = indKeep.sum()

        muTr = np.mean(true[indKeep])
        muRt = np.mean(rtrv[indKeep])
        sigTr = np.std(true[indKeep])
        rmseRt = errFun(rtrv[indKeep], true[indKeep])
        if relativeErr:
            sigTr = sigTr/muTr
            rmseRt = rmseRt/muTr
    
        typeColor = colors[int(round(i/dxPerType))]
        typeColor = (typeColor + typeColor.mean())/2
        typeColor[-1] = 1
        typeColor = typeColor**2
        whtExp = 0.29 if i==0 else 0.7
        ax1.bar(i,   muTr, color=typeColor, edgecolor='white', alpha=0.5)
        ax1.bar(i+1, muRt, color=typeColor, edgecolor='white')
        ax1.bar(i+2, sigTr, color=typeColor, edgecolor='white', hatch='\\', alpha=0.5)
        ax1.bar(i+3, rmseRt, color=typeColor, edgecolor='white', hatch='\\')

        print('Type %2d – N=%7d' % (round(i/dxPerType), NpixNow))
        Npix.append(NpixNow)

    x0 = (colors.shape[0]-1)*dxPerType-1.4
    yMax = ax1.get_ylim()[1]
    ax1.plot([x0,x0], [0, yMax], '--', color='gray')
    ax1.set_ylim([0, yMax])

    ax1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    
    labels = [label+'\n(%d)  ' % n for label,n in zip(labels, Npix)]
    ax1.set_xticks(np.r_[1.5:(Ntype*dxPerType+2):dxPerType])
    ax1.set_xticklabels(labels, ha='center')
    ax1.set_ylabel(label)
    ax1.set_xlim(-1.5,Ntype*dxPerType-1.8)
    plt.tight_layout()

    if logY:
        ax1.set_ylim([0.01, 2])
        ax1.set_yscale('log')
        version = 'log_' + version 

    if relativeErr:
        version = 'relErr_' + version 
    
    fn = 'errorBarPlots_AODgt%05.3f_%s_%4.2fum_%s.pdf' % (aodThresh, label[0:-11], wavelng, version)
    fig.savefig('/discover/nobackup/wrespino/synced/Working/AGU2021_Plots/%s' % fn)
    print('Plot saved as %s' % fn)
plt.ion()
plt.show()
