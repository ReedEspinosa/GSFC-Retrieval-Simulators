# GSFC-Retrieval-Simulators

A collection of tools for performing aerosol retrieval simulations. The core of the repository is the ``simulation`` class defined in ``simulateRetrieval.py``. This class relies heavily on the class in ``runGRASP.py`` within the *GSFC-GRASP-Python-Interface* repository and those files must be in the path for ``simulation`` to work properly. ``readOSSEnetCDF.py`` also enables a significant chunk of the functionality of this repository but is only applicable to simulations involving the GEOS-5 Nature Run.

The two primary modes this code base is run in are described below. Note that each of the working templates discussed below have simpler, well documented versions in ``GSFC-Retrieval-Simulators/Examples`` – this is the place to start if you are not familiar with the repository.

## Canonical Case Retrieval Simulations ##
In this mode the code will run a retrieval simulation using the A-CCP canonical cases and corresponding architectures defined in the ACCP_ArchitectureAndCanonicalCases directory within this repo.

``runRetrievalSimulation.py`` serves as a working template for running the code in Canonical Case mode


## GEOS-5 Nature Run OSSE Retrieval Simulations ##
In this mode the code runs a retrieval simulation using OSSE results and the osseData class

``runRetrievalOSSE.py`` serves as a working template for running the code in G5NR OSSE mode.


## Common Definitions of key simulation parameters ##

### Instrument configuration ###
All the instrument definitions can be found in the `returnPixel()` function within the file: 
``GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/architectureMap.py``

In addition to viewing angles, wavelengths and types of measurement each instrument architecture (`archName`) has an ``errModel`` function associated with it. These are generally built from special cases of the ``addError()`` function at the bottom of the file but do not have to be. 

``archName=='polar07'`` corresponds to an instrument very similar to Mega-HARP. Adding '00' to the end of most ``archName`` options will produce the same instrument but without any random noise added (all defined within ``addError()`` function.

### Scene properties ###
These are primarily defined in the file: ``GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases/canonicalCaseMap.py``

``setupConCaseYAML()`` is the top level function here. It's purpose is to take in a string that defines the aerosol properties to be used in the forward calculation of the simulated signal. Note that multiple aerosol states can be chained with '+'. So for a case corresponding to smoke over marine aerosol ``caseStrs='smoke+marine'``. (See ``GSFC-Retrieval-Simulators/Examples/runRetrievalSimulation.py`` for more information on its inputs.)

After the case string is parsed by xxx the cases are actually defined in ``conCaseDefinitions()``. The elements of the ``vals`` dict correspond to the analogous quantity in the forward YAML file used to simulate the observations. ``vals['lgrnm']`` is where you would define the size distribution. Jeff Reid's size distributions are not lognormal so we will need to update the code here to work with the 22 bin (or probably more) representation in GRASP.
  

### YAML files ###
There are a series of YAML files in ``GSFC-Retrieval-Simulators/ACCP_ArchitectureAndCanonicalCases`` that all correspond to various forward model simulation ('FWD') configurations and inversion mode ('BCK') configurations. 

**Forward Calculation YAMLs**
``settings_FWD_IQU_POLAR_1lambda.yml`` and ``settings_FWD_POLARandLIDAR_1lambda.yml`` are the YAML files that should be used for polarimeter-only and polarimeter+lidar simulations, respectively. The simulations code heavily modifies these files so it is rare that they need to be edited manually. Although, a new version of each of these will likely need to be created to support the 22-bin-style forward modeling needed for the ACCDAM project.

**Inversion YAMLs**
These are the inversion setting to use when retrieving on the simulated data. These files are much less heavily modified by the simulation code, although it will do smaller things like repeating the last item in ``value`` multiple times so that the number of wavelengths matches the forward file. The current files correspond to:
- ``settings_BCK_POLAR_2modes.yml``: Polarimeter only inversion w/ fine & coarse lognormal modes
- ``settings_BCK_POLAR_3modes.yml``: Polarimeter only inversion w/ fine, sea salt and dust-like lognormal modes
- ``settings_BCK_POLAR_16bins.yml``: Polarimeter only inversion w/ 16 size bins (single RI)
- ``settings_BCK_POLARandLIDAR_10Vbins_2modes.yml``: Polarimeter and lidar inversion with fine and corse modes 
- ``settings_BCK_POLARandLIDAR_10Vbins_4modes.yml``: Polarimeter and lidar inversion with two layers, each with their own fine and corse modes


### Other ACCP_functions ###
A few of the more important functions used in some simulations include:

``normalizeError()`` - Defines the error targets against which we will normalize the simulation derived RMSE in various plotting schemes.

``boundBackYaml()`` – This will adjust the inversion (back) YAML file to correspond to some reasonable assumptions about the scene (e.g., marine aerosol type is unlikely to be found over land or in the free troposphere). 

``selectGeometryEntry()`` - Select one of ~100 observational geometries representative of Sun-sync and GPM-like inclined orbits compiled by Feng Xu.

``readKathysLidarσ()`` – Read uncertainties for each lidar provided during the SIT-A work (typically called by ``addError()`` function).

``readSharonsLidarProfs()`` – Read vertical concentration profiles provided by Sharon Burton during the SIT-A work
