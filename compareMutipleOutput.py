#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This script will run a retrieval simulations on user defined scenes and instruments 

Created by: Anin Puthukkudy (Research Scientist, ESI, UMBC)
Created on: 01/22/2024
"""

#==============================================================================
# Import necessary modules
#==============================================================================
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.figsize': [8, 6]})
plt.rcParams.update({'figure.dpi': 160})
plt.rcParams.update({'font.family': 'cmr10'})
import numpy as np
import sys

#==============================================================================
# Path append
#==============================================================================
# Append the path to the GRASP Python interface
RtrvSimParentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # we assume GSFC-GRASP-Python-Interface is in parent of GSFC-Retrieval-Simulators
sys.path.append(os.path.join(RtrvSimParentDir, "GSFC-GRASP-Python-Interface"))

#==============================================================================
# Dependencies
#==============================================================================
# Import the GRASP Python interface to read the output file
from runGRASP import graspRun
gr = graspRun()

#==============================================================================
# Define functions
#==============================================================================

def plot_grasp_runs(run_list, output_list, wavInd=2, runVar=['RRI', 1.42],
                    err=0.03, plot_type='I', ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,6))

    for run, output_file in zip(run_list, output_list):
        # Read the data from the output file
        data = read_grasp_output(output_file)[0]

        # Plot the data
        x_ = data['sca_ang'][:,wavInd]
        if plot_type == 'I':
            y_ = data['fit_I'][:,wavInd]
        elif plot_type == 'DoLP':
            y_ = np.sqrt(data['fit_Q'][:,wavInd]**2 + data['fit_U'][:,wavInd]**2)/data['fit_I'][:,wavInd]
        ax.plot(x_, y_, '.-', label=f"{runVar[0]} = {runVar[1] + run}", lw = 0.5)

        # if abs(run) < 0.001: # if run is zero, plot the original data plus or minus 3% noise shaded region
        if abs(run) < 0.001: # if run is zero, plot the original data plus or minus 3% noise shaded region
            if plot_type == 'I': ax.fill_between(x_, y_*(1-err), y_*(1+err), alpha=0.3)
            elif plot_type == 'DoLP': ax.fill_between(x_, (y_-err), (y_+err), alpha=0.3)

    # Add labels and legend
    ax.set_ylabel(plot_type)
    if not plot_type == 'I':
        ax.set_title(f"AOD = {data['aod'][wavInd]}, SSA = {data['ssa'][wavInd]}, IRI = {data['k'][0,wavInd]}")
        ax.legend()
        ax.set_xlabel("Scattering angle")
    ax.grid()

def read_grasp_output(output_file):
    # Code to read the GRASP output file and extract the data
    # Replace this with your own implementation

    rslt = gr.readOutput(output_file)

    return rslt

#==============================================================================
# Main script
#==============================================================================
if __name__ == "__main__":
    # Define the list of runs and output file locations
    runVar = ['RRI', 1.42]  # RRI = 1.42 is the default value
    run_list = [-0.02, -0.01, 0.0, 0.01, 0.02]
    wavInd = 4
    parentDirectory = '/tmp/tmp_z7lh6rb'
    output_list = ["bench_FWD_IQU_rslts_mp2.txt", "bench_FWD_IQU_rslts_mp1.txt",
                    "bench_FWD_IQU_rslts.txt", "bench_FWD_IQU_rslts_pp1.txt",
                    "bench_FWD_IQU_rslts_pp2.txt"]

    # Plot the GRASP runs
    fig_, ax_ = plt.subplots(ncols=1, nrows=2, figsize=(8,6), sharex=True)
    plot_grasp_runs(run_list, [os.path.join(parentDirectory, x) for x in output_list],
                     ax=ax_[0], wavInd=wavInd,)
    plot_grasp_runs(run_list, [os.path.join(parentDirectory, x) for x in output_list],
                     ax=ax_[1], wavInd=wavInd,
                    err= 0.005, plot_type='DoLP')
    
    # Show the plot
    plt.suptitle(f"I, DoLP simulation for {runVar[0]} = {runVar[1]} +/- {run_list[2]} \n shaded region is +/- 3% noise in I and 0.005% DoLP")
    plt.show()