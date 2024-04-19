#!/bin/env python
"""
Program to manage and run ATS demo problems.

With inspiration, and in some places just stolen, from Ben Andre's
PFloTran regression test suite.

Author: Ethan Coon (ecoon@lanl.gov)
"""
from __future__ import print_function

import sys
import os
import time
import argparse
import textwrap

if not os.path.join(os.environ['ATS_SRC_DIR'], 'tools', 'utils') in sys.path:
    sys.path.append(os.path.join(os.environ['ATS_SRC_DIR'], 'tools', 'utils'))

import ats_xdmf
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib import ticker

sns.set(style='ticks', font_scale=1.25)
# plt.rcParams['font.sans-serif'] = ['Heiti TC']
# plt.rcParams['axes.unicode_minus'] = False

# %config InlineBackend.figure_format = 'retina'

os.path.join(os.environ['ATS_SRC_DIR'], 'tools', 'utils')

try:
    import test_manager
except ImportError:
    sys.path.append(os.path.join(
        os.environ['ATS_SRC_DIR'], 'tools', 'testing'))
    import test_manager


class my_utils:

    def get_variable_field(folder, variable= 'temperature', 
                         date_onset = '2011-01-01', 
                         length=365, 
                         time_unit = 'd'):
        
        # import ats_xdmf
        vis = ats_xdmf.VisFile(folder, time_unit=time_unit)
        vis.loadMesh(columnar=True)
        va = vis.getArray(variable)[1:length+1, ]
        t0 = pd.date_range(date_onset, periods=va.shape[0], freq='1'+time_unit)
        da = vis.centroids[:, 2] # depth

        return t0, da, va

    def get_single_layer(folder, variable, 
                         domain='snow', 
                         date_onset = '2011-01-01', 
                         length=365, 
                         time_unit = 'd'):
        
        # import ats_xdmf
        vis = ats_xdmf.VisFile(folder, domain, time_unit=time_unit)
        va = vis.getArray(variable)[1:length+1]
        t0 = pd.date_range(date_onset, periods=va.shape[0], freq='1'+time_unit)
        va = np.asarray(va)

        if va.shape[0]*va.shape[1] == len(va): va = np.reshape(va, len(va))

        return t0, va

    def quickplot_ts(date_index, data, ylabel='Rainfall (mm)', xdateticks=True, figsize=[8, 4], **kwargs):

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.plot(date_index, data, **kwargs)

        if xdateticks:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        if (data.min() < 0) & (data.max() > 0):
            ax.plot([date_index[0], date_index[-1]],
                    [0, 0], color='gray', linestyle=':')

        ax.set_xlim(date_index[0], date_index[-1])

        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)

        return fig, ax

    def add_new_path(path='..'):

        if path in sys.path:
            pass
        else:
            sys.path.append(path)
            print(path + ' has been added')

        # os.path.join(os.environ['ATS_SRC_DIR'],'tools', 'utils')

    def mk_new_folder(foldername):

        if os.path.exists(foldername):

            pass

        else:

            os.mkdir(foldername)

            print("Creating "+foldername)

    def get_surface_moisture(directory, onset_date='2011-01-01', length=365):

        vis = ats_xdmf.VisFile(directory,  'surface', time_unit='d')
        vis.loadMesh(columnar=True)

        sm = vis.getArray('water_content')[1:]

        t0 = pd.date_range(onset_date, periods=sm.shape[0], freq='1D')

        return t0[0:length], sm[0:length]/55500*1000

    def get_snow_depth(directory, onset_date='2011-01-01', length=365):

        vis = ats_xdmf.VisFile(directory,  'snow', time_unit='d')

        vis.loadMesh(columnar=True)

        snod = vis.getArray('snow-depth')[1:]

        t0 = pd.date_range(onset_date, periods=snod.shape[0], freq='1D')

        return t0[0:length], snod[0:length]

    def get_liquid_saturation(directory, onset_date='2011-01-01', length=365):

        vis = ats_xdmf.VisFile(directory, time_unit='d')

        vis.loadMesh(columnar=True)

        sat = vis.getArray('saturation_liquid')
        sat = sat[1:,]

        t0 = pd.date_range(onset_date, periods=sat.shape[0], freq='1D')

        aa = vis.centroids[:, 2]

        # print(aa.shape[0], tsoil.shape)

        return t0[0:length], aa, sat[0:length, :]

    def get_soil_temperatures(directory, onset_date='2011-01-01', length=365):

        vis = ats_xdmf.VisFile(directory, time_unit='d')

        vis.loadMesh(columnar=True)

        tsoil = vis.getArray('temperature') - 273.15
        tsoil = tsoil[1:,]

        t0 = pd.date_range(onset_date, periods=tsoil.shape[0], freq='1D')

        aa = vis.centroids[:, 2]

        # print(aa.shape[0], tsoil.shape)

        return t0[0:length], aa, tsoil[0:length, :]

    def get_soil_moistures(directory, onset_date='2011-01-01', length=365):

        vis = ats_xdmf.VisFile(directory, time_unit='d')

        vis.loadMesh(columnar=True)

        cv = vis.getArray('cell_volume')[0]
        wc = vis.getArray('water_content')
        rho_l = vis.getArray('molar_density_liquid')
        theta_t = wc/rho_l/cv
        theta_t = theta_t[1:,]

        t0 = pd.date_range(onset_date, periods=theta_t.shape[0], freq='1D')

        aa = vis.centroids[:, 2]

        # print(aa.shape[0], tsoil.shape)

        return t0[0:length], aa, theta_t[0:length, :]

    def plot_snow_soil(times, depth, snowdepth, tsoil, figsize=[4, 5], ylim_top=[0, 0.45], ylim_bot=[-0.75, 0]):

        fig = plt.figure(tight_layout=True, figsize=figsize)
        gs = gridspec.GridSpec(3, 1, hspace=0.0)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(times, snowdepth, 'k', linewidth = 2)
        ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
        ax1.set_ylim(ylim_top)
        ax1.set_ylabel('Snow Depth (m)')
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2 = fig.add_subplot(gs[1:, 0], sharex=ax1)
        t00 = ax2.pcolor(times, depth, tsoil.T,
                         cmap='seismic', vmin=-15, vmax=15)
        ax2.contour(times, depth, tsoil.T, [0], colors=['k'], linewidths = [2])
        ax2.set_ylim(ylim_bot)
        ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        fig.colorbar(t00, ax=ax2, extend='both', orientation='horizontal',
                     aspect=50, pad=0.18, label='Soil temperatures (C)')
        ax2.set_ylabel('Depth (m)')

        # return fig