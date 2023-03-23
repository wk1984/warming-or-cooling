import sys,os
sys.path.append(os.path.join(os.environ['ATS_SRC_DIR'],'tools', 'utils'))
import ats_xdmf
import plot_column_data
import matplotlib.cm
import parse_xmf
import column_data
import numpy as np
import matplotlib.cm
from matplotlib import pyplot as plt
from matplotlib import colors
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.pyplot import cm
import datetime as dt
from matplotlib.lines import Line2D
import locale
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str,
                    help="subdirectory in the plot directory")
parser.add_argument('-t', '--texture', type=str,
			help='thickness of organic layer')
args = parser.parse_args()

plot_dir = str(args.directory)

locale.setlocale(locale.LC_ALL,'en_US.utf8')

def absmax(a):
    mi = np.min(a)
    ma = np.max(a)
    if np.abs(mi) > np.abs(ma):
        m = mi
    else:
        m = ma
    return m

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

import colors as clrs

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
regions=['cd','cw','wd','ww']
r_labels = ["cold-dry", "cold-wet", "warm-dry", "warm-wet"]

scenarios=['dry_c']
tex = str(args.texture)
dr = pd.date_range(start='01/01/2017', end='12/31/2021', freq='D')
months = mdates.MonthLocator(interval = 4)

dr2 = pd.date_range(start='01/01/2017', end='01/01/2018', freq='D')
months2 = mdates.MonthLocator(interval = 4)
basepath = os.path.abspath(os.path.join(os.getcwd(), os.pardir,os.pardir))

plotpath = basepath+'/plots/' + plot_dir + '/'
dtFmt2 = mdates.DateFormatter('%b')

for k in range(len(regions)):
    plt.figure(figsize=(7,5))
    zaa_c = np.zeros((138,4))
    zaa_i = np.zeros((138,4))
    directory2 = basepath + '/regions/'+regions[k]+'/'+tex+'/dry_i'
    vis = ats_xdmf.VisFile(directory2, time_unit='d')
    vis.loadMesh(columnar=True)
    dati = np.array([vis.getArray(v) for v in ["temperature"]])
    z = vis.centroids[:,2]
    temp_i = dati[0,:365*2,:]-273.15


    directory3 = basepath + '/regions/'+regions[k]+'/'+tex+'/dry_c'
    vis = ats_xdmf.VisFile(directory3, time_unit='d')
    vis.loadMesh(columnar=True)
    z = vis.centroids[:,2]
    dati = np.array([vis.getArray(v) for v in ["temperature"]])
    temp_c = dati[0,:365*2,:]-273.15


colors = ['darkorange','navy', 'darkorange', 'navy']
ls = ['-', '-', '--', '--']
alt = np.zeros((2,len(regions)))
scenarios = ['dry_c', 'dry_i']
fig,ax = plt.subplots(figsize=(10,3))
for k in range(len(regions)):
    print(regions[k])
    al = np.zeros((366,2))
    td_all = np.zeros((366,2))
    for j in range(len(scenarios)):
        directory=basepath + '/regions/'+regions[k] + '/'+tex+'/' + scenarios[j]

        vis = ats_xdmf.VisFile(directory, time_unit='d')
        vis.loadMesh(columnar=True)
        dati = np.array([vis.getArray(v) for v in ["temperature"]])
        times = vis.times
        z = vis.centroids[:,2]

        for i,t in enumerate(times[:365]):
            temp = dati[0,i,:]
            td = np.argmax(temp > 273.15)
            if td == 0:
                z_td = 0
            else:
                z_td = z[td]
            al[i,j] = z_td
    if tex == '02peat':
        ax.plot(dr[135:280],al[135:280,0], color = colors[k], linestyle=ls[k], label=r_labels[k])
    else:
        ax.plot(dr[135:290],al[135:290,0], color = colors[k], linestyle=ls[k], label=r_labels[k])
    ax.fill_between(dr[135:280],al[135:280,0],al[135:280,1], color= colors[k], alpha=0.4, edgecolor="none")




    print("Max ALT in REF is: " + str(np.min(al,axis=0)[0]))
    print("Max ALT in HR is: " + str(np.min(al,axis=0)[1]))
    print("REF thaw begins: " + str(dr2[al[:,0]<0][0].strftime("%B %d")))
    print("REF freezeup begins: " + str(dr2[al[:,0]<0][-1].strftime("%B %d")))
    print("HR thaw begins: " + str(dr2[al[:,1]<0][0].strftime("%B %d")))
    print("HR freezeup begins: " + str(dr2[al[:,1]<0][-1].strftime("%B %d")))
    ref_alt = np.min(al,axis=0)[0]
    hr_alt = np.min(al,axis=0)[1]
    alt[0,k] = ref_alt
    alt[1,k] = hr_alt
ax.axvline(dt.datetime(2017, 6, 15), color = 'gainsboro', alpha = 0.5,lw=3)
ax.axvline(dt.datetime(2017, 7, 15), color = 'gainsboro', alpha = 0.5,lw=3)
ax.axvline(dt.datetime(2017, 8, 15), color = 'gainsboro', alpha = 0.5,lw=3)
ax.text(0.01, 0.99, 'c) thaw depth', transform=ax.transAxes, va='top', ha='left', fontsize=15, fontweight='bold')
ax.set_ylabel('thawdepth [m]')
if tex == '02peat':
    ax.set_ylim(-0.8,0.1)
else:
    ax.set_ylim(-1.2,0.15)
ax.xaxis.set_major_formatter(dtFmt2)
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig(plotpath + 'thawdepth.svg', bbox_inches='tight', dpi=300)
fig.savefig(plotpath + 'thawdepth.pdf', bbox_inches='tight', dpi=300)
fig.savefig(plotpath + 'thawdepth_transparent.svg', bbox_inches='tight', dpi=300, transparent=True)
