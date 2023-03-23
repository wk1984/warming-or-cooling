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
def absmax(a):
    mi = np.min(a)
    ma = np.max(a)
    if np.abs(mi) > np.abs(ma):
        m = mi
    else:
        m = ma
    return m
dtFmt = mdates.DateFormatter('%b') # define the formatting

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

import colors as clrs

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    #return array[idx]
    return idx

hc_w = 4180 # [J m^-3 K^-1] or 76 in [J mol^-1 K^-1]
# to get from 76 J mol^-1 K^-1: 4.18e6*1.8e-5
# (Volumetric heat capacity of water J m^-3 K^-1 x10^6 times molar density of water (18g/mol)
hc_ice = 2090 #37
hc_s = 620
hc_g = 1010 #28
def get_heat_capacity(sw, si, sg, phi):
    wc = phi * sw # water content = porosity times saturation water
    ic = phi * si
    gc = phi * sg
    HC = wc * hc_w  + ic * hc_ice + gc * hc_g + (1-phi)*hc_s
    return HC

regions=['cd','cw','wd','ww']
r_labels = ["cold-dry", "cold-wet", "warm-dry", "warm-wet"]
ls = ['-', '-', '--', '--']
letters = ['a)', 'b)', 'c)', 'd)']
#regions= ['barrow', 'siberia']
scenarios=['dry_c']
textures = ['02peat']#,'05peat', '1peat']
#al = np.zeros((366,2))
fig, axes = plt.subplots(2,2,figsize=(13,8))
axes = axes.ravel()
cbar_ax = fig.add_axes([0.85, 0.1, 0.025, 0.8])
fig2, axes2 = plt.subplots(2,2, figsize=(7,5))
axes2 = axes2.ravel()
fig3, axes3 = plt.subplots(2,2, figsize=(7,5))
axes3 = axes3.ravel()
fig4, axes4 = plt.subplots(2,2, figsize=(13,8))
axes4 = axes4.ravel()
cbar_ax4 = fig4.add_axes([0.85, 0.1, 0.025, 0.8])
fig5, axes5 = plt.subplots(figsize=(9,3))
fig6, axes6 = plt.subplots(2,2,figsize=(13,8))
axes6 = axes6.ravel()
cbar_ax6 = fig6.add_axes([0.85, 0.1, 0.025, 0.8])

dr = pd.date_range(start='01/01/2017', end='12/31/2021', freq='D')
months = mdates.MonthLocator(interval = 2)

dr2 = pd.date_range(start='01/01/2017', end='01/01/2018', freq='D')
months2 = mdates.MonthLocator(interval = 4)
#al_sand = np.zeros((366,2))
sims = ['dry']
basepath = os.path.abspath(os.path.join(os.getcwd(), os.pardir,os.pardir))

plotpath = basepath+'/plots/' + plot_dir + '/'
# empty temperature arrays
t_top_i = np.zeros((730, len(regions)*len(sims)))
t_top_c = np.zeros((730, len(regions)*len(sims)))
t_sub_i = np.zeros((730, len(regions)*len(sims)))
t_sub_c = np.zeros((730, len(regions)*len(sims)))
# empty thermal conductivity arrays
tc_top_i = np.zeros((730, len(regions)*len(sims)))
tc_top_c = np.zeros((730, len(regions)*len(sims)))
tc_sub_i = np.zeros((730, len(regions)*len(sims)))
tc_sub_c = np.zeros((730, len(regions)*len(sims)))
hc_top_c = np.zeros((730, len(regions)*len(sims)))
hc_top_i = np.zeros((730, len(regions)*len(sims)))
hc_sub_c = np.zeros((730, len(regions)*len(sims)))
hc_sub_i = np.zeros((730, len(regions)*len(sims)))
hc_sub_c = np.zeros((730, len(regions)*len(sims)))
hc_sub_i = np.zeros((730, len(regions)*len(sims)))
# empty saturation arrays
lsat_top_i = np.zeros((730, len(regions)*len(sims)))
lsat_sub_i = np.zeros((730, len(regions)*len(sims)))
counter = 0
s=135
e=280
z_lims=[]
al_depths = np.zeros((len(textures),len(regions)))
for k in range(len(regions)):
    print(regions[k])
    for tex, tt in enumerate(textures):
        al = np.zeros((366,1))
        if tex == 0:
            td_all = np.zeros((366,1))
       # print('texture: ' + tt)
        for j in range(len(scenarios)):
            directory=basepath + '/regions/'+regions[k] + '/' + tt + '/' + scenarios[j]

        #    print(directory)
            vis = ats_xdmf.VisFile(directory, time_unit='d')
            vis.loadMesh(columnar=True)
            dati = np.array([vis.getArray(v) for v in ["temperature"]])
            times = vis.times
            z = vis.centroids[:,2]

            for i,t in enumerate(times[:365]):
                temp = dati[0,i,:]
                td = np.argmax(temp > 273.15)
                #print(td)
                if td == 0:
                    z_td = 0
                else:
                    z_td = z[td]
                al[i,j] = z_td
                if tex == 0:
                    td_all[i,j] = td

        if tex == 0:
            al_min = np.min(al, axis=0)[0]
            z_top = find_nearest(z,al_min*0.25)
            z_sub = find_nearest(z,al_min*0.75)
            print('topsoil depth (dry, '+tt+'): ' + str(al_min*0.25))
            print('subsoil depth (dry, '+tt+'): ' + str(al_min*0.75))


        print('max AL depth (dry, '+tt+'): ' + str(np.min(al, axis=0)[0]))
        al_depths[tex,k] = np.min(al, axis=0)[0]

        al_df = pd.DataFrame(al, columns=scenarios)
        al_df['time'] = dr2


    for ii,a in enumerate(sims):
        directory2 = '/home/ats_1_2/modeling/syn_impale/regions/'+regions[k]+'/02peat/'+a+'_i'
        vis = ats_xdmf.VisFile(directory2, time_unit='d')
        vis.loadMesh(columnar=True)
        dati = np.array([vis.getArray(v) for v in ["temperature","thermal_conductivity",
        "saturation_liquid", "saturation_ice", "saturation_gas", "porosity"]])
        z = vis.centroids[:,2]
        temp_i = dati[0,:365*2,:]-273.15
        tc_i = dati[1,:365*2,:]*1000000
        lsat_i = dati[2,:365*2,:]
        sw_i = dati[2,:365*2,:]
        si_i = dati[3,:365*2,:]
        sg_i = dati[4,:365*2,:]
        phi_i = dati[5,:365*2,:]
        hc_i = get_heat_capacity(sw_i, si_i, sg_i, phi_i)

        directory3 = basepath + '/regions/'+regions[k]+'/02peat/'+a+'_c'
        vis = ats_xdmf.VisFile(directory3, time_unit='d')
        vis.loadMesh(columnar=True)
        z = vis.centroids[:,2]
        dati = np.array([vis.getArray(v) for v in ["temperature","thermal_conductivity",
        "saturation_liquid", "saturation_ice", "saturation_gas", "porosity"]])
        temp_c = dati[0,:365*2,:]-273.15
        tc_c= dati[1,:365*2,:]*1000000
        sw_c = dati[2,:365*2,:]
        si_c = dati[3,:365*2,:]
        sg_c = dati[4,:365*2,:]
        phi_c = dati[5,:365*2,:]
        hc_c = get_heat_capacity(sw_c, si_c, sg_c, phi_c)
        z_lim = int(np.min(td_all[np.nonzero(td_all)])-2)
        z_lims.append(z_lim)

        t_top_c[:,counter] = temp_c[:,z_top]
        t_sub_c[:,counter] = temp_c[:,z_sub]
        t_top_i[:,counter] = temp_i[:,z_top]
        t_sub_i[:,counter] = temp_i[:,z_sub]
        tc_top_c[:,counter] = tc_c[:,z_top]
        tc_sub_c[:,counter] = tc_c[:,z_sub]
        tc_top_i[:,counter] = tc_i[:,z_top]
        tc_sub_i[:,counter] = tc_i[:,z_sub]
        lsat_sub_i[:,counter] = lsat_i[:,z_sub]
        lsat_top_i[:,counter] = lsat_i[:,z_top]
        hc_top_c[:,counter] = hc_c[:,z_top]
        hc_sub_c[:,counter] = hc_c[:,z_sub]
        hc_top_i[:,counter] = hc_i[:,z_top]
        hc_sub_i[:,counter] = hc_i[:,z_sub]

        counter += 1

        timesh, zmesh = np.meshgrid(dr[365+s:365+e], z[z_lim:])


        t_diff = temp_i[:365] - temp_c[:365]
        hc_diff = hc_i[s:e] - hc_c[s:e]
        tc_diff = tc_i[s:e]- tc_c[s:e]
        lsat_diff = sw_i[s:e] - sw_c[s:e]
        print('min tc_diff ' + a + ': ' + str(np.min(tc_diff)))
        print('max tc_diff ' + a + ': ' + str(np.max(tc_diff)))
        midnorm = MidpointNormalize(vmin=-450, vcenter=0, vmax=900)
        intervals = np.linspace(-450, 900, 20)
        im = axes[k].contourf(timesh,zmesh,hc_diff[:,z_lim:].T,levels=intervals,cmap='RdBu_r',norm=midnorm)
        if (k == 0 or k == 2):
            axes[k].set_ylabel('Depth [m]', fontsize=15)
        axes[k].xaxis.set_major_locator(months)
        axes[k].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        cl = axes[k].contour(timesh, zmesh, temp_i[s:e,z_lim:].T, [0], colors='navy', linewidths=1)
        cl2 = axes[k].contour(timesh, zmesh, temp_c[s:e,z_lim:].T, [0], colors='darkorange', linewidths=1)
        axes[k].set_title(letters[k] + ' ' + r_labels[k], fontsize=14,fontweight="bold")
        h1,_ = cl.legend_elements()
        h2,_ = cl2.legend_elements()

        axes[k].tick_params(axis='both', which='major', labelsize=15)
        axes[k].locator_params(axis='y', nbins=4)
        axes[0].legend([h1[0], h2[0]], ['HR case', 'ref. case'], frameon=False,loc='lower left', fontsize=14)

        midnorm = MidpointNormalize(vmin=-0.1, vcenter=0, vmax=0.35)
        intervals = np.linspace(-0.1,0.35, 20)
        im6 = axes6[k].contourf(timesh,zmesh,lsat_diff[:,z_lim:].T,levels=intervals,cmap='RdBu_r',norm=midnorm)
        if (k == 0 or k == 2):
            axes6[k].set_ylabel('Depth [m]', fontsize=15)
        axes6[k].xaxis.set_major_locator(months)
        axes6[k].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        cl = axes6[k].contour(timesh, zmesh, temp_i[s:e,z_lim:].T, [0], colors='navy', linewidths=1)
        cl2 = axes6[k].contour(timesh, zmesh, temp_c[s:e,z_lim:].T, [0], colors='darkorange', linewidths=1)
        axes6[k].set_title(letters[k] + ' ' + r_labels[k], fontsize=14,fontweight="bold")
        h1,_ = cl.legend_elements()
        h2,_ = cl2.legend_elements()

        axes6[k].tick_params(axis='both', which='major', labelsize=15)
        axes6[k].locator_params(axis='y', nbins=4)
        axes6[0].legend([h1[0], h2[0]], ['HR case', 'ref. case'], frameon=False,loc='lower left', fontsize=14)


        im2 = axes2[k].contourf(timesh,zmesh,hc_c[s:e,z_lim:].T,cmap='OrRd',vmin=800,vmax=4000)
        axes2[k].xaxis.set_major_locator(months)
        axes2[k].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        clb = fig2.colorbar(im2, ax = axes2[k])
        clb.ax.set_title('$\Delta$ $\it{C}$')
        cl = axes2[k].contour(timesh, zmesh, temp_i[s:e,z_lim:].T, [0], colors='#56B4E9', linewidths=1)
        cl2 = axes2[k].contour(timesh, zmesh, temp_c[s:e,z_lim:].T, [0], colors='#E69F00', linewidths=1)
        axes2[k].set_title(regions[k])
        h1,_ = cl.legend_elements()
        h2,_ = cl2.legend_elements()
        axes2[0].legend([h1[0], h2[0]], ['HR', 'ref'],frameon=False, loc='best')

        im3 = axes3[k].contourf(timesh,zmesh,hc_i[s:e,z_lim:].T,cmap='OrRd',vmin=800,vmax=4000)
        axes3[k].xaxis.set_major_locator(months)
        axes3[k].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        clb = fig3.colorbar(im3, ax = axes3[k])
        clb.ax.set_title('$\Delta$ $\it{C}$')
        cl = axes3[k].contour(timesh, zmesh, temp_i[s:e,z_lim:].T, [0], colors='#56B4E9', linewidths=1)
        cl2 = axes3[k].contour(timesh, zmesh, temp_c[s:e,z_lim:].T, [0], colors='#E69F00', linewidths=1)
        axes3[k].set_title(regions[k])
        h1,_ = cl.legend_elements()
        h2,_ = cl2.legend_elements()
        axes3[0].legend([h1[0], h2[0]], ['HR', 'ref'],frameon=False, loc='best')


        midnorm = MidpointNormalize(vmin=-0.723, vcenter=0, vmax=1.03)
        intervals = np.linspace(-0.723, 1.03, 20)
        im4 = axes4[k].contourf(timesh,zmesh,tc_diff[:,z_lim:].T,levels=intervals,cmap='RdBu_r',norm=midnorm)
        if (k == 0 or k == 2):
            axes4[k].set_ylabel('Depth [m]', fontsize=15)
        axes4[k].xaxis.set_major_locator(months)
        axes4[k].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        cl = axes4[k].contour(timesh, zmesh, temp_i[s:e,z_lim:].T, [0], colors='navy', linewidths=1)
        cl2 = axes4[k].contour(timesh, zmesh, temp_c[s:e,z_lim:].T, [0], colors='darkorange', linewidths=1)
        axes4[k].set_title(letters[k] + ' ' + r_labels[k], fontsize=14,fontweight="bold")
        h1,_ = cl.legend_elements()
        h2,_ = cl2.legend_elements()
        axes4[k].tick_params(axis='both', which='major', labelsize=15)
        axes4[k].locator_params(axis='y', nbins=4)
        axes4[0].legend([h1[0], h2[0]], ['HR case', 'ref. case'], frameon=False,loc='lower left', fontsize=14)



        cl = axes5.contour(timesh, zmesh, temp_i[s:e,z_lim:].T, [0], colors='navy', linewidths=1)
        cl2 = axes5.contour(timesh, zmesh, temp_c[s:e,z_lim:].T, [0], colors='darkorange', linewidths=1)
        axes5.xaxis.set_major_locator(months)
        axes5.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        axes5.tick_params(axis='both', which='major', labelsize=15)



fig.subplots_adjust(right=0.8)
clb = fig.colorbar(im, cax = cbar_ax)
clb.ax.set_title('$\Delta$ $\it{C}$ $kJ m^{-3} K^{-1}$', fontsize=15)
clb.ax.tick_params(labelsize=15)
fig.savefig(plotpath + 'heatCapacity_plots.png', bbox_inches='tight',dpi=300)
fig.savefig(plotpath + 'heatCapacity_plots.pdf', bbox_inches='tight',dpi=300)
fig2.savefig(plotpath + 'ref_heatCapacity.png', bbox_inches='tight',dpi=300)
fig3.savefig(plotpath + 'HR_heatCapacity.png', bbox_inches='tight',dpi=300)
fig4.subplots_adjust(right=0.8)
clb = fig4.colorbar(im4, cax = cbar_ax4,format='%.2f')
clb.ax.set_title('$\Delta$ $\it{k}$ $W m^{-2}$', fontsize=15)
clb.ax.tick_params(labelsize=15)
fig4.savefig(plotpath + 'thermalconductivity_plots.png', bbox_inches='tight',dpi=300)
fig4.savefig(plotpath + 'thermalconductivity_plots.pdf', bbox_inches='tight',dpi=300)
fig5.tight_layout()
fig5.savefig(plotpath + 'alt_contour.png', bbox_inches='tight',dpi=300)
fig6.subplots_adjust(right=0.8)
clb = fig6.colorbar(im6, cax = cbar_ax6,format='%.2f')
clb.ax.set_title('$\Delta$ liquid sat. [-]', fontsize=15)
clb.ax.tick_params(labelsize=15)
fig6.savefig(plotpath + 'liquidSaturation_plots.png', bbox_inches='tight',dpi=300)

col_names = []
for r in regions:
    for s in sims:
        col_names.append(r + '_' + s)
s=160
e=259
hc_sub_diff =  (hc_sub_i/hc_sub_c-1)*100
hc_top_diff = (hc_top_i/hc_top_c-1)*100

hc_sub_diff_df = pd.DataFrame(hc_sub_diff, columns=col_names)
hc_top_diff_df = pd.DataFrame(hc_top_diff, columns=col_names)

tc_sub_diff =  (tc_sub_i/tc_sub_c-1)*100
tc_top_diff = (tc_top_i/tc_top_c-1)*100

tc_sub_diff_df = pd.DataFrame(tc_sub_diff, columns=col_names)
tc_top_diff_df = pd.DataFrame(tc_top_diff, columns=col_names)
dtFmte = mdates.DateFormatter('%b') # define the formatting

color = ["darkorange", "navy", "darkorange", "navy"]
plt.figure(figsize=(7,5))
for rr,r in enumerate(regions):
    c = color[rr]
    plt.plot(dr[s:e],hc_sub_diff_df[r+'_dry'][s:e],
    label = r_labels[rr], c=c, linestyle=ls[rr])
plt.axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
plt.legend(loc = 'best',facecolor='white',framealpha=0.5)
plt.gca().xaxis.set_major_formatter(dtFmt)
plt.ylabel('${\Delta}$ k (%)')
plt.savefig(plotpath + 'subsoil_HCdiff.png', bbox_inches='tight',dpi=300)

plt.figure(figsize=(7,5))
for rr,r in enumerate(regions):
    c = color[rr]
    plt.plot(dr[s:e],hc_top_diff_df[r+'_dry'][s:e],
    label = r_labels[rr], c=c, linestyle=ls[rr])
plt.axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
plt.legend(loc = 'best',facecolor='white',framealpha=0.5)
plt.gca().xaxis.set_major_formatter(dtFmt)
plt.ylabel('${\Delta}$ k (%)')
plt.savefig(plotpath + 'topsoil_HCdiff.png', bbox_inches='tight',dpi=300)

fig, ax = plt.subplots(figsize=(7,5))
axin = ax.inset_axes([0.05, 0.66, 0.3, 0.25])
for rr,r in enumerate(regions):
    c = color[rr]
    ax.plot(dr[s:e],hc_sub_diff_df[r+'_dry'][s:e],
    label = r_labels[rr], c=c, linestyle=ls[rr])
    axin.plot(dr[s:e],hc_top_diff_df[r+'_dry'][s:e],
         label = r_labels[rr], c=c, linestyle=ls[rr],lw=0.8)
ax.axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
axin.axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
axin.tick_params(axis='both', labelsize=7)
axin.text(0.01, 0.98, 'b) topsoil', transform=axin.transAxes, va='top', ha='left', fontsize=7, fontweight='bold')
ax.text(0.01, 0.99, 'a) subsoil', transform=ax.transAxes, va='top', ha='left', fontsize=15, fontweight='bold')
ax.xaxis.set_major_formatter(dtFmt)
ax.xaxis.set_major_locator(mdates.MonthLocator())
axin.xaxis.set_major_formatter(dtFmt)
axin.xaxis.set_major_locator(mdates.MonthLocator())
ax.set_ylabel('${\Delta}$ k (%)')
ax.legend(loc = 'lower right',facecolor='white',framealpha=0.5,ncol=2, frameon=False)
fig.savefig(plotpath + 'both_HCdiff.png', bbox_inches='tight',dpi=300)

fig, ax = plt.subplots(figsize=(7,5))
axin = ax.inset_axes([0.05, 0.65, 0.3, 0.25])
for rr,r in enumerate(regions):
    c = color[rr]
    ax.plot(dr[s:e],tc_top_diff_df[r+'_dry'][s:e],
    label = r_labels[rr], c=c, linestyle=ls[rr])
    axin.plot(dr[s:e],tc_sub_diff_df[r+'_dry'][s:e],
         label = r_labels[rr], c=c, linestyle=ls[rr],lw=0.8)
ax.axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
axin.axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
axin.tick_params(axis='both', labelsize=7)
axin.text(0.01, 0.98, 'b) subsoil', transform=axin.transAxes, va='top', ha='left', fontsize=7, fontweight='bold')
ax.text(0.01, 0.99, 'a) topsoil', transform=ax.transAxes, va='top', ha='left', fontsize=15, fontweight='bold')
ax.xaxis.set_major_formatter(dtFmt)
ax.xaxis.set_major_locator(mdates.MonthLocator())
axin.xaxis.set_major_formatter(dtFmt)
axin.xaxis.set_major_locator(mdates.MonthLocator())
ax.set_ylabel('${\Delta}$ TC (%)')
ax.legend(loc = 'upper right',facecolor='white',framealpha=0.5,ncol=1,frameon=False)#, bbox_to_anchor=(0.03, 0.15))
fig.savefig(plotpath + 'both_TCdiff.png', bbox_inches='tight',dpi=300)
