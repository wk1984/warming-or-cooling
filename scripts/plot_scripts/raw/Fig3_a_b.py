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
textures = [str(args.texture)]

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

    def __call__(self, value, clip=None):.
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
basepath = os.path.abspath(os.path.join(os.getcwd(), os.pardir,os.pardir))

plotpath = basepath+'/plots/' + plot_dir + '/'
#al = np.zeros((366,2))
fig, axes = plt.subplots(len(regions), len(textures)+1,figsize=((len(textures)+1)*4.5, len(regions)*4))
dr = pd.date_range(start='01/01/2017', end='12/31/2021', freq='D')
months = mdates.MonthLocator(interval = 4)

dr2 = pd.date_range(start='01/01/2017', end='01/01/2018', freq='D')
months2 = mdates.MonthLocator(interval = 4)
#al_sand = np.zeros((366,2))
sims = ['dry']

# empty temperature arrays
t_top_i = np.zeros((730, len(regions)*len(sims)))
t_top_c = np.zeros((730, len(regions)*len(sims)))
t_sub_i = np.zeros((730, len(regions)*len(sims)))
t_sub_c = np.zeros((730, len(regions)*len(sims)))
t_pf_i = np.zeros((730, len(regions)*len(sims)))
t_pf_c = np.zeros((730, len(regions)*len(sims)))

# empty thermal conductivity arrays
tc_top_i = np.zeros((730, len(regions)*len(sims)))
tc_top_c = np.zeros((730, len(regions)*len(sims)))
tc_sub_i = np.zeros((730, len(regions)*len(sims)))
tc_sub_c = np.zeros((730, len(regions)*len(sims)))
# empty saturation arrays
lsat_top_i = np.zeros((730, len(regions)*len(sims)))
lsat_sub_i = np.zeros((730, len(regions)*len(sims)))
counter = 0
z_lims=[]
al_depths = np.zeros((len(textures),len(regions)))
for k in range(len(regions)):
    print(regions[k])
    for tex, tt in enumerate(textures):
        al = np.zeros((366,1))
        if tex == 0:
            td_all = np.zeros((366,1))
        for j in range(len(scenarios)):
            directory=basepath + '/regions/'+regions[k] + '/' + tt + '/' + scenarios[j]

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
                if tex == 0:
                    td_all[i,j] = td

        if tex == 0:
            al_min = np.min(al, axis=0)[0]
            z_top = find_nearest(z,al_min*0.25)
            z_sub = find_nearest(z,al_min*0.75)
            print('topsoil depth (dry, '+tt+'): ' + str(al_min*0.25))
            print('subsoil depth (dry, '+tt+'): ' + str(al_min*0.75))


        print('max AL depth (dry, '+tt+'): ' + str(np.min(al, axis=0)[0]))
        #print('max AL depth (sat, '+tt+'): ' + str(np.min(al, axis=0)[1]))
        al_depths[tex,k] = np.min(al, axis=0)[0]

        al_df = pd.DataFrame(al, columns=scenarios)
        al_df['time'] = dr2

        axes[k,tex].plot(al_df.time,al_df.dry_c, color='orangered')
        #axes[k,tex].plot(al_df.time,al_df.sat_c, color='blue')
        axes[0,tex].set_title(tt, fontsize = 20)
        axes[0,tex].legend(labels=['thaw depth'], frameon=False, fontsize =14)
        axes[k,tex].xaxis.set_major_locator(months2)
        axes[k,tex].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        axes[k,tex].set_xlim([dt.date(2017, 1, 1), dt.date(2017, 12, 31)])
        axes[k,0].text(-0.45, 0.5, regions[k], weight='bold',
                         va='center', ha='left', transform=axes[k,0].transAxes,fontsize=18, rotation=90)
        axes[k,0].set_ylabel('depth [m]', fontsize=13)
        axes[k,tex].tick_params(axis='both', which='major', labelsize=17)


    for ii,a in enumerate(sims):
        directory2 = basepath + '/regions/'+regions[k]+'/02peat/'+a+'_i'
        vis = ats_xdmf.VisFile(directory2, time_unit='d')
        vis.loadMesh(columnar=True)
        dati = np.array([vis.getArray(v) for v in ["temperature","thermal_conductivity","saturation_liquid"]])
        z = vis.centroids[:,2]
        temp_i = dati[0,:365*2,:]-273.15
        tc_i = dati[1,:365*2,:]*1000000
        lsat_i = dati[2,:365*2,:]
        #t_top_i[:,counter] = temp_i[:,80]
        #t_sub_i[:,counter] = temp_i[:,70]


        directory3 = basepath + '/regions/'+regions[k]+'/02peat/'+a+'_c'
        vis = ats_xdmf.VisFile(directory3, time_unit='d')
        vis.loadMesh(columnar=True)
        z = vis.centroids[:,2]
        dati = np.array([vis.getArray(v) for v in ["temperature","thermal_conductivity"]])
        temp_c = dati[0,:365*2,:]-273.15
        tc_c= dati[1,:365*2,:]*1000000
        #print(np.min(td_all[np.nonzero(td_all)]))
        z_lim = int(np.min(td_all[np.nonzero(td_all)])-2)
        z_lims.append(z_lim)

        t_top_c[:,counter] = temp_c[:,z_top]
        t_sub_c[:,counter] = temp_c[:,z_sub]
        t_top_i[:,counter] = temp_i[:,z_top]
        t_sub_i[:,counter] = temp_i[:,z_sub]
        t_pf_i[:,counter] = temp_i[:,87]
        t_pf_c[:,counter] = temp_c[:,87]
        tc_top_c[:,counter] = tc_c[:,z_top]
        tc_sub_c[:,counter] = tc_c[:,z_sub]
        tc_top_i[:,counter] = tc_i[:,z_top]
        tc_sub_i[:,counter] = tc_i[:,z_sub]
        lsat_sub_i[:,counter] = lsat_i[:,z_sub]
        lsat_top_i[:,counter] = lsat_i[:,z_top]
        t_sub_diff = temp_i[:,z_sub]-temp_c[:,z_sub]
        t_top_diff = temp_i[:,z_top]-temp_c[:,z_top]

        counter += 1

        timesh, zmesh = np.meshgrid(dr[365:365*3], z[z_lim:])


        t_diff = temp_i - temp_c
        midnorm = MidpointNormalize(vmin=np.min(t_diff), vcenter=0, vmax=np.max(t_diff))
        intervals = np.linspace(np.min(t_diff), np.max(t_diff), 12)
        im = axes[k,ii+1].contourf(timesh,zmesh,t_diff[:,z_lim:].T,levels=intervals,cmap='RdBu_r',norm=midnorm)
        axes[k,ii+1].xaxis.set_major_locator(months)
        axes[k,ii+1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        clb = fig.colorbar(im, ax = axes[k,ii+1])
        clb.ax.set_title('$\Delta T$')
        cl = axes[k,ii+1].contour(timesh, zmesh, temp_i[:,z_lim:].T, [0], colors='#56B4E9', linewidths=1)
        cl2 = axes[k,ii+1].contour(timesh, zmesh, temp_c[:,z_lim:].T, [0], colors='#E69F00', linewidths=1)
        axes[k,ii+1].set_title(a)
        h1,_ = cl.legend_elements()
        h2,_ = cl2.legend_elements()
        axes[0,ii+1].legend([h1[0], h2[0]], ['HR', 'ref'],frameon=False, loc=3)
        #axes[k,ii+1].legend(frameon=False)
        axes[k,ii+1].tick_params(axis='both', which='major', labelsize=15)
        print('absolute max. temp difference SUBSOIL '+ regions[k] + ': ' + str(round(max(t_sub_diff[160:259], key=abs),3)))
        print('absolute temperature in HR is: ' + str(round(float(temp_i[160:259,z_sub][t_sub_diff[160:259] == max(t_sub_diff[160:259],key=abs)]),2)))
        print('absolute temperature in ref. is: ' + str(round(float(temp_c[160:259,z_sub][t_sub_diff[160:259] == max(t_sub_diff[160:259],key=abs)]),2)))
        print('relative max. temp difference SUBSOIL '+ regions[k] + ': ' + str(round(float(temp_i[160:259,z_sub][t_sub_diff[160:259] == max(t_sub_diff[160:259],key=abs)]/max(t_sub_diff[160:259]*100, key=abs))*100,2))+'%')





pd.DataFrame(z_lims).to_csv(plotpath+'td_all.csv',header=None)
plt.tight_layout(rect=[-0.1, 0.0, 1, 1])
plt.savefig(plotpath + 'td_deltaT_plots.png', bbox_inches='tight',dpi=300)
pd.DataFrame(al_depths, index=textures, columns=regions).to_csv(plotpath+'al_depth_dry_c.csv')

col_names = []
for r in regions:
    for s in sims:
        col_names.append(r + '_' + s)
s=160
e=259
t_sub_diff = (t_sub_i - t_sub_c)
t_top_diff = (t_top_i - t_top_c)

tc_sub_diff =  (tc_sub_i/tc_sub_c-1)*100
tc_top_diff = (tc_top_i/tc_top_c-1)*100

t_sub_diff_df = pd.DataFrame(t_sub_diff, columns=col_names)
t_top_diff_df = pd.DataFrame(t_top_diff, columns=col_names)
t_top_c_df = pd.DataFrame(t_top_c, columns=col_names)
t_top_i_df = pd.DataFrame(t_top_i, columns=col_names)
t_sub_c_df = pd.DataFrame(t_sub_c, columns=col_names)
t_sub_i_df = pd.DataFrame(t_sub_i, columns=col_names)

t_pf_c_df = pd.DataFrame(t_pf_c, columns=col_names)
t_pf_i_df = pd.DataFrame(t_pf_i, columns=col_names)

tc_sub_diff_df = pd.DataFrame(tc_sub_diff, columns=col_names)
tc_top_diff_df = pd.DataFrame(tc_top_diff, columns=col_names)

lsat_sub_df = pd.DataFrame(lsat_sub_i, columns=col_names)
lsat_top_df = pd.DataFrame(lsat_top_i, columns=col_names)

dr3 = pd.date_range(start='01/01/2017', end='12/31/2018', freq='D')
dtFmt = mdates.DateFormatter('%b') # define the formatting
dtFmt2 = mdates.DateFormatter('%b-%d')

smoothing = 7
rain_day_indicator_top = np.zeros(e-s)
rain_day_indicator_sub = np.zeros(e-s)
rain_day_indicator_top[[6,36,67]] = t_top_diff[s:e].max()
rain_day_indicator_sub[[6,36,67]] = t_sub_diff[s:e].max()


# plot temp differences over the entire 2-year period and the immediate
# temperature response after the precipitation treatment
# 2 year difference
plt.figure(figsize=(7,5))

blues = cm.get_cmap('Blues')
blues_sub = blues(np.linspace(0.15,0.95,5))
reds = cm.get_cmap('Reds')
reds_sub = reds(np.linspace(0.15,0.95,6))
color_list = np.concatenate([reds_sub, blues_sub])
color = iter(color_list)
color = iter(cm.turbo(np.linspace(0, 1, len(regions))))

for r in regions:
    c = next(color)
    plt.plot(dr3,t_top_diff_df[r+'_dry'].rolling(smoothing,1).mean(),label = r, c=c)
plt.axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
plt.legend(loc = 'best',facecolor='white',framealpha=0.5)
plt.gca().xaxis.set_major_formatter(dtFmt)
plt.ylabel('${\Delta}$ T ($^\circ$C)')
plt.axvline(dt.datetime(2018, 1, 1), color = 'grey', linestyle = '--', alpha=0.4)
plt.title('Topsoil (25% ALT) temperature diffrence (EXTREME)')
plt.savefig(plotpath + 'topsoil_Tdiff.png', bbox_inches='tight',dpi=300)

plt.figure(figsize=(7,5))
color = iter(cm.turbo(np.linspace(0, 1, len(regions))))
for r in regions:
    c = next(color)
    plt.plot(dr3,t_sub_diff_df[r+'_dry'].rolling(smoothing,1).mean(),label = r, c=c)
plt.axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
plt.legend(loc = 'best',facecolor='white',framealpha=0.5)
plt.gca().xaxis.set_major_formatter(dtFmt)
plt.axvline(dt.datetime(2018, 1, 1), color = 'grey', linestyle = '--', alpha=0.4)
plt.ylabel('${\Delta}$ T ($^\circ$C)')
plt.title('Subsoil (75% ALT) temperature diffrence (EXTREME)')
plt.savefig(plotpath + 'subsoil_Tdiff.png', bbox_inches='tight',dpi=300)

# immediate difference
color = ["darkorange", "navy", "darkorange", "navy"]
ls = ['-', '-', '--', '--']
plt.figure(figsize=(7,5))
for rr,r in enumerate(regions):
    c = color[rr]
    plt.plot(dr3[s:e],t_top_diff_df[s:e][r+'_dry'].rolling(smoothing,1).mean(),
    label = r_labels[rr], c=c, linestyle=ls[rr])
    #print(len(t_top_diff_df[s:e][r+'_dry'].rolling(smoothing,1).mean()))
    # print('absolute max. temp difference SUBSOIL: '+ regions[rr] + ': ' + str(round(max(t_sub_diff[s:e,rr], key=abs),3)))
    # print('absolute max. temp difference TOPSOIL: '+ regions[rr] + ': ' + str(round(max(t_top_diff[s:e,rr], key=abs),3)))
plt.axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
plt.legend(loc = 'best',facecolor='white',framealpha=0.5,ncol=1)
plt.gca().xaxis.set_major_formatter(dtFmt2)
plt.ylabel('${\Delta}$ T ($^\circ$C)')
#plt.title('Immediate topsoil (25% ALT) T-difference (EXTREME)')
plt.savefig(plotpath + 'immediate_topsoil_Tdiff.png', bbox_inches='tight',dpi=300)

integrated_df = np.zeros((len(regions),1))
#regions=['cd','cw','wd','ww']
plt.figure(figsize=(7,5))
color = iter(cm.turbo(np.linspace(0, 1, len(regions))))
color = ["darkorange", "navy", "darkorange", "navy"]
for rr,r in enumerate(regions):
    c = color[rr]
    plt.plot(dr3[s:e],t_sub_diff_df[s:e][r+'_dry'].rolling(smoothing,1).mean(),
        label = r_labels[rr], c=c, linestyle=ls[rr])
    integrated = np.trapz(t_sub_diff_df[s:e][r+'_dry'], t_sub_diff_df[s:e].index)
    integrated_df[rr] = np.round(integrated,2)
plt.axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
plt.bar(dr3[s:e],rain_day_indicator_sub, color = 'gainsboro', alpha = 0.6)
plt.legend(loc = 'best',facecolor='white',framealpha=0.5,ncol=1)
plt.gca().xaxis.set_major_formatter(dtFmt2)
plt.ylabel('${\Delta}$ T ($^\circ$C)')
plt.savefig(plotpath + 'MAINRESULT.png', bbox_inches='tight',dpi=300)
pd.DataFrame(integrated_df, index=regions, columns=["integrated"]).to_csv(plotpath+'integrated_deltaT.csv')


months = mdates.MonthLocator(interval = 1)


alt_df = pd.read_csv(plotpath+'alt.csv')
fig,ax = plt.subplots(1,2, figsize=(10,3.5))
#fig = plt.figure(figsize=(6,8))
for rr,r in enumerate(regions):
    c = color[rr]
    ax[0].plot(dr3[s:e],t_sub_diff_df[s:e][r+'_dry'].rolling(smoothing,1).mean(),
        label = r_labels[rr], c=c, linestyle=ls[rr])
    # print('SUBSOIL max difference in ' + r + ' is: ' + str(np.round(t_sub_diff_df[s:e][r+'_dry'].rolling(smoothing,1).mean().max(),3)))
    # print('SUBSOIL min difference in ' + r + ' is: ' + str(np.round(t_sub_diff_df[s:e][r+'_dry'].rolling(smoothing,1).mean().min(),3)))
ax[0].axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
ax[0].xaxis.set_major_formatter(dtFmt)
ax[0].xaxis.set_major_locator(months)
ax[0].set_ylabel('${\Delta}$ T [$^\circ$C]')
#axin=fig.add_axes([1,0,0.5,0.8])
#axin = ax[0].inset_axes([0.73, 0.73, 0.25, 0.25])
#alt_df.plot.bar(color={'ref':"darkorange",'HR':'navy'},ax=axin)
#axin.set_xticklabels(['cd', 'cw', 'wd', 'ww'], rotation=0)
#axin.legend(frameon=False)
#ax[0].set_ylim(-0.15, 0.4)
ax[0].text(0.01, 0.99, 'a) subsoil', transform=ax[0].transAxes, va='top', ha='left', fontsize=15, fontweight='bold')
ax[0].axvline(dt.datetime(2017, 6, 15), color = 'gainsboro', alpha = 0.5,lw=3)
ax[0].axvline(dt.datetime(2017, 7, 15), color = 'gainsboro', alpha = 0.5,lw=3)
ax[0].axvline(dt.datetime(2017, 8, 15), color = 'gainsboro', alpha = 0.5,lw=3)
#ax[0].bar(dr3[s:e],rain_day_indicator_sub, color = 'gainsboro', alpha = 0.7)


for rr,r in enumerate(regions):
    c = color[rr]
    ax[1].plot(dr3[s:e],t_top_diff_df[s:e][r+'_dry'].rolling(smoothing,1).mean(),
        label = r_labels[rr], c=c, linestyle=ls[rr])
    # print('TOPSOIL max difference in ' + r + ' is: ' + str(np.round(t_top_diff_df[s:e][r+'_dry'].rolling(smoothing,1).mean().max(),3)))
    # print('topSOIL min difference in ' + r + ' is: ' + str(np.round(t_top_diff_df[s:e][r+'_dry'].rolling(smoothing,1).mean().min(),3)))
#ax[1].bar(dr3[s:e],rain_day_indicator_top, color = 'gainsboro', alpha = 0.7)
ax[1].axvline(dt.datetime(2017, 6, 15), color = 'gainsboro', alpha = 0.5,lw=3)
ax[1].axvline(dt.datetime(2017, 7, 15), color = 'gainsboro', alpha = 0.5,lw=3)
ax[1].axvline(dt.datetime(2017, 8, 15), color = 'gainsboro', alpha = 0.5,lw=3)
ax[1].axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
ax[1].legend(loc = 'best',facecolor='white',framealpha=0.5,ncol=1,frameon=False)
ax[1].xaxis.set_major_formatter(dtFmt)
ax[1].xaxis.set_major_locator(months)
#ax[1].set_ylabel('${\Delta}$ T [$^\circ$C]')
ax[1].text(0.01, 0.99, 'b) topsoil', transform=ax[1].transAxes, va='top', ha='left', fontsize=15, fontweight='bold')
plt.tight_layout()

fig.savefig(plotpath + 'both_deltaT.png', bbox_inches='tight',dpi=300)
fig.savefig(plotpath + 'both_deltaT_transparent.png', bbox_inches='tight',dpi=300, transparent=True)
fig.savefig(plotpath + 'both_deltaT.svg', bbox_inches='tight',dpi=300)
fig.savefig(plotpath + 'both_deltaT.pdf', bbox_inches='tight',dpi=300)
fig.savefig(plotpath + 'both_deltaT_transparent.svg', bbox_inches='tight',dpi=300, transparent=True)

fig,ax = plt.subplots(1,2, figsize=(10,3.5))


for rr,r in enumerate(regions):
    y1 = np.array(t_sub_i_df[s:e][r+'_dry'].rolling(smoothing,1).mean())
    y2 = np.array(t_sub_c_df[s:e][r+'_dry'].rolling(smoothing,1).mean())
    c = color[rr]
    #ax[0].plot(dr3[s:e],t_sub_i_df[s:e][r+'_dry'].rolling(smoothing,1).mean(),
    #    label = r_labels[rr], c=c, linestyle=ls[rr], alpha = 0.3)
    ax[0].plot(dr3[s:e],t_sub_c_df[s:e][r+'_dry'].rolling(smoothing,1).mean(),
        c=c, linestyle=ls[rr], lw=1,label = r_labels[rr])
    ax[0].fill_between(dr3[s:e], y1, y2,color= c, alpha=0.4, edgecolor="none")
ax[0].axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
ax[0].xaxis.set_major_formatter(dtFmt)
ax[0].xaxis.set_major_locator(months)
ax[0].set_ylabel('T [$^\circ$C]')

ax[0].text(0.01, 0.99, 'a) subsoil', transform=ax[0].transAxes, va='top', ha='left', fontsize=15, fontweight='bold')
ax[0].axvline(dt.datetime(2017, 6, 15), color = 'gainsboro', alpha = 0.5,lw=3)
ax[0].axvline(dt.datetime(2017, 7, 15), color = 'gainsboro', alpha = 0.5,lw=3)
ax[0].axvline(dt.datetime(2017, 8, 15), color = 'gainsboro', alpha = 0.5,lw=3)

for rr,r in enumerate(regions):
    c = color[rr]
    #ax[1].plot(dr3[s:e],t_top_i_df[s:e][r+'_dry'].rolling(smoothing,1).mean(),
    #    label = r_labels[rr], c=c, linestyle=ls[rr], alpha = 0.3)
    ax[1].plot(dr3[s:e],t_top_c_df[s:e][r+'_dry'].rolling(smoothing,1).mean(),
         c=c, linestyle=ls[rr],lw=1,label = r_labels[rr])
    ax[1].fill_between(dr3[s:e],t_top_i_df[s:e][r+'_dry'].rolling(smoothing,1).mean(),
        t_top_c_df[s:e][r+'_dry'].rolling(smoothing,1).mean(), color= c, alpha=0.4, edgecolor="none")

ax[1].axvline(dt.datetime(2017, 6, 15), color = 'gainsboro', alpha = 0.5,lw=3)
ax[1].axvline(dt.datetime(2017, 7, 15), color = 'gainsboro', alpha = 0.5,lw=3)
ax[1].axvline(dt.datetime(2017, 8, 15), color = 'gainsboro', alpha = 0.5,lw=3)
ax[1].axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
ax[0].legend(loc = 'best',facecolor='white',framealpha=0.5,ncol=1,frameon=False)
ax[1].xaxis.set_major_formatter(dtFmt)
ax[1].xaxis.set_major_locator(months)
#ax[1].set_ylabel('${\Delta}$ T [$^\circ$C]')
ax[1].text(0.01, 0.99, 'b) topsoil', transform=ax[1].transAxes, va='top', ha='left', fontsize=15, fontweight='bold')
plt.tight_layout()

fig.savefig(plotpath + 'both_actualT.png', bbox_inches='tight',dpi=300)
# fig.savefig(plotpath + 'both_deltaT_transparent.png', bbox_inches='tight',dpi=300, transparent=True)
# fig.savefig(plotpath + 'both_deltaT.svg', bbox_inches='tight',dpi=300)
# fig.savefig(plotpath + 'both_deltaT_transparent.svg', bbox_inches='tight',dpi=300, transparent=True)



# plot thermal conductivity differences over the entire 2-year period
color = ["darkorange", "navy", "darkorange", "navy"]
plt.figure(figsize=(7,5))
for rr,r in enumerate(regions):
    c = color[rr]
    plt.plot(dr3[s:e],tc_top_diff_df[r+'_dry'][s:e],
    label = r_labels[rr], c=c, linestyle=ls[rr])
plt.axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
plt.legend(loc = 'best',facecolor='white',framealpha=0.5)
plt.gca().xaxis.set_major_formatter(dtFmt)
plt.ylabel('${\Delta}$ TC (%)')
#plt.title('Topsoil (25% AL depth) TC diffrence (EXTREME)')
plt.savefig(plotpath + 'topsoil_TCdiff.png', bbox_inches='tight',dpi=300)

plt.figure(figsize=(7,5))
for rr,r in enumerate(regions):
    c = color[rr]
    plt.plot(dr3[s:e],tc_sub_diff_df[r+'_dry'][s:e],label = r_labels[rr], c=c,linestyle=ls[rr])
plt.axhline(y=0, color = 'grey', linestyle='--', linewidth=0.8)
plt.legend(loc = 'best',facecolor='white',framealpha=0.5)
plt.gca().xaxis.set_major_formatter(dtFmt)
plt.ylabel('${\Delta}$ TC (%)')
plt.title('Subsoil (75% AL depth) TC diffrence (EXTREME)')
plt.savefig(plotpath + 'subsoil_TCdiff.png', bbox_inches='tight',dpi=300)


### time-temperature plots for specific depths in specific regions #####
t_top_c_df = pd.DataFrame(t_top_c, columns=col_names)
t_top_i_df = pd.DataFrame(t_top_i, columns=col_names)
t_sub_c_df = pd.DataFrame(t_sub_c, columns=col_names)
t_sub_i_df = pd.DataFrame(t_sub_i, columns=col_names)
t_top_c_df.to_csv(plotpath+'topsoil_t_ref_case.csv')
t_top_i_df.to_csv(plotpath+'topsoil_t_heavy_rain.csv')
t_sub_c_df.to_csv(plotpath+'subsoil_t_ref_case.csv')
t_sub_i_df.to_csv(plotpath+'subsoil_t_heavy_rain.csv')
