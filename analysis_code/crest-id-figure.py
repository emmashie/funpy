import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import pandas as pd
import xarray as xr
import funpy.postprocess as fp
import funpy.model_utils as mod_utils
import cmocean.cm as cmo 
from funpy import filter_functions as ff 
from matplotlib.colors import ListedColormap

plotsavedir = os.path.join('/gscratch', 'nearshore', 'enuss', 'lab_runs_y550', 'postprocessing', 'plots')

dx = 0.05
dy = 0.1
arealim = 0.375 

def load_creststats(fdir, crests, crestfile, crestendsfile):
	crest_stats = np.loadtxt(crestfile, delimiter=',', skiprows=1)
	crestends = np.loadtxt(crestendsfile, delimiter=',')

	crestlen = crest_stats[:,0]
	minx = crest_stats[:,1] 
	miny = crest_stats[:,2] 
	maxx = crest_stats[:,3] 
	maxy = crest_stats[:,4] 
	crestfbr_std = crest_stats[:,5] 
	crestfbr_abs = crest_stats[:,6] 
	crestfbr_sq = crest_stats[:,7] 
	crestfbr_mean = crest_stats[:,8]
	time = crest_stats[:,9]

	avgx = (minx+maxx)/2

	ncrest = np.asarray([float(np.max(crests[i,:,:])) for i in range(len(crests))])

	areacrest = []
	for i in range(len(crests)):
		N = int(ncrest[i])
		for n in range(0,N):
			areacrest.append(len(np.where(crests[i,:,:]==n+1)[0])*dx*dy)
	areacrest = np.asarray(areacrest)

	return time, crestlen, avgx, minx, maxx, miny, maxy, crestfbr_std, crestfbr_abs, crestfbr_sq, crestfbr_mean, crestends, areacrest

def remove_small_crests(var, areas, arealim):
    return var[areas>arealim]

def restrict_cross_crests(var, avgx, sz=26+22, shore=32.5+22):
	return var[(avgx>sz)&(avgx<shore)]

rundir = 'lab_runs_y550'
outputdir = 'output_hmo25_dir10_tp2'
fdir = os.path.join('/gscratch', 'nearshore', 'enuss', rundir, outputdir)
savefulldir = os.path.join('/gscratch', 'nearshore', 'enuss', rundir, 'postprocessing', 'compiled_' + outputdir, 'full_netcdfs')
savedir = os.path.join('/gscratch', 'nearshore', 'enuss', rundir, 'postprocessing', 'compiled_' + outputdir, 'lab_netcdfs')

nubrk, x, y = mod_utils.load_masked_variable(savedir, 'nubrk', 'nubrk_*.nc', 'mask', 'mask_*.nc')
x = x - 22
nubrk = nubrk[1500:,:,:]
[xx, yy] = np.meshgrid(x, y)

flist = [os.path.join(savedir, 'fbrx_1.nc'), os.path.join(savedir, 'fbrx_2.nc'), os.path.join(savedir, 'fbrx_3.nc'), os.path.join(savedir, 'fbrx_4.nc')]
fbrx = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']
flist = [os.path.join(savedir, 'fbry_1.nc'), os.path.join(savedir, 'fbry_2.nc'), os.path.join(savedir, 'fbry_3.nc'), os.path.join(savedir, 'fbry_4.nc')]
fbry = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

fbr = np.gradient(fbry, 0.05, axis=2) - np.gradient(fbrx, 0.1, axis=1)

flist = [os.path.join(savedir, 'crest_1.nc'), os.path.join(savedir, 'crest_2.nc'), os.path.join(savedir, 'crest_3.nc'), os.path.join(savedir, 'crest_4.nc')]
crests = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['labels']

time_dir0_all, crestlen_dir0_all, avgx_dir0_all, minx_dir0_all, maxx_dir0_all, miny_dir0_all, maxy_dir0_all, \
fbr_std_dir0_all, fbr_abs_dir0_all, fbr_sq_dir0_all, fbr_mean_dir0_all, crestends_dir0_all, areacrest_dir0_all = \
    load_creststats(savedir, crests, os.path.join(savedir, 'crest_stats.txt'), os.path.join(savedir, 'crest_ends.txt'))

T = 1000
crest_ind_all = np.where(time_dir0_all==T)[0]

time_all = time_dir0_all[crest_ind_all]
avgx_all = avgx_dir0_all[crest_ind_all]
minx_all = minx_dir0_all[crest_ind_all]
maxx_all = maxx_dir0_all[crest_ind_all]
miny_all = miny_dir0_all[crest_ind_all]
maxy_all = maxy_dir0_all[crest_ind_all]

time_subset = remove_small_crests(time_all, areacrest_dir0_all[crest_ind_all], arealim)
avgx_subset = remove_small_crests(avgx_all, areacrest_dir0_all[crest_ind_all], arealim)
minx_subset = remove_small_crests(minx_all, areacrest_dir0_all[crest_ind_all], arealim)
maxx_subset = remove_small_crests(maxx_all, areacrest_dir0_all[crest_ind_all], arealim)
miny_subset = remove_small_crests(miny_all, areacrest_dir0_all[crest_ind_all], arealim)
maxy_subset = remove_small_crests(maxy_all, areacrest_dir0_all[crest_ind_all], arealim)
area_ind = np.where(areacrest_dir0_all[crest_ind_all]<arealim)[0]
sz=26+22; shore=32.5+22
cross_ind = np.where((avgx_all>sz)&(avgx_all<shore))[0]

time = restrict_cross_crests(time_subset, avgx_subset)
avgx = restrict_cross_crests(avgx_subset, avgx_subset)
minx = restrict_cross_crests(minx_subset, avgx_subset)
maxx = restrict_cross_crests(maxx_subset, avgx_subset)
miny = restrict_cross_crests(miny_subset, avgx_subset)
maxy = restrict_cross_crests(maxy_subset, avgx_subset)

crest_plot = crests[T,:,:].values
for i in range(1,len(avgx_all)+1):
	if np.any(i==cross_ind+1):
		if np.any(i==area_ind+1):
			crest_plot[crest_plot==i] = 100 # 100 == crest counted
		else:
			crest_plot[crest_plot==i] = 200 # 200 == crest too small
	else:
		crest_plot[crest_plot==i] = 300 # 300 == crest out of SZ region
		
crest_plot[crest_plot==100] = 1
crest_plot[crest_plot==200] = 2
crest_plot[crest_plot==300] = 3

SZmod = 32.5+22 - 49
SZxmin = 27
SZymin = 20 
SZxmax = SZxmin + SZmod 
SZymax = SZymin + SZmod 

labwidth = 26.5
ystart1 = 0.5
yend1 = ystart1+labwidth

numin = 0; numax = 0.15
fbrmin = -10; fbrmax = -fbrmin
fig, ax = plt.subplots(figsize=(9,6), ncols=4, sharex=True, sharey=True)

p0 = ax[0].pcolormesh(xx, yy, nubrk[T,:,:], cmap=cmo.ice_r)
fig.colorbar(p0, ax=ax[0], ticks=[0, 0.05, 0.1, 0.15], location='top', label=r'$\nu_{br}$')
p0.set_clim(numin, numax)
ax[0].plot(np.array([SZxmin, SZxmax]), np.array([SZymin, SZymin]), '--', color='grey')
ax[0].plot(np.array([SZxmin, SZxmax]), np.array([SZymax, SZymax]), '--', color='grey')
ax[0].plot(np.array([SZxmin, SZxmin]), np.array([SZymin, SZymax]), '--', color='grey')
ax[0].plot(np.array([SZxmax, SZxmax]), np.array([SZymin, SZymax]), '--', color='grey')
ax[0].set_aspect('equal', 'box')
ax[0].set_ylim(ystart1, yend1)
ax[0].set_ylabel('$y$ $\mathrm{(m)}$')
ax[0].set_xlabel('$x$ $\mathrm{(m)}$')
ax[0].yaxis.set_ticks([4.25, 9.25, 14.25, 19.25, 24.25], [-10, -5, 0, 5, 10])
ax[0].xaxis.set_ticks([25,29,33])
ax[0].text(25.5, 1, r'$\mathrm{(a)}$',  fontweight='bold', fontsize='15')

window = ff.lanczos_2Dwindow(y, x, 1, 0.5, 0.5)
var_bar = ff.lanczos_2D(nubrk[T,:,:].data, nubrk[T,:,:].mask, window, len(y), len(x))
p1 = ax[1].pcolormesh(xx, yy, var_bar, cmap=cmo.ice_r)
fig.colorbar(p1, ax=ax[1], ticks=[0, 0.05, 0.1, 0.15], location='top', label=r'$\overline{\nu_{br}}$')
p1.set_clim(numin, numax)
ax[1].plot(np.array([SZxmin, SZxmax]), np.array([SZymin, SZymin]), '--', color='grey')
ax[1].plot(np.array([SZxmin, SZxmax]), np.array([SZymax, SZymax]), '--', color='grey')
ax[1].plot(np.array([SZxmin, SZxmin]), np.array([SZymin, SZymax]), '--', color='grey')
ax[1].plot(np.array([SZxmax, SZxmax]), np.array([SZymin, SZymax]), '--', color='grey')
ax[1].set_aspect('equal', 'box')
ax[1].set_ylim(ystart1, yend1)
ax[1].set_xlabel('$x$ $\mathrm{(m)}$')
ax[1].text(25.5, 1, r'$\mathrm{(b)}$',  fontweight='bold', fontsize='15')

#crest_cmap = ListedColormap(['#3a617d', '#008f91', '#5cb352', '#ffb60d'])
crest_cmap = ListedColormap(['#f1eceb', '#fd6581', '#3a617d', '#ffa600'])
p2 = ax[2].pcolormesh(xx, yy, crest_plot, cmap=crest_cmap)
p2.set_clim(0,3)
ax[2].plot(minx-22, miny, 'x', color='black')
ax[2].plot(maxx-22, maxy, 'x', color='black')
cbar2 = fig.colorbar(p2, ax=ax[2], ticks=[0, 1, 2, 3], location='top', label=r'$\mathrm{Crests}$')
cbar2.ax.set_xticklabels(['0', '1', '2', '3']) 
ax[2].plot(np.array([SZxmin, SZxmax]), np.array([SZymin, SZymin]), '--', color='grey')
ax[2].plot(np.array([SZxmin, SZxmax]), np.array([SZymax, SZymax]), '--', color='grey')
ax[2].plot(np.array([SZxmin, SZxmin]), np.array([SZymin, SZymax]), '--', color='grey')
ax[2].plot(np.array([SZxmax, SZxmax]), np.array([SZymin, SZymax]), '--', color='grey')
ax[2].set_aspect('equal', 'box')
ax[2].set_ylim(ystart1, yend1)
ax[2].set_xlabel('$x$ $\mathrm{(m)}$')
ax[2].text(25.5, 1, r'$\mathrm{(c)}$',  fontweight='bold', fontsize='15')

p3 = ax[3].pcolormesh(xx, yy, fbr[T,:,:], cmap=cmo.balance)
ax[3].set_xlim(25, 33)
fig.colorbar(p3, ax=ax[3], location='top', label=r'$\nabla \times F_{br}$')
p3.set_clim(fbrmin, fbrmax)
ax[3].plot(np.array([SZxmin, SZxmax]), np.array([SZymin, SZymin]), '--', color='grey')
ax[3].plot(np.array([SZxmin, SZxmax]), np.array([SZymax, SZymax]), '--', color='grey')
ax[3].plot(np.array([SZxmin, SZxmin]), np.array([SZymin, SZymax]), '--', color='grey')
ax[3].plot(np.array([SZxmax, SZxmax]), np.array([SZymin, SZymax]), '--', color='grey')
ax[3].set_aspect('equal', 'box')
ax[3].set_ylim(ystart1, yend1)
ax[3].set_xlabel('$x$ $\mathrm{(m)}$')
ax[3].text(25.5, 1, r'$\mathrm{(d)}$',  fontweight='bold', fontsize='15')
fig.tight_layout()
fig.savefig(os.path.join(plotsavedir, 'crest_id.png'))
