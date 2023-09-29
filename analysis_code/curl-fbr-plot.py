import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import numpy.ma as ma
import pandas as pd 
import funpy.model_utils as mod_utils
import cmocean.cm as cmo 
import re
import glob
from scipy.signal import welch
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
import datetime
from matplotlib.colors import ListedColormap

plt.style.use('classic')

dx = 0.05; dy = 0.1; dt = 0.2
plotsavedir = os.path.join('/gscratch', 'nearshore', 'enuss', 'lab_runs_y550', 'postprocessing', 'plots')

rundir = 'hmo25_dir1_tp2'
rootdir = os.path.join('/gscratch', 'nearshore','enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'crest_1.nc'), os.path.join(fdir, 'crest_2.nc'), os.path.join(fdir, 'crest_3.nc'), os.path.join(fdir, 'crest_4.nc')]
crests1 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['labels']

fbrxflist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbryflist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]

fbrx1 = xr.open_mfdataset(fbrxflist, combine='nested', concat_dim='time')['fbrx']
fbry1 = xr.open_mfdataset(fbryflist, combine='nested', concat_dim='time')['fbry']
x = xr.open_mfdataset(fbrxflist, combine='nested', concat_dim='time')['x']
y = xr.open_mfdataset(fbrxflist, combine='nested', concat_dim='time')['y']

curlfbr1 = np.gradient(fbry1, dx, axis=2) - np.gradient(fbrx1, dy, axis=1)

uflist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
vflist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]

u_psi1 = xr.open_mfdataset(uflist, combine='nested', concat_dim='time')['u_psi']
v_psi1 = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')['v_psi']

vort_psi1 = np.gradient(v_psi1, dx, axis=2) - np.gradient(u_psi1, dy, axis=1)

rundir = 'hmo25_dir10_tp2'
rootdir = os.path.join('/gscratch', 'nearshore','enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'crest_1.nc'), os.path.join(fdir, 'crest_2.nc'), os.path.join(fdir, 'crest_3.nc'), os.path.join(fdir, 'crest_4.nc')]
crests10 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['labels']

fbrxflist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbryflist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]

fbrx10 = xr.open_mfdataset(fbrxflist, combine='nested', concat_dim='time')['fbrx']
fbry10 = xr.open_mfdataset(fbryflist, combine='nested', concat_dim='time')['fbry']

curlfbr10 = np.gradient(fbry10, dx, axis=2) - np.gradient(fbrx10, dy, axis=1)

uflist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
vflist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]

u_psi10 = xr.open_mfdataset(uflist, combine='nested', concat_dim='time')['u_psi']
v_psi10 = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')['v_psi']

vort_psi10 = np.gradient(v_psi10, dx, axis=2) - np.gradient(u_psi10, dy, axis=1)

rundir = 'hmo25_dir40_tp2'
rootdir = os.path.join('/gscratch', 'nearshore','enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'crest_1.nc'), os.path.join(fdir, 'crest_2.nc'), os.path.join(fdir, 'crest_3.nc'), os.path.join(fdir, 'crest_4.nc')]
crests30 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['labels']

fbrxflist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbryflist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]

fbrx30 = xr.open_mfdataset(fbrxflist, combine='nested', concat_dim='time')['fbrx']
fbry30 = xr.open_mfdataset(fbryflist, combine='nested', concat_dim='time')['fbry']

curlfbr30 = np.gradient(fbry30, dx, axis=2) - np.gradient(fbrx30, dy, axis=1)

uflist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
vflist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]

u_psi30 = xr.open_mfdataset(uflist, combine='nested', concat_dim='time')['u_psi']
v_psi30 = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')['v_psi']

vort_psi30 = np.gradient(v_psi30, dx, axis=2) - np.gradient(u_psi30, dy, axis=1)

xx, yy = np.meshgrid(x-22, y-np.max(y)/2)
fmax = 15; vmax = 1

crest_cmap = ListedColormap(['#f1eceb','#3a617d'])
fsize = 16

for i in range(len(curlfbr1)):
    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(13,18), sharex=True, sharey=True)
    crest1_tmp = crests1[i,:,:].values
    crest1_tmp[crest1_tmp>0] = 1
    p00 = ax[0,0].pcolormesh(xx,yy, crest1_tmp, cmap=crest_cmap)
    cbar00 = fig.colorbar(p00, ax=ax[0,0], label=r'$\mathrm{Crest}$')
    cbar00.set_ticks([])
    ax[0,0].set_title(r'$\sigma_\theta = 0.2^\circ$', fontsize=fsize)
    ax[0,0].set_ylabel(r'$y\ \mathrm{(m)}$', fontsize=fsize)
    ax[0,0].text(16,24, r'$\mathrm{(a)}$', fontsize=fsize)
    ax[0,0].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    crest10_tmp = crests10[i,:,:].values
    crest10_tmp[crest10_tmp>0] = 1
    p01 = ax[0,1].pcolormesh(xx,yy, crest10_tmp, cmap=crest_cmap)
    cbar01 = fig.colorbar(p01, ax=ax[0,1], label=r'$\mathrm{Crest}$')
    cbar01.set_ticks([])
    ax[0,1].set_title(r'$\sigma_\theta = 9.6^\circ$', fontsize=fsize)
    ax[0,1].text(16,24, r'$\mathrm{(b)}$', fontsize=fsize)
    ax[0,1].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    crest30_tmp = crests30[i,:,:].values
    crest30_tmp[crest30_tmp>0] = 1
    p02 = ax[0,2].pcolormesh(xx,yy, crest30_tmp, cmap=crest_cmap)
    cbar02 = fig.colorbar(p02, ax=ax[0,2], label=r'$\mathrm{Crest}$')    
    cbar02.set_ticks([])
    ax[0,2].set_title(r'$\sigma_\theta = 25.6^\circ$', fontsize=fsize)
    ax[0,2].text(16,24, r'$\mathrm{(c)}$', fontsize=fsize)
    ax[0,2].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    p10 = ax[1,0].pcolormesh(xx,yy, curlfbr1[i,:,:], cmap=cmo.balance)
    fig.colorbar(p10, ax=ax[1,0], label=r'$\nabla \times F_{\bf br}\ \mathrm{(s^{-2})}$')
    p10.set_clim(-fmax,fmax)
    ax[1,0].set_ylabel(r'$y\ \mathrm{(m)}$', fontsize=fsize)
    ax[1,0].text(16,24, r'$\mathrm{(d)}$', fontsize=fsize)
    ax[1,0].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    p11 = ax[1,1].pcolormesh(xx,yy, curlfbr10[i,:,:], cmap=cmo.balance)
    fig.colorbar(p11, ax=ax[1,1], label=r'$\nabla \times F_{\bf br}\ \mathrm{(s^{-2})}$')
    p11.set_clim(-fmax,fmax)
    ax[1,1].text(16,24, r'$\mathrm{(e)}$', fontsize=fsize)
    ax[1,1].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    p12 = ax[1,2].pcolormesh(xx,yy, curlfbr30[i,:,:], cmap=cmo.balance)
    fig.colorbar(p12, ax=ax[1,2], label=r'$\nabla \times F_{\bf br}\ \mathrm{(s^{-2})}$')    
    p12.set_clim(-fmax,fmax)
    ax[1,2].text(16,24, r'$\mathrm{(f)}$', fontsize=fsize)
    ax[1,2].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    p20 = ax[2,0].pcolormesh(xx,yy, vort_psi1[i,:,:], cmap=cmo.balance)
    fig.colorbar(p20, ax=ax[2,0], label=r'$\nabla \times \bf{u_{\psi}}\ \mathrm{(s^{-1})}$')
    p20.set_clim(-vmax,vmax)
    ax[2,0].set_xlabel(r'$x\ (\mathrm{m})$', fontsize=fsize)
    ax[2,0].set_ylabel(r'$y\ \mathrm{(m)}$', fontsize=fsize)
    ax[2,0].text(16,24, r'$\mathrm{(g)}$', fontsize=fsize)
    ax[2,0].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    p21 = ax[2,1].pcolormesh(xx,yy, vort_psi10[i,:,:], cmap=cmo.balance)
    fig.colorbar(p21, ax=ax[2,1], label=r'$\nabla \times \bf{u_{\psi}}\ \mathrm{(s^{-1})}$')
    p21.set_clim(-vmax,vmax)
    ax[2,1].set_xlabel(r'$x\ \mathrm{(m)}$', fontsize=fsize)
    ax[2,1].text(16,24, r'$\mathrm{(h)}$', fontsize=fsize)
    ax[2,1].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    p22 = ax[2,2].pcolormesh(xx,yy, vort_psi30[i,:,:], cmap=cmo.balance)
    fig.colorbar(p22, ax=ax[2,2], label=r'$\nabla \times \bf{u_{\psi}}\ \mathrm{(s^{-1})}$')    
    p22.set_clim(-vmax,vmax)
    ax[2,2].set_xlim(15,34)
    ax[2,2].set_ylim(np.min(yy), np.max(yy))
    ax[2,2].set_xlabel(r'$x\ \mathrm{(m)}$', fontsize=fsize)
    ax[2,2].text(16,24, r'$\mathrm{(i)}$', fontsize=fsize)
    ax[2,2].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 
    ax[2,2].set_xlim(15,32.5)

    #fig.suptitle(r'%d $\mathrm{seconds}$' % (i*dt), fontsize=16)
    fig.savefig(os.path.join(plotsavedir, 'snaps', '%05d.png' % i))
    plt.close()

#########################################################################

rundir = 'hmo25_dir20_tp15'
rootdir = os.path.join('/gscratch', 'nearshore','enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'crest_1.nc'), os.path.join(fdir, 'crest_2.nc'), os.path.join(fdir, 'crest_3.nc'), os.path.join(fdir, 'crest_4.nc')]
crests1 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['labels']

fbrxflist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbryflist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]

fbrx1 = xr.open_mfdataset(fbrxflist, combine='nested', concat_dim='time')['fbrx']
fbry1 = xr.open_mfdataset(fbryflist, combine='nested', concat_dim='time')['fbry']
x = xr.open_mfdataset(fbrxflist, combine='nested', concat_dim='time')['x']
y = xr.open_mfdataset(fbrxflist, combine='nested', concat_dim='time')['y']

curlfbr1 = np.gradient(fbry1, dx, axis=2) - np.gradient(fbrx1, dy, axis=1)

uflist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
vflist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]

u_psi1 = xr.open_mfdataset(uflist, combine='nested', concat_dim='time')['u_psi']
v_psi1 = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')['v_psi']

vort_psi1 = np.gradient(v_psi1, dx, axis=2) - np.gradient(u_psi1, dy, axis=1)

rundir = 'hmo25_dir20_tp2'
rootdir = os.path.join('/gscratch', 'nearshore','enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'crest_1.nc'), os.path.join(fdir, 'crest_2.nc'), os.path.join(fdir, 'crest_3.nc'), os.path.join(fdir, 'crest_4.nc')]
crests10 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['labels']

fbrxflist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbryflist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]

fbrx10 = xr.open_mfdataset(fbrxflist, combine='nested', concat_dim='time')['fbrx']
fbry10 = xr.open_mfdataset(fbryflist, combine='nested', concat_dim='time')['fbry']

curlfbr10 = np.gradient(fbry10, dx, axis=2) - np.gradient(fbrx10, dy, axis=1)

uflist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
vflist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]

u_psi10 = xr.open_mfdataset(uflist, combine='nested', concat_dim='time')['u_psi']
v_psi10 = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')['v_psi']

vort_psi10 = np.gradient(v_psi10, dx, axis=2) - np.gradient(u_psi10, dy, axis=1)

rundir = 'hmo25_dir20_tp25'
rootdir = os.path.join('/gscratch', 'nearshore','enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'crest_1.nc'), os.path.join(fdir, 'crest_2.nc'), os.path.join(fdir, 'crest_3.nc'), os.path.join(fdir, 'crest_4.nc')]
crests30 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['labels']

fbrxflist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbryflist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]

fbrx30 = xr.open_mfdataset(fbrxflist, combine='nested', concat_dim='time')['fbrx']
fbry30 = xr.open_mfdataset(fbryflist, combine='nested', concat_dim='time')['fbry']

curlfbr30 = np.gradient(fbry30, dx, axis=2) - np.gradient(fbrx30, dy, axis=1)

uflist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
vflist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]

u_psi30 = xr.open_mfdataset(uflist, combine='nested', concat_dim='time')['u_psi']
v_psi30 = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')['v_psi']

vort_psi30 = np.gradient(v_psi30, dx, axis=2) - np.gradient(u_psi30, dy, axis=1)

xx, yy = np.meshgrid(x-22, y-np.max(y)/2)
fmax = 15; vmax = 1

crest_cmap = ListedColormap(['#f1eceb','#3a617d'])
fsize = 16

for i in range(len(curlfbr1)):
    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(13,18), sharex=True, sharey=True)
    crest1_tmp = crests1[i,:,:].values
    crest1_tmp[crest1_tmp>0] = 1
    p00 = ax[0,0].pcolormesh(xx,yy, crest1_tmp, cmap=crest_cmap)
    cbar00 = fig.colorbar(p00, ax=ax[0,0], label=r'$\mathrm{Crest}$')
    cbar00.set_ticks([])
    ax[0,0].set_title(r'$T_p = 1.5\ \mathrm{s}$', fontsize=fsize)
    ax[0,0].set_ylabel(r'$y\ \mathrm{(m)}$', fontsize=fsize)
    ax[0,0].text(16,24, r'$\mathrm{(a)}$', fontsize=fsize)
    ax[0,0].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    crest10_tmp = crests10[i,:,:].values
    crest10_tmp[crest10_tmp>0] = 1
    p01 = ax[0,1].pcolormesh(xx,yy, crest10_tmp, cmap=crest_cmap)
    cbar01 = fig.colorbar(p01, ax=ax[0,1], label=r'$\mathrm{Crest}$')
    cbar01.set_ticks([])
    ax[0,1].set_title(r'$T_p = 2.0\ \mathrm{s}$', fontsize=fsize)
    ax[0,1].text(16,24, r'$\mathrm{(b)}$', fontsize=fsize)
    ax[0,1].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    crest30_tmp = crests30[i,:,:].values
    crest30_tmp[crest30_tmp>0] = 1
    p02 = ax[0,2].pcolormesh(xx,yy, crest30_tmp, cmap=crest_cmap)
    cbar02 = fig.colorbar(p02, ax=ax[0,2], label=r'$\mathrm{Crest}$')    
    cbar02.set_ticks([])
    ax[0,2].set_title(r'$T_p = 2.5\ \mathrm{s}$', fontsize=fsize)
    ax[0,2].text(16,24, r'$\mathrm{(c)}$', fontsize=fsize)
    ax[0,2].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    p10 = ax[1,0].pcolormesh(xx,yy, curlfbr1[i,:,:], cmap=cmo.balance)
    fig.colorbar(p10, ax=ax[1,0], label=r'$\nabla \times F_{\bf br}\ \mathrm{(s^{-2})}$')
    p10.set_clim(-fmax,fmax)
    ax[1,0].set_ylabel(r'$y\ \mathrm{(m)}$', fontsize=fsize)
    ax[1,0].text(16,24, r'$\mathrm{(d)}$', fontsize=fsize)
    ax[1,0].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    p11 = ax[1,1].pcolormesh(xx,yy, curlfbr10[i,:,:], cmap=cmo.balance)
    fig.colorbar(p11, ax=ax[1,1], label=r'$\nabla \times F_{\bf br}\ \mathrm{(s^{-2})}$')
    p11.set_clim(-fmax,fmax)
    ax[1,1].text(16,24, r'$\mathrm{(e)}$', fontsize=fsize)
    ax[1,1].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    p12 = ax[1,2].pcolormesh(xx,yy, curlfbr30[i,:,:], cmap=cmo.balance)
    fig.colorbar(p12, ax=ax[1,2], label=r'$\nabla \times F_{\bf br}\ \mathrm{(s^{-2})}$')    
    p12.set_clim(-fmax,fmax)
    ax[1,2].text(16,24, r'$\mathrm{(f)}$', fontsize=fsize)
    ax[1,2].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    p20 = ax[2,0].pcolormesh(xx,yy, vort_psi1[i,:,:], cmap=cmo.balance)
    fig.colorbar(p20, ax=ax[2,0], label=r'$\nabla \times \bf{u_{\psi}}\ \mathrm{(s^{-1})}$')
    p20.set_clim(-vmax,vmax)
    ax[2,0].set_xlabel(r'$x\ (\mathrm{m})$', fontsize=fsize)
    ax[2,0].set_ylabel(r'$y\ \mathrm{(m)}$', fontsize=fsize)
    ax[2,0].text(16,24, r'$\mathrm{(g)}$', fontsize=fsize)
    ax[2,0].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    p21 = ax[2,1].pcolormesh(xx,yy, vort_psi10[i,:,:], cmap=cmo.balance)
    fig.colorbar(p21, ax=ax[2,1], label=r'$\nabla \times \bf{u_{\psi}}\ \mathrm{(s^{-1})}$')
    p21.set_clim(-vmax,vmax)
    ax[2,1].set_xlabel(r'$x\ \mathrm{(m)}$', fontsize=fsize)
    ax[2,1].text(16,24, r'$\mathrm{(h)}$', fontsize=fsize)
    ax[2,1].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 

    p22 = ax[2,2].pcolormesh(xx,yy, vort_psi30[i,:,:], cmap=cmo.balance)
    fig.colorbar(p22, ax=ax[2,2], label=r'$\nabla \times \bf{u_{\psi}}\ \mathrm{(s^{-1})}$')    
    p22.set_clim(-vmax,vmax)
    ax[2,2].set_xlim(15,34)
    ax[2,2].set_ylim(np.min(yy), np.max(yy))
    ax[2,2].set_xlabel(r'$x\ \mathrm{(m)}$', fontsize=fsize)
    ax[2,2].text(16,24, r'$\mathrm{(i)}$', fontsize=fsize)
    ax[2,2].axvspan(31.5, 32.5, color='tab:grey', alpha=0.3) 
    ax[2,2].set_xlim(15,32.5)

    fig.savefig(os.path.join(plotsavedir, 'snaps_tp', '%05d.png' % i))
    plt.close()


