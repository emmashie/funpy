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
import funpy.wave_functions as wf

start = datetime.datetime.now()

plt.ion()
#plt.style.use('ggplot')
plt.style.use('classic')

plotsavedir = os.path.join('/gscratch', 'nearshore', 'enuss', 'lab_runs_y550', 'postprocessing', 'plots')

dx = 0.05; dy = 0.1; dt = 0.2; g = 9.8; WL = 128; OL = 64; lf = 0.2; fmin = 0.25; fmax = 1.2

def compute_wave_stats(fdir, dt=0.2, lf=0.2, WL=128, OL=64, fmin=0.25, fmax=1.2):
    dep = np.loadtxt(os.path.join(fdir, 'dep.out'))

    eta_flist = [os.path.join(fdir, 'eta_1.nc'), os.path.join(fdir, 'eta_2.nc'), os.path.join(fdir, 'eta_3.nc'), os.path.join(fdir, 'eta_4.nc')]
    eta_dat = xr.open_mfdataset(eta_flist, combine='nested', concat_dim='time')
    eta = eta_dat['eta']
    x = eta_dat['x']
    y = eta_dat['y']

    uflist = [os.path.join(fdir, 'u_1.nc'), os.path.join(fdir, 'u_2.nc'), os.path.join(fdir, 'u_3.nc'), os.path.join(fdir, 'u_4.nc')]
    vflist = [os.path.join(fdir, 'v_1.nc'), os.path.join(fdir, 'v_2.nc'), os.path.join(fdir, 'v_3.nc'), os.path.join(fdir, 'v_4.nc')]

    u = xr.open_mfdataset(uflist, combine='nested', concat_dim='time')['u']
    v = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')['v']

    freq, Sf = welch(eta, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
    Hs = mod_utils.compute_Hsig_spectrally(freq, Sf, np.min(freq), np.max(freq))
    Hs_alongmean = np.nanmean(Hs, axis=0)

    xpos = 15 + 22 # near wave gages in lab
    xind = np.argmin(np.abs(x.values-xpos))

    Sf_alongmean = np.nanmean(Sf, axis=1)
    Tp_off = 1/freq[np.where(Sf_alongmean[:,xind]==np.max(Sf_alongmean[:,xind]))[0][0]]

    energy_density = wf.energy_density(Hs_alongmean/2)
    k = wf.wavenum(1/Tp_off*np.ones(len(dep[0,:])), dep[0,:])
    cg = wf.group_speed(2*np.pi/k, Tp_off, dep[0,:])
    energy_flux = wf.energy_flux(energy_density, cg)
    eflux_rat = energy_flux/energy_flux[xind]
    xsz_ind = np.argmin(np.abs(eflux_rat[np.isfinite(eflux_rat)]-0.89))
    xsz = x.values[xsz_ind]-22

    Tp_sz = 1/freq[np.where(Sf_alongmean[:,xsz_ind]==np.max(Sf_alongmean[:,xsz_ind]))[0][0]]

    dirspread_off, theta_off = mod_utils.calculate_dirspread(eta[:,:,xind], u[:,:,xind], v[:,:,xind], dt, lf, WL, OL)
    dirspread_sz, theta_sz = mod_utils.calculate_dirspread(eta[:,:,xsz_ind], u[:,:,xsz_ind], v[:,:,xsz_ind], dt, lf, WL, OL)

    maskflist = [os.path.join(fdir, 'mask_1.nc'), os.path.join(fdir, 'mask_2.nc'), os.path.join(fdir, 'mask_3.nc'), os.path.join(fdir, 'mask_4.nc')]
    mask = xr.open_mfdataset(maskflist, combine='nested', concat_dim='time')['mask']

    rundown = 10e5
    shoreline = np.zeros(mask[:,:,0].shape)
    [nt, ny] = shoreline.shape
    for i in range(ny):
        for j in range(ny):
            if len(np.where(mask[i,j,:]==0)[0])==0:
                land_ind = -1
            else:
                land_ind = np.where(mask[i,j,:]==0)[0][0] # pull first instance of land mask
            shoreline[i,j] = x.values[land_ind]
            rundown = np.min((rundown, x.values[land_ind]))
    xsl = np.mean(shoreline)-22

    return Hs_alongmean[xind], Hs_alongmean[xsz_ind], xsz, xsl, rundown-22, Tp_off, Tp_sz, np.mean(dirspread_off), np.mean(dirspread_sz), np.mean(theta_off), np.mean(theta_sz)

rundir = 'hmo25_dir1_tp2'
rootdir = os.path.join('/gscratch', 'nearshore','enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

eta = xr.open_mfdataset(os.path.join(fdir, 'eta_*.nc'), combine='nested', concat_dim='time')['eta']
print(np.mean(eta[:,:,330].values))

Hs_off1, Hs_sz1, xsz1, xsl1, xr1, Tp_off1, Tp_sz1, dir_off1, dir_sz1, theta_off1, theta_sz1 = compute_wave_stats(fdir)

print('%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (Hs_off1, Hs_sz1, xsz1, xsl1, xr1, Tp_off1, Tp_sz1, dir_off1, dir_sz1, theta_off1, theta_sz1))

rundir = 'hmo25_dir5_tp2_ntheta15'
rootdir = os.path.join('/gscratch', 'nearshore','enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

eta = xr.open_mfdataset(os.path.join(fdir, 'eta_*.nc'), combine='nested', concat_dim='time')['eta']
print(np.mean(eta[:,:,330].values))

Hs_off5, Hs_sz5, xsz5, xsl5, xr5, Tp_off5, Tp_sz5, dir_off5, dir_sz5, theta_off5, theta_sz5 = compute_wave_stats(fdir)

print('%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (Hs_off5, Hs_sz5, xsz5, xsl5, xr5, Tp_off5, Tp_sz5, dir_off5, dir_sz5, theta_off5, theta_sz5))

rundir = 'hmo25_dir10_tp2'
rootdir = os.path.join('/gscratch', 'nearshore','enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

eta = xr.open_mfdataset(os.path.join(fdir, 'eta_*.nc'), combine='nested', concat_dim='time')['eta']
print(np.mean(eta[:,:,330].values))

Hs_off10, Hs_sz10, xsz10, xsl10, xr10, Tp_off10, Tp_sz10, dir_off10, dir_sz10, theta_off10, theta_sz10 = compute_wave_stats(fdir)

print('%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (Hs_off10, Hs_sz10, xsz10, xsl10, xr10, Tp_off10, Tp_sz10, dir_off10, dir_sz10, theta_off10, theta_sz10))

rundir = 'hmo25_dir20_tp2'
rootdir = os.path.join('/gscratch', 'nearshore','enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

eta = xr.open_mfdataset(os.path.join(fdir, 'eta_*.nc'), combine='nested', concat_dim='time')['eta']
print(np.mean(eta[:,:,330].values))

Hs_off20, Hs_sz20, xsz20, xsl20, xr20, Tp_off20, Tp_sz20, dir_off20, dir_sz20, theta_off20, theta_sz20 = compute_wave_stats(fdir)

print('%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (Hs_off20, Hs_sz20, xsz20, xsl20, xr20, Tp_off20, Tp_sz20, dir_off20, dir_sz20, theta_off20, theta_sz20))

rundir = 'hmo25_dir30_tp2'
rootdir = os.path.join('/gscratch', 'nearshore','enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

eta = xr.open_mfdataset(os.path.join(fdir, 'eta_*.nc'), combine='nested', concat_dim='time')['eta']
print(np.mean(eta[:,:,330].values))

Hs_off30, Hs_sz30, xsz30, xsl30, xr30, Tp_off30, Tp_sz30, dir_off30, dir_sz30, theta_off30, theta_sz30 = compute_wave_stats(fdir)

print('%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (Hs_off30, Hs_sz30, xsz30, xsl30, xr30, Tp_off30, Tp_sz30, dir_off30, dir_sz30, theta_off30, theta_sz30))

rundir = 'hmo25_dir40'
rootdir = os.path.join('/gscratch', 'nearshore','enuss','lab_server_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

eta = xr.open_mfdataset(os.path.join(fdir, 'eta_*.nc'), combine='nested', concat_dim='time')['eta']
print(np.mean(eta[:,:,330].values))

Hs_off40, Hs_sz40, xsz40, xsl40, xr40, Tp_off40, Tp_sz40, dir_off40, dir_sz40, theta_off40, theta_sz40 = compute_wave_stats(fdir)

print('%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (Hs_off40, Hs_sz40, xsz40, xsl40, xr40, Tp_off40, Tp_sz40, dir_off40, dir_sz40, theta_off40, theta_sz40))

rundir = 'hmo25_dir20_tp15'
rootdir = os.path.join('/gscratch', 'nearshore','enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

eta = xr.open_mfdataset(os.path.join(fdir, 'eta_*.nc'), combine='nested', concat_dim='time')['eta']
print(np.mean(eta[:,:,330].values))

Hs_off20_15, Hs_sz20_15, xsz20_15, xsl20_15, xr20_15, Tp_off20_15, Tp_sz20_15, dir_off20_15, dir_sz20_15, theta_off20_15, theta_sz20_15 = compute_wave_stats(fdir)

print('%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (Hs_off20_15, Hs_sz20_15, xsz20_15, xsl20_15, xr20_15, Tp_off20_15, Tp_sz20_15, dir_off20_15, dir_sz20_15, theta_off20_15, theta_sz20_15))

rundir = 'hmo25_dir20_tp25'
rootdir = os.path.join('/gscratch', 'nearshore','enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

eta = xr.open_mfdataset(os.path.join(fdir, 'eta_*.nc'), combine='nested', concat_dim='time')['eta']
print(np.mean(eta[:,:,330].values))

Hs_off20_25, Hs_sz20_25, xsz20_25, xsl20_25, xr20_25, Tp_off20_25, Tp_sz20_25, dir_off20_25, dir_sz20_25, theta_off20_25, theta_sz20_25 = compute_wave_stats(fdir)

print('%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (Hs_off20_25, Hs_sz20_25, xsz20_25, xsl20_25, xr20_25, Tp_off20_25, Tp_sz20_25, dir_off20_25, dir_sz20_25, theta_off20_25, theta_sz20_25))

