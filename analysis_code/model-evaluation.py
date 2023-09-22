import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import numpy.ma as ma
import funpy.model_utils as mod_utils
import funpy.obs_utils as obs_utils
import cmocean.cm as cmo 
import re
import glob
import pandas as pd
from scipy.signal import welch, detrend
from funpy import wave_functions as wf 
from matplotlib.lines import Line2D
from scipy.stats import chi2

#plt.ion()
plt.style.use('classic')
###### PLOTTING ON/OFF #######
plot_full = True 
plot_Hs = False 
plot_spec = False
###### DEFINE RUN & OBSERVATION INFO #######
dir10 = False
dir20 = True

WL = 512; OL = WL/2
T = 6000
Y = 55
DOF = (T*Y)/WL

probability = 0.95 # confidence interval, i.e., 95% is 0.95

alpha = 1 - probability
c = chi2.ppf([1 - alpha / 2, alpha / 2], DOF)
c2 = DOF / c # percentage representing confidence interval

### model output paths ###
rundir = 'hmo25_dir20_tp2'
rootdir = os.path.join('/data2','enuss','hyakbackup')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir)

eta_flist = [os.path.join(fdir, 'eta_1.nc'), os.path.join(fdir, 'eta_2.nc'), os.path.join(fdir, 'eta_3.nc'), os.path.join(fdir, 'eta_4.nc')]
mask_flist = [os.path.join(fdir, 'mask_1.nc'), os.path.join(fdir, 'mask_2.nc'), os.path.join(fdir, 'mask_3.nc'), os.path.join(fdir, 'mask_4.nc')]
u_flist = [os.path.join(fdir, 'u_1.nc'), os.path.join(fdir, 'u_2.nc'), os.path.join(fdir, 'u_3.nc'), os.path.join(fdir, 'u_4.nc')]
v_flist = [os.path.join(fdir, 'v_1.nc'), os.path.join(fdir, 'v_2.nc'), os.path.join(fdir, 'v_3.nc'), os.path.join(fdir, 'v_4.nc')]

color1='#003f5c'
color2='#444e86'
color3='#955196'
color4='#dd5182'
color5='#ff6e54'
color6='#ffa600'

### observation data paths ###
if dir10==True:
    rootdir_remote = os.path.join('/data2/baker/lab_experiments/data/processed/conditions')
    Hs = '25'
    Tp = '2'
    spread = '10'
    date = '09-06-2018-2342UTC'
    time = 'time_2357-0006'
    remote_dir = os.path.join('Hs%s_Tp%s_tide107_spread%s' % (Hs, Tp, spread))
    remote_path = os.path.join(rootdir_remote, remote_dir, date, time, 'model_data_comp')

    obsrootdir = os.path.join('/data1', 'TRC_Fall_2018_Experiment', 'PRJ-1873', 'data', 'inter')
    random_outer = 9
    trial_outer = 9
    random_inner = 5
    trail_inner = 5

if dir20==True:
    rootdir_remote = os.path.join('/data2/baker/lab_experiments/data/processed/conditions')
    Hs = '25'
    Tp = '2'
    spread = '20'
    date = '09-06-2018-2049UTC'
    time = 'time_2104-2113'
    remote_dir = os.path.join('Hs%s_Tp%s_tide107_spread%s' % (Hs, Tp, spread))
    remote_path = os.path.join(rootdir_remote, remote_dir, date, time, 'model_data_comp')

    obsrootdir = os.path.join('/data1', 'TRC_Fall_2018_Experiment', 'PRJ-1873', 'data', 'inter')
    random_outer = 9
    trial_outer = 6
    random_inner = 5
    trail_inner = 4

###### LOAD LAB BATHY ######
dat = np.loadtxt(os.path.join('/data2', 'enuss', 'TRC_cross-shore_profile.txt'), delimiter=',')
labx = dat[:,1]
labz = dat[:,0]-1.07

###### LOAD MODEL OUTPUT (eta, bathy) ######
### model time slice to compute over
tstart = int(5*60/0.2)
tint = int(10*60/0.2)
## frequency cutoffs for Hs calculation
fmin = 0.25
fmax = 1.2

eta, x, y = mod_utils.load_masked_variable('eta', eta_flist, 'mask', mask_flist)
dep = np.loadtxt(os.path.join(fdir, 'dep.out'))
eta = eta[tstart:tstart+tint,:,:]

eta_freq, eta_spec = mod_utils.compute_spec(eta, dt=0.2, WL=WL, OL=OL, n=1, axis=0)
del eta 

Hs = mod_utils.compute_Hsig_spectrally(eta_freq, eta_spec, fmin=fmin, fmax=fmax)
Hs_alongmean = np.nanmean(Hs, axis=0)
Hs_alongstd = np.nanstd(Hs, axis=0)

x = mod_utils.model2lab(x)

### load camera data ###
cam_Hs = pd.read_csv(os.path.join(remote_path, 'cam_Hs.txt')).values
cam_Tp = pd.read_csv(os.path.join(remote_path, 'cam_Tp.txt')).values
cam_xp = np.loadtxt(os.path.join(remote_path, 'cam_xp.txt'), delimiter=',')
cam_yp = np.loadtxt(os.path.join(remote_path, 'cam_yp.txt'), delimiter=',')
[cam_xx, cam_yy] = np.meshgrid(cam_xp, cam_yp)

#cam_Hs[cam_Hs > 0.26] = np.nan

### load lidar data ###
lidar_Hs = pd.read_csv(os.path.join(remote_path, 'lidar_Hs.txt')).values
lidar_Tp = pd.read_csv(os.path.join(remote_path, 'lidar_Tp.txt')).values
lidar_xp = np.loadtxt(os.path.join(remote_path, 'lidar_xp.txt'), delimiter=',')
lidar_yp = np.loadtxt(os.path.join(remote_path, 'lidar_yp.txt'), delimiter=',')

[lidar_xx, lidar_yy] = np.meshgrid(lidar_xp, lidar_yp)

### load pressure data
eta_outer, Hs_outer, u_outer, v_outer, xpos_outer, ypos_outer, dt_outer, dm_outer = obs_utils.load_array(obsrootdir, random_outer, trial_outer, WL=WL, OL=OL)
eta_inner, Hs_inner, u_inner, v_inner, xpos_inner, ypos_inner, dt_inner, dm_inner = obs_utils.load_array(obsrootdir, random_inner, trail_inner, WL=WL, OL=OL)

outer_break = 25
inner_break = 29
ind1, ind2, ind1_loc, ind2_loc = obs_utils.array_ind(xpos_outer, outer_break)
ind3, ind4, ind3_loc, ind4_loc = obs_utils.array_ind(xpos_inner, inner_break)

#remove 3
#ind2 = ind2[ind2!=3]

## 
#Hs_outer[6] = np.nan
Hs_outer[Hs_outer>0.22] = np.nan
Hs_outer[Hs_outer<0.1] = np.nan
Hs_inner[Hs_inner>0.3] = np.nan
Hs_inner[Hs_inner<0.1] = np.nan

valid_outer = np.where(np.isfinite(eta_outer[0,:])==True)[0]
eta_outer = eta_outer[:,valid_outer]

valid_inner = np.where(np.isfinite(eta_inner[0,:])==True)[0]
eta_inner = eta_inner[:,valid_inner]

Hs_wg, xpos_wg, ypos_wg, dt_wg = obs_utils.load_wg(obsrootdir, random_outer, trial_outer, WL=WL, OL=OL)


###### COMPUTE SPECTRA COMPARISON ######

### compute SSE and velocity spectra (model) ###
### SSE ###
mod_ind1 = np.argmin(np.abs(x-ind1_loc))
mod_ind2 = np.argmin(np.abs(x-ind2_loc))
mod_ind3 = np.argmin(np.abs(x-ind3_loc))
mod_ind4 = np.argmin(np.abs(x-ind4_loc))

mod_eta_spec1 = np.mean(eta_spec[:,:,mod_ind1], axis=-1)
mod_eta_spec2 = np.mean(eta_spec[:,:,mod_ind2], axis=-1)
mod_eta_spec3 = np.mean(eta_spec[:,:,mod_ind3], axis=-1)
mod_eta_spec4 = np.mean(eta_spec[:,:,mod_ind4], axis=-1)

### velocity ###
u, x, y = mod_utils.load_masked_variable('u', u_flist, 'mask', mask_flist)

mod_u1 = u[:,:,mod_ind1]
mod_u2 = u[:,:,mod_ind2]
mod_u3 = u[:,:,mod_ind3]
mod_u4 = u[:,:,mod_ind4]
del u 

vfile = os.path.join(rootdir, 'compiled_output_' + rundir, 'v_*.nc')
v, x, y = mod_utils.load_masked_variable('v', v_flist, 'mask', mask_flist)
x = mod_utils.model2lab(x)

mod_v1 = v[:,:,mod_ind1]
mod_v2 = v[:,:,mod_ind2]
mod_v3 = v[:,:,mod_ind3]
mod_v4 = v[:,:,mod_ind4]
del v

mod_freq, mod_u_spec1 = mod_utils.compute_spec(mod_u1[tstart:tstart+tint,:], WL=WL, OL=OL, dt=0.2, n=1, axis=0)
mod_freq, mod_u_spec2 = mod_utils.compute_spec(mod_u2[tstart:tstart+tint,:], WL=WL, OL=OL, dt=0.2, n=1, axis=0)
mod_freq, mod_u_spec3 = mod_utils.compute_spec(mod_u3[tstart:tstart+tint,:], WL=WL, OL=OL, dt=0.2, n=1, axis=0)
mod_freq, mod_u_spec4 = mod_utils.compute_spec(mod_u4[tstart:tstart+tint,:], WL=WL, OL=OL, dt=0.2, n=1, axis=0)

mod_freq, mod_v_spec1 = mod_utils.compute_spec(mod_v1[tstart:tstart+tint,:], WL=WL, OL=OL, dt=0.2, n=1, axis=0)
mod_freq, mod_v_spec2 = mod_utils.compute_spec(mod_v2[tstart:tstart+tint,:], WL=WL, OL=OL, dt=0.2, n=1, axis=0)
mod_freq, mod_v_spec3 = mod_utils.compute_spec(mod_v3[tstart:tstart+tint,:], WL=WL, OL=OL, dt=0.2, n=1, axis=0)
mod_freq, mod_v_spec4 = mod_utils.compute_spec(mod_v4[tstart:tstart+tint,:], WL=WL, OL=OL, dt=0.2, n=1, axis=0)

mod_u_spec1 = np.mean(mod_u_spec1, axis=1)
mod_u_spec2 = np.mean(mod_u_spec2, axis=1)
mod_u_spec3 = np.mean(mod_u_spec3, axis=1)
mod_u_spec4 = np.mean(mod_u_spec4, axis=1)

mod_v_spec1 = np.mean(mod_v_spec1, axis=1)
mod_v_spec2 = np.mean(mod_v_spec2, axis=1)
mod_v_spec3 = np.mean(mod_v_spec3, axis=1)
mod_v_spec4 = np.mean(mod_v_spec4, axis=1)

### compute SSE and velocity spectra (observations) ###
### SSE ###
freq_outer, eta_outer_spec = mod_utils.compute_spec(eta_outer, dt_outer, WL=WL, OL=OL, n=20, axis=1)
freq_inner, eta_inner_spec = mod_utils.compute_spec(eta_inner, dt_inner, WL=WL, OL=OL, n=20, axis=1)

# obs DOF
DOFobs = (len(eta_outer.T)*len(eta_outer))/(WL*20)
cobs = chi2.ppf([1 - alpha / 2, alpha / 2], DOFobs)
cobs2 = DOFobs / c # percentage representing confidence interval

def transfer_function(spec, freq, d):
    tf = np.zeros(spec.shape)
    for i in range(len(spec)):
        k = wf.wavenum(2*np.pi*freq, d[i])
        tf[i,:] = (np.cosh(k*dm_outer[i])/np.cosh(k*(d[i])))**-2
    # set transfer function to last value where wavenum converged
    freq_ind = np.where(freq>1)[0]
    tf[:,freq_ind] = np.expand_dims(tf[:,freq_ind[0]],1)*np.ones((len(tf), len(freq_ind)))
    return tf

## depth attenuation correction for SSE spectra from in situ gages 
d = np.asarray([-labz[np.argmin(np.abs(labx-loc))] for loc in xpos_outer])
tf_outer = transfer_function(eta_outer_spec, freq_outer, d) 

d = np.asarray([-labz[np.argmin(np.abs(labx-loc))] for loc in xpos_inner])
tf_inner = transfer_function(eta_inner_spec, freq_inner, d)

eta_outer_spec = eta_outer_spec*tf_outer
eta_inner_spec = eta_inner_spec*tf_inner

ind2 = np.array([ 1,  2, 6,  7,  8,  9, 10, 11]) # remove 3 due to bad sensor
ind3 = np.array([ 0,  4,  5,  7,  8,  9, 10]) # remove 6 due to bad sensor

eta_spec1 = np.nanmean(eta_outer_spec[ind1,:], axis=0)
eta_spec2 = np.nanmean(eta_outer_spec[ind2,:], axis=0)
eta_spec3 = np.nanmean(eta_inner_spec[ind3,:], axis=0)
eta_spec4 = np.nanmean(eta_inner_spec[ind4,:], axis=0)

### velocity ###
freq_outer, u_outer_spec = mod_utils.compute_spec(u_outer, dt_outer, WL=WL, OL=OL, n=20, axis=1)
freq_outer, v_outer_spec = mod_utils.compute_spec(v_outer, dt_outer, WL=WL, OL=OL, n=20, axis=1)
freq_inner, u_inner_spec = mod_utils.compute_spec(u_inner, dt_outer, WL=WL, OL=OL, n=20, axis=1)
freq_inner, v_inner_spec = mod_utils.compute_spec(v_inner, dt_outer, WL=WL, OL=OL, n=20, axis=1)

u_spec1 = np.nanmean(u_outer_spec[ind1,:], axis=0)
u_spec2 = np.nanmean(u_outer_spec[ind2,:], axis=0)
u_spec3 = np.nanmean(u_inner_spec[ind3,:], axis=0)
u_spec4 = np.nanmean(u_inner_spec[ind4,:], axis=0)

v_spec1 = np.nanmean(v_outer_spec[ind1,:], axis=0)
v_spec2 = np.nanmean(v_outer_spec[ind2,:], axis=0)
v_spec3 = np.nanmean(v_inner_spec[ind3,:], axis=0)
v_spec4 = np.nanmean(v_inner_spec[ind4,:], axis=0)


def SSE_spec_plot(eta_spec, u_spec, v_spec, mod_eta_spec, mod_u_spec, mod_v_spec, labz, labx, ind_loc, freq, mod_freq):
    d = -labz[np.argmin(np.abs(labx-ind_loc))]
    k = wf.wavenum(2*np.pi*freq, d)
    hor_vel_obs = (2*np.pi*freq*np.cosh(k*d)/np.sinh(k*d))**2
    hor_vel_obs[0] = 9.8/d ## set shallow water assumption for nan val freq = 0
    k = wf.wavenum(2*np.pi*mod_freq, d)
    hor_vel_mod = (2*np.pi*mod_freq*np.cosh(k*d)/np.sinh(k*d))**2
    hor_vel_mod[0] = 9.8/d ## set shallow water assumption for nan val freq = 0
    return eta_spec*hor_vel_obs, u_spec, v_spec, mod_eta_spec*hor_vel_mod, mod_u_spec, mod_v_spec

if 1:
    dm = 0.05
    fig = plt.figure(figsize=(20,13), facecolor="white")
    spec = fig.add_gridspec(5,4)
    ax0 = fig.add_subplot(spec[:3,:])
  
    color = 'tab:grey'
    ax0.plot(x, Hs_alongmean, color=color)
    ax0.plot(x, Hs_alongmean - Hs_alongstd, color=color, alpha=0.8)
    ax0.plot(x, Hs_alongmean + Hs_alongstd, color=color, alpha=0.8)
    ax0.fill_between(x, Hs_alongmean - Hs_alongstd, Hs_alongmean + Hs_alongstd, color=color, alpha=0.3, label='Model')

    ind = np.where(cam_xp<34.5)[0]
    ax0.plot(cam_xp[ind], np.nanmean(cam_Hs[:,ind], axis=0), color=color3)
    ax0.plot(cam_xp[ind], np.nanmean(cam_Hs[:,ind], axis=0) - np.nanstd(cam_Hs[:,ind], axis=0), color=color3, alpha=0.6)
    ax0.plot(cam_xp[ind], np.nanmean(cam_Hs[:,ind], axis=0) + np.nanstd(cam_Hs[:,ind], axis=0), color=color3, alpha=0.6)
    ax0.fill_between(cam_xp[ind], np.nanmean(cam_Hs[:,ind], axis=0) - np.nanstd(cam_Hs[:,ind], axis=0), np.nanmean(cam_Hs[:,ind], axis=0) + np.nanstd(cam_Hs[:,ind], axis=0), color=color3, alpha=0.3, label='Stereo Reconstruction')

    ax0.plot(lidar_xp, np.nanmean(lidar_Hs, axis=0), color=color1)
    ax0.plot(lidar_xp, np.nanmean(lidar_Hs, axis=0) - np.nanstd(lidar_Hs, axis=0), color=color1, alpha=0.6)
    ax0.plot(lidar_xp, np.nanmean(lidar_Hs, axis=0) + np.nanstd(lidar_Hs, axis=0), color=color1, alpha=0.6)
    ax0.fill_between(lidar_xp, np.nanmean(lidar_Hs, axis=0) - np.nanstd(lidar_Hs, axis=0), np.nanmean(lidar_Hs, axis=0) + np.nanstd(lidar_Hs, axis=0), color=color1, alpha=0.3, label='LiDAR')

    msize = 15
    ax0.plot(xpos_outer, Hs_outer, '^', markersize=msize, color='black', alpha=0.5)
    ax0.plot(xpos_inner, Hs_inner, '^', markersize=msize, color='black', alpha=0.5)   
    ax0.plot(xpos_wg, Hs_wg, '^', markersize=msize, color='black', alpha=0.5)

    ax0.set_ylabel(r'$H_s$ $(\mathrm{m})$', fontsize=20)
    ax0.set_xlim((18,35))
    ax0.set_ylim((-0.2, 0.3))
    #ax0.legend(loc='lower left', fontsize=14)

    ax0.text(18.2, 0.28, r'$\mathrm{(a)}$', fontsize='20')   
    ax0.tick_params(axis='x', which='major', labelsize=16)
    ax0.tick_params(axis='y', which='major', labelsize=16) 

    custom_lines = [Line2D([0], [0], color=color, lw=6, alpha=0.3), 
                    Line2D([0], [0], color=color3, lw=6, alpha=0.3), 
                    Line2D([0], [0], color=color1, lw=6, alpha=0.3), 
                    Line2D([0], [0], linestyle='None', marker='^', markersize=msize, color='black', alpha=0.5)]

    ax0.legend(custom_lines, [r'$\mathrm{Model}$', r'$\mathrm{Stereo\ reconstruction}$', r'$\mathrm{LiDAR}$', r'$\mathrm{In\ situ\ sensors}$'], fontsize=14, loc='upper right')


    ax_1 = ax0.twinx()
    ax_1.plot(labx, labz, color='lightgrey', linewidth=5, label=r'$\mathrm{Lab\ Bathymetry}$')
    ax_1.plot(x, -dep[0,:], '--', color='black', linewidth=3, label=r'$\mathrm{Model\ Bathymetry}$')
    ax_1.fill_between(x, -dep[0,:], np.ones(len(dep[0,:]))*(-1.2), color='lightgrey')
    ax_1.set_ylim((-1.2, 2.2))
    ax_1.legend(loc='lower right')
    ax_1.axis('off')
    ax0.set_xlabel(r'$x (\mathrm{m})$', fontsize=20)
    ax0.grid(True)  

    eta_var, u_var, v_var, mod_eta_var, mod_u_var, mod_v_var = SSE_spec_plot(eta_spec1, u_spec1, v_spec1, mod_eta_spec1, mod_u_spec1, mod_v_spec1, labz, labx, ind1_loc, freq_outer, mod_freq)

    alph = 0.8; ymin = 0.5
    ax2 = fig.add_subplot(spec[3:,0])   
    ax2.loglog(freq_outer, eta_var, '--', label=r'S$_{\eta\eta} * \sigma^2 \frac{\cosh^2(kh)}{\sinh^2(kh)}$  (obs)', color=color3, linewidth=2, alpha=alph)
    ax2.loglog(freq_outer, u_var + v_var, label='S$_{uu}$ + S$_{vv}$ (obs)', color=color3, linewidth=2, alpha=alph)
    ax2.loglog(mod_freq, mod_eta_var, '--', label=r'S$_{\eta\eta} * \sigma^2 \frac{\cosh^2(kh)}{\sinh^2(kh)}$ (mod)', color=color, linewidth=2, alpha=alph)
    ax2.loglog(mod_freq, mod_u_var+mod_v_var, label='S$_{uu}$ + S$_{vv}$ (mod)', color=color, linewidth=2, alpha=alph)
    ax2.set_ylabel(r'$\mathrm{Energy}\ (\mathrm{m}^2\ \mathrm{s}^{-3})$', fontsize=20)
    ax2.set_xlabel(r'$\mathrm{Frequency}\ (\mathrm{s}^{-1})$', fontsize=20)
    ax2.grid(True)
    ax2.set_title(r'$x = %.01f\ \mathrm{m}$' % ind1_loc, fontsize=20)
    ax2.set_xlim((0.0095, 2))
    ax2.set_ylim((10**-4, 1))   
    ax2.text(0.01, 0.56, r'$\mathrm{(b)}$', fontsize='20')    
    ax2.plot(np.array([1.1,1.1]), np.array([ymin, ymin+(c2[1]-c2[0])]), marker='_', color=color, linewidth=1.5)
    ax2.plot(np.array([1.2,1.2]), np.array([ymin+(c2[1]-c2[0])/2-(cobs2[1]-cobs2[0])/2, ymin+(c2[1]-c2[0])/2-(cobs2[1]-cobs2[0])/2+(cobs2[1]-cobs2[0])]), marker='_', color=color3, linewidth=1.5)
    ax2.tick_params(axis='x', which='major', labelsize=16)
    ax2.tick_params(axis='y', which='major', labelsize=16) 

    eta_var, u_var, v_var, mod_eta_var, mod_u_var, mod_v_var = SSE_spec_plot(eta_spec2, u_spec2, v_spec2, mod_eta_spec2, mod_u_spec2, mod_v_spec2, labz, labx, ind2_loc, freq_outer, mod_freq)

    ax3 = fig.add_subplot(spec[3:,1])   
    ax3.loglog(freq_outer, eta_var, '--', label=r'S$_{\eta\eta} * \sigma^2 \frac{\cosh^2(kh)}{\sinh^2(kh)}$  (obs)', color=color3, linewidth=2, alpha=alph)
    ax3.loglog(freq_outer, u_var + v_var, label='S$_{uu}$ + S$_{vv}$ (obs)', color=color3, linewidth=2, alpha=alph)
    ax3.loglog(mod_freq, mod_eta_var, '--', label=r'S$_{\eta\eta} * \sigma^2 \frac{\cosh^2(kh)}{\sinh^2(kh)}$ (mod)', color=color, linewidth=2, alpha=alph)
    ax3.loglog(mod_freq, mod_u_var+mod_v_var, label='S$_{uu}$ + S$_{vv}$ (mod)', color=color, linewidth=2, alpha=alph)
    ax3.set_ylabel(r'$\mathrm{Energy}\ (\mathrm{m}^2\ \mathrm{s}^{-3})$', fontsize=20)
    ax3.set_xlabel(r'$\mathrm{Frequency}\ (\mathrm{s}^{-1})$', fontsize=20)
    ax3.grid(True)
    ax3.set_title(r'$x = %.01f\ \mathrm{m}$' % ind2_loc, fontsize=20)
    ax3.set_xlim((0.0095, 2))
    ax3.set_ylim((10**-4, 1))   
    ax3.text(0.01, 0.56, r'$\mathrm{(c)}$', fontsize='20')    
    ax3.plot(np.array([1.1,1.1]), np.array([ymin, ymin+(c2[1]-c2[0])]), marker='_', color=color, linewidth=1.5)
    ax3.plot(np.array([1.2,1.2]), np.array([ymin+(c2[1]-c2[0])/2-(cobs2[1]-cobs2[0])/2, ymin+(c2[1]-c2[0])/2-(cobs2[1]-cobs2[0])/2+(cobs2[1]-cobs2[0])]), marker='_', color=color3, linewidth=1.5)
    ax3.tick_params(axis='x', which='major', labelsize=16)
    ax3.tick_params(axis='y', which='major', labelsize=16) 

    eta_var, u_var, v_var, mod_eta_var, mod_u_var, mod_v_var = SSE_spec_plot(eta_spec3, u_spec3, v_spec3, mod_eta_spec3, mod_u_spec3, mod_v_spec3, labz, labx, ind3_loc, freq_outer, mod_freq)

    ax4 = fig.add_subplot(spec[3:,2])   
    ax4.loglog(freq_outer, eta_var, '--', label=r'S$_{\eta\eta} * \sigma^2 \frac{\cosh^2(kh)}{\sinh^2(kh)}$  (obs)', color=color3, linewidth=2, alpha=alph)
    ax4.loglog(freq_outer, u_var + v_var, label='S$_{uu}$ + S$_{vv}$ (obs)', color=color3, linewidth=2, alpha=alph)
    ax4.loglog(mod_freq, mod_eta_var, '--', label=r'S$_{\eta\eta} * \sigma^2 \frac{\cosh^2(kh)}{\sinh^2(kh)}$ (mod)', color=color, linewidth=2, alpha=alph)
    ax4.loglog(mod_freq, mod_u_var+mod_v_var, label='S$_{uu}$ + S$_{vv}$ (mod)', color=color, linewidth=2, alpha=alph)
    ax4.set_ylabel(r'$\mathrm{Energy}\ (\mathrm{m}^2\ \mathrm{s}^{-3})$', fontsize=20)
    ax4.set_xlabel(r'$\mathrm{Frequency}\ (\mathrm{s}^{-1})$', fontsize=20)
    ax4.grid(True)
    ax4.set_title(r'$x = %.01f\ \mathrm{m}$' % ind3_loc, fontsize=20)
    ax4.set_xlim((0.0095, 2))
    ax4.set_ylim((10**-4, 1))   
    ax4.text(0.01, 0.56, r'$\mathrm{(d)}$', fontsize='20')    
    ax4.plot(np.array([1.1,1.1]), np.array([ymin, ymin+(c2[1]-c2[0])]), marker='_', color=color, linewidth=1.5)
    ax4.plot(np.array([1.2,1.2]), np.array([ymin+(c2[1]-c2[0])/2-(cobs2[1]-cobs2[0])/2, ymin+(c2[1]-c2[0])/2-(cobs2[1]-cobs2[0])/2+(cobs2[1]-cobs2[0])]), marker='_', color=color3, linewidth=1.5)
    ax4.tick_params(axis='x', which='major', labelsize=16)
    ax4.tick_params(axis='y', which='major', labelsize=16) 

    eta_var, u_var, v_var, mod_eta_var, mod_u_var, mod_v_var = SSE_spec_plot(eta_spec4, u_spec4, v_spec4, mod_eta_spec4, mod_u_spec4, mod_v_spec4, labz, labx, ind4_loc, freq_outer, mod_freq)

    ax5 = fig.add_subplot(spec[3:,3])   
    ax5.loglog(freq_outer, eta_var, '--', label=r'S$_{\eta\eta} * \sigma^2 \frac{\cosh^2(kh)}{\sinh^2(kh)}$  (obs)', color=color3, linewidth=2, alpha=alph)
    ax5.loglog(freq_outer, u_var + v_var, label='S$_{uu}$ + S$_{vv}$ (obs)', color=color3, linewidth=2, alpha=alph)
    ax5.loglog(mod_freq, mod_eta_var, '--', label=r'S$_{\eta\eta} * \sigma^2 \frac{\cosh^2(kh)}{\sinh^2(kh)}$ (mod)', color=color, linewidth=2, alpha=alph)
    ax5.loglog(mod_freq, mod_u_var+mod_v_var, label='S$_{uu}$ + S$_{vv}$ (mod)', color=color, linewidth=2, alpha=alph)
    ax5.set_ylabel(r'$\mathrm{Energy}\ (\mathrm{m}^2\ \mathrm{s}^{-3})$', fontsize=20)
    ax5.set_xlabel(r'$\mathrm{Frequency}\ (\mathrm{s}^{-1})$', fontsize=20)
    ax5.grid(True)
    ax5.set_title(r'$x = %.01f\ \mathrm{m}$' % ind4_loc, fontsize=20)
    ax5.set_xlim((0.0095, 2))
    ax5.set_ylim((10**-4, 1))  
    ax5.text(0.01, 0.56, r'$\mathrm{(e)}$', fontsize='20')    
    ax5.plot(np.array([1.1,1.1]), np.array([ymin, ymin+(c2[1]-c2[0])]), marker='_', color=color, linewidth=1.5)
    ax5.plot(np.array([1.2,1.2]), np.array([ymin+(c2[1]-c2[0])/2-(cobs2[1]-cobs2[0])/2, ymin+(c2[1]-c2[0])/2-(cobs2[1]-cobs2[0])/2+(cobs2[1]-cobs2[0])]), marker='_', color=color3, linewidth=1.5)
    ax5.tick_params(axis='x', which='major', labelsize=16)
    ax5.tick_params(axis='y', which='major', labelsize=16) 

    custom_lines = [Line2D([0], [0], color=color, lw=2), 
                    Line2D([0], [0], color=color3, lw=2), 
                    Line2D([0], [0], linestyle='--', color='black', lw=2), 
                    Line2D([0], [0], color='black', lw=2)]

    ax5.legend(custom_lines, [r'$\mathrm{Model}$', r'$\mathrm{Observations}$', r'$S_{\eta\eta} * (\omega \frac{\cosh(kh)}{\sinh(kh)})^2$', 'S$_{uu}$ + S$_{vv}$'], fontsize=14, loc='lower left')

    fig.tight_layout()
    fig.savefig(os.path.join(savedir, 'model_lab_comparison_Hs_spec.png'))



if 1:
    import cv2
    import mat73
    image = cv2.imread('/data2/enuss/c2_07379.tiff')
    grey_image = np.sum(image, axis=-1)
    lidar_dict = mat73.loadmat('/data2/enuss/lidar_footprint.mat')
    lidar_foot = lidar_dict['extent']
    lidar_x = lidar_dict['x']
    lidar_y = lidar_dict['y']
    lidarxx, lidaryy = np.meshgrid(lidar_x, lidar_y)
    stereo_dict = mat73.loadmat('/data2/enuss/stereo_footprint.mat')
    stereo_foot = stereo_dict['extent']
    stereo_x = stereo_dict['x']
    stereo_y = stereo_dict['y']


    fig = plt.figure(figsize=(12,10), facecolor="white")
    spec = fig.add_gridspec(3,5)  

    ax0 = fig.add_subplot(spec[0,:])
    ax0.plot(labx, -labz, color='lightgrey', linewidth=5, label=r'$\mathrm{Lab\ Bathymetry}$')
    ax0.plot(x, dep[0,:], '--', color='black', linewidth=3, label=r'$\mathrm{Model\ Bathymetry}$')
    ax0.fill_between(x, dep[0,:], -np.ones(len(dep[0,:]))*(-1.2), color='lightgrey')
    ax0.plot(x[x<32.5], np.zeros(len(x[x<32.5])), color='#346888')
    ax0.legend(loc='upper right', fontsize=16)
    ax0.set_xlabel(r'$x\ (\mathrm{m})$', fontsize=16)
    ax0.axvline(x[0]+20, color='#004c6d', linewidth=8, label='Wavemaker')
    ax0.axvspan(x[0], x[0]+15, color='#7aa6c2', alpha=0.5, label='Sponge Layer')        
    ax0.grid(True) 
    ax0.set_ylim((-1.2, 1.2))
    ax0.set_xlim(-22, 40)    
    ax0.invert_yaxis()         
    ax0.text(38, 1.1, r'$\mathrm{(a)}$', fontweight='bold', fontsize=18)
    ax0.text(-20, -0.9, 'Sponge', fontweight='bold', fontsize=16)
    ax0.text(-20, -0.7, 'Layer', fontweight='bold', fontsize=16)
    ax0.text(-1.6, 0.5, 'Wavemaker', fontweight='bold', fontsize=16, rotation='vertical')
    ax0.text(10.1, -0.05, 'Still Water', fontweight='bold', fontsize=16)
    ax0.set_ylabel(r'$\mathrm{Bathymetry\ (m)}$', fontsize=18)
    ax0.tick_params(axis='x', which='major', labelsize=16)
    ax0.tick_params(axis='y', which='major', labelsize=16) 

    ax1 = fig.add_subplot(spec[1:,:3])
    y_lab = y - 55/2
    [xx, yy] = np.meshgrid(x, y_lab)    
    p1 = ax1.pcolormesh(xx, yy, dep, cmap=cmo.deep)
    cb1 = fig.colorbar(p1, ax=ax1)
    cb1.set_label(label=r'$\mathrm{Model\ Bathymetry\ (m)}$', size=18)
    ax1.set_xlim(-22,40)
    ax1.set_ylim(-55/2, 55/2)
    ax1.set_ylabel(r'$y\ (\mathrm{m})$', fontsize=18)
    ax1.set_xlabel(r'$x\ (\mathrm{m})$', fontsize=18)
    ax1.text(-21, -27, r'$\mathrm{(b)}$', color='white', fontweight='bold', fontsize=20)  
    SZymin = 0.5; SZymax = SZymin + 26.5; SZxmin = 18.35; SZxmax = 35
    ax1.plot(np.array([SZxmin, SZxmax]), np.array([SZymin, SZymin]), '--', linewidth=3, color='white')
    ax1.plot(np.array([SZxmin, SZxmax]), np.array([SZymax, SZymax]), '--', linewidth=3, color='white')
    ax1.plot(np.array([SZxmin, SZxmin]), np.array([SZymin, SZymax]), '--', linewidth=3, color='white')
    ax1.plot(np.array([SZxmax, SZxmax]), np.array([SZymin, SZymax]), '--', linewidth=3, color='white')
    SZymin = -55/2+0.5; SZymax = SZymin + 26.5; SZxmin = 18.35; SZxmax = 35
    ax1.plot(np.array([SZxmin, SZxmax]), np.array([SZymin, SZymin]), '--', linewidth=3, color='white')
    ax1.plot(np.array([SZxmin, SZxmax]), np.array([SZymax, SZymax]), '--', linewidth=3, color='white')
    ax1.plot(np.array([SZxmin, SZxmin]), np.array([SZymin, SZymax]), '--', linewidth=3, color='white')
    ax1.plot(np.array([SZxmax, SZxmax]), np.array([SZymin, SZymax]), '--', linewidth=3, color='white')
    ax1.set_aspect('equal', 'box')
    ax1.tick_params(axis='x', which='major', labelsize=16)
    ax1.tick_params(axis='y', which='major', labelsize=16) 

    ax2 = fig.add_subplot(spec[1:,3:])
    y_lab = y[y<=26.5] - 26.5/2
    [xx, yy] = np.meshgrid(x, y_lab)
    p2 = ax2.imshow(image, extent=[18,35,-14,14])
    ax2.pcolormesh(lidarxx, lidaryy, lidar_foot, cmap='Greys_r', alpha=0.3)
    ax2.axvline(25, linestyle='--', linewidth=3, color='white')
    ax2.set_xlim(18.35,35)
    ax2.set_ylim(-26.5/2, 26.5/2)
    msize = 300
    lsize = 2
    ax2.scatter(xpos_inner, ypos_inner, s=msize, marker='o', color='white', linewidths=lsize, edgecolors='black', label=r'$\mathrm{Surf\ zone\ array}$')
    ax2.scatter(xpos_outer, ypos_outer, s=msize, marker='s', color='white', linewidths=lsize, edgecolors='black', label=r'$\mathrm{Inner\ shelf\ array}$')
    ax2.scatter(xpos_wg, ypos_wg, s=msize, marker='^', color='white', linewidths=lsize, edgecolors='black', label=r'$\mathrm{Offshore\ array}$')
    ax2.legend(loc='upper right', scatterpoints=1)
    ax2.set_ylabel(r'$y\ (\mathrm{m})$', fontsize=18)
    ax2.set_xlabel(r'$x\ (\mathrm{m})$', fontsize=18)
    ax2.text(18.5, -13, r'$\mathrm{(c)}$', color='white', fontweight='bold', fontsize=20)  
    ax2.set_aspect('equal', 'box')
    ax2.tick_params(axis='x', which='major', labelsize=16)
    ax2.tick_params(axis='y', which='major', labelsize=16) 

    fig.tight_layout()
    fig.savefig(os.path.join(savedir, 'model_lab_bathy_setup.png'))




