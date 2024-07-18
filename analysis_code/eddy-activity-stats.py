import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import numpy.ma as ma
import pandas as pd 
import funpy.model_utils as mod_utils
import funpy.postprocess as fp 
import cmocean.cm as cmo 
import re
import glob
from scipy.signal import welch
from funpy import filter_functions as ff
import datetime
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
from matplotlib.lines import Line2D
plt.ion()
#plt.style.use('ggplot')
plt.style.use('classic')

dx = 0.05; dy = 0.1; dt = 0.2
n = 1; WL = 512; OL = WL/2 

color1='#003f5c'
color2='#444e86'
color3='#955196'
color4='#dd5182'
color5='#ff6e54'
color6='#ffa600'
color = 'tab:grey'

plotsavedir = os.path.join('/gscratch', 'nearshore', 'enuss', 'lab_runs_y550', 'postprocessing', 'plots')
dirspread = np.array([0.3, 3.8, 11.2, 16.5, 21.6, 24.9])
sz = 23.2 + 22

def compute_uex(u, dy, yaxis=1):
	Ly = u.shape[yaxis]*dy
	uex = np.sum(u, axis=yaxis)*dy/Ly
	return uex

def move_avg(signal, T, dt):
	N = int(T/dt)
	return np.convolve(signal, np.ones(N)/N, mode='same')[int(N/2):-int(N/2)]

######################################################################################
rundir = 'hmo25_dir1_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

dep = np.loadtxt(os.path.join(fdir, 'dep.out'))

uflist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
vflist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]

u_psi_dat = xr.open_mfdataset(uflist, combine='nested', concat_dim='time')
u_psi = u_psi_dat['u_psi']
x = np.asarray(u_psi_dat.x) 
y = np.asarray(u_psi_dat.y)
[xx, yy] = np.meshgrid(x,y)

freq1, u_spec1 = welch(u_psi-np.expand_dims(np.mean(u_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean1 = np.mean(u_spec1, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq1, v_spec1 = welch(v_psi-np.expand_dims(np.mean(v_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean1 = np.mean(v_spec1, axis=1)

xind = np.argmin(np.abs(x-sz))

u_psi = ma.masked_where(u_psi>=0, u_psi)

u_ex1 = compute_uex(u_psi[:,:,xind], dy) 
u_ex_avg1 = np.mean(u_ex1, axis=0)
u_ex_cross1 = np.mean(compute_uex(u_psi, dy, yaxis=1), axis=0)

###############################################################
rundir = 'hmo25_dir5_tp2_ntheta15'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

uflist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
vflist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]

u_psi_dat = xr.open_mfdataset(uflist, combine='nested', concat_dim='time')
u_psi = u_psi_dat['u_psi']

freq5, u_spec5 = welch(u_psi-np.expand_dims(np.mean(u_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean5 = np.mean(u_spec5, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq5, v_spec5 = welch(v_psi-np.expand_dims(np.mean(v_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean5 = np.mean(v_spec5, axis=1)

xind = np.argmin(np.abs(x-sz))

u_psi = ma.masked_where(u_psi>=0, u_psi)

u_ex5 = compute_uex(u_psi[:,:,xind], dy) 
u_ex_avg5 = np.mean(u_ex5, axis=0)
u_ex_cross5 = np.mean(compute_uex(u_psi, dy, yaxis=1), axis=0)

###############################################################
rundir = 'hmo25_dir10_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

uflist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
vflist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]

u_psi_dat = xr.open_mfdataset(uflist, combine='nested', concat_dim='time')
u_psi = u_psi_dat['u_psi']

freq10, u_spec10 = welch(u_psi-np.expand_dims(np.mean(u_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean10 = np.mean(u_spec10, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq10, v_spec10 = welch(v_psi-np.expand_dims(np.mean(v_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean10 = np.mean(v_spec10, axis=1)

xind = np.argmin(np.abs(x-sz))

u_psi = ma.masked_where(u_psi>=0, u_psi)

u_ex10 = compute_uex(u_psi[:,:,xind], dy) 
u_ex_avg10 = np.mean(u_ex10, axis=0)
u_ex_cross10 = np.mean(compute_uex(u_psi, dy, yaxis=1), axis=0)

###############################################################
rundir = 'hmo25_dir20_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

uflist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
vflist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]

u_psi_dat = xr.open_mfdataset(uflist, combine='nested', concat_dim='time')
u_psi = u_psi_dat['u_psi']

freq20, u_spec20 = welch(u_psi-np.expand_dims(np.mean(u_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean20 = np.mean(u_spec20, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq20, v_spec20 = welch(v_psi-np.expand_dims(np.mean(v_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean20 = np.mean(v_spec20, axis=1)

xind = np.argmin(np.abs(x-sz))

u_psi = ma.masked_where(u_psi>=0, u_psi)

u_ex20 = compute_uex(u_psi[:,:,xind], dy) 
u_ex_avg20 = np.mean(u_ex20, axis=0)
u_ex_cross20 = np.mean(compute_uex(u_psi, dy, yaxis=1), axis=0)

###############################################################
rundir = 'hmo25_dir30_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

uflist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
vflist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]

u_psi_dat = xr.open_mfdataset(uflist, combine='nested', concat_dim='time')
u_psi = u_psi_dat['u_psi']

freq30, u_spec30 = welch(u_psi-np.expand_dims(np.mean(u_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean30 = np.mean(u_spec30, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq30, v_spec30 = welch(v_psi-np.expand_dims(np.mean(v_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean30 = np.mean(v_spec30, axis=1)

xind = np.argmin(np.abs(x-sz))

u_psi = ma.masked_where(u_psi>=0, u_psi)

u_ex30 = compute_uex(u_psi[:,:,xind], dy) 
u_ex_avg30 = np.mean(u_ex30, axis=0)
u_ex_cross30 = np.mean(compute_uex(u_psi, dy, yaxis=1), axis=0)

###############################################################
rundir = 'hmo25_dir40_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

uflist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
vflist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]

u_psi_dat = xr.open_mfdataset(uflist, combine='nested', concat_dim='time')
u_psi = u_psi_dat['u_psi']

freq40, u_spec40 = welch(u_psi-np.expand_dims(np.mean(u_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean40 = np.mean(u_spec40, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq40, v_spec40 = welch(v_psi-np.expand_dims(np.mean(v_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean40 = np.mean(v_spec40, axis=1)

xind = np.argmin(np.abs(x-sz))

u_psi = ma.masked_where(u_psi>=0, u_psi)

u_ex40 = compute_uex(u_psi[:,:,xind], dy) 
u_ex_avg40 = np.mean(u_ex40, axis=0)
u_ex_cross40 = np.mean(compute_uex(u_psi, dy, yaxis=1), axis=0)

###############################################################
rundir = 'hmo25_dir20_tp15'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

uflist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
vflist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]

u_psi_dat = xr.open_mfdataset(uflist, combine='nested', concat_dim='time')
u_psi = u_psi_dat['u_psi']

freq20_15, u_spec20_15 = welch(u_psi-np.expand_dims(np.mean(u_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean20_15 = np.mean(u_spec20_15, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq20_15, v_spec20_15 = welch(v_psi-np.expand_dims(np.mean(v_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean20_15 = np.mean(v_spec20_15, axis=1)

xind = np.argmin(np.abs(x-sz))

u_psi = ma.masked_where(u_psi>=0, u_psi)

u_ex20_15 = compute_uex(u_psi[:,:,xind], dy) 
u_ex_avg20_15 = np.mean(u_ex20_15, axis=0)
u_ex_cross20_15 = np.mean(compute_uex(u_psi, dy, yaxis=1), axis=0)

###############################################################
rundir = 'hmo25_dir20_tp25'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

uflist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
vflist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]

u_psi_dat = xr.open_mfdataset(uflist, combine='nested', concat_dim='time')
u_psi = u_psi_dat['u_psi']

freq20_25, u_spec20_25 = welch(u_psi-np.expand_dims(np.mean(u_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean20_25 = np.mean(u_spec20_25, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq20_25, v_spec20_25 = welch(v_psi-np.expand_dims(np.mean(v_psi, axis=0), axis=0), fs=1/dt, detrend='constant', window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean20_25 = np.mean(v_spec20_25, axis=1)

xind = np.argmin(np.abs(x-sz))

u_psi = ma.masked_where(u_psi>=0, u_psi)

u_ex20_25 = compute_uex(u_psi[:,:,xind], dy) 
u_ex_avg20_25 = np.mean(u_ex20_25, axis=0)
u_ex_cross20_25 = np.mean(compute_uex(u_psi, dy, yaxis=1), axis=0)

########################################
sz = 31.5 - 23.2

xloc1 = 31.5+22
xloc2 = (31.5 - sz*0.5)+22 
xloc3 = (31.5 - sz*1)+22
xloc4 = (31.5 - sz*1.5+22)

xind1 = np.where((x<xloc1) & (x>=xloc2))[0]
xind2 = np.where((x<xloc2) & (x>=xloc3))[0]
xind3 = np.where((x<xloc3) & (x>=xloc4))[0]
sz_xind = np.argmin(np.abs(x-23.5-22))

freqind_lf = np.where((freq1>0.003)&(freq1<0.2))[0]
freqind_vlf = np.where((freq1>0.003)&(freq1<0.02))[0]

vel1 = u_spec1 + v_spec1 
vel5 = u_spec5 + v_spec5 
vel10 = u_spec10 + v_spec10
vel20 = u_spec20 + v_spec20 
vel30 = u_spec30 + v_spec30 
vel40 = u_spec40 + v_spec40 
vel20_15 = u_spec20_15 + v_spec20_15
vel20_25 = u_spec20_25 + v_spec20_25

lf_vel1 = np.sum(vel1[freqind_lf, :, :], axis=0)*(freq1[1]-freq1[0])
lf_vel5 = np.sum(vel5[freqind_lf, :, :], axis=0)*(freq1[1]-freq1[0])
lf_vel10 = np.sum(vel10[freqind_lf, :, :], axis=0)*(freq1[1]-freq1[0])
lf_vel20 = np.sum(vel20[freqind_lf, :, :], axis=0)*(freq1[1]-freq1[0])
lf_vel30 = np.sum(vel30[freqind_lf, :, :], axis=0)*(freq1[1]-freq1[0])
lf_vel40 = np.sum(vel40[freqind_lf, :, :], axis=0)*(freq1[1]-freq1[0])
lf_vel20_15 = np.sum(vel20_15[freqind_lf,:,:], axis=0)*(freq1[1]-freq1[0])
lf_vel20_25 = np.sum(vel20_25[freqind_lf,:,:], axis=0)*(freq1[1]-freq1[0])

vlf_vel1 = np.sum(vel1[freqind_vlf, :, :], axis=0)*(freq1[1]-freq1[0])
vlf_vel5 = np.sum(vel5[freqind_vlf, :, :], axis=0)*(freq1[1]-freq1[0])
vlf_vel10 = np.sum(vel10[freqind_vlf, :, :], axis=0)*(freq1[1]-freq1[0])
vlf_vel20 = np.sum(vel20[freqind_vlf, :, :], axis=0)*(freq1[1]-freq1[0])
vlf_vel30 = np.sum(vel30[freqind_vlf, :, :], axis=0)*(freq1[1]-freq1[0])
vlf_vel40 = np.sum(vel40[freqind_vlf, :, :], axis=0)*(freq1[1]-freq1[0])
vlf_vel20_15 = np.sum(vel20_15[freqind_vlf,:,:], axis=0)*(freq1[1]-freq1[0])
vlf_vel20_25 = np.sum(vel20_25[freqind_vlf,:,:], axis=0)*(freq1[1]-freq1[0])

##### FIGURES #####
labdirspread = np.array([2.4, 9.6, 18.3, 24.2, 25.9])
labvlf_x245 = np.array([0.0001076, 0.00027463, 0.0002926, 0.00030265, 0.00018583])
labvlf_std_x245 = np.array([3.16E-06, 4.28E-05, 3.02E-05, 1.60E-05, 8.04E-05])
labvlf_x266 = np.array([0.00041068, 0.00085144, 0.00086143, 0.00084166, 0.00080124])
labvlf_std_x266 = np.array([0.00039378, 0.00023135, 0.00018897, 0.00027543, 0.00022364])
labvlf_x284 = np.array([0.00119555, 0.00336905, 0.00466043, 0.00512779, 0.00379607])
labvlf_std_x284 = np.array([0.00016174, 0.00055601, 0.00048737, 0.00048762, 0.00101546])
labvlf_x307 = np.array([0.00112287, 0.00350467, 0.00468539, 0.0039609, 0.0036955])
labvlf_std_x307 = np.array([0.00020207, 0.00029536, 0.0007438, 0.00030158, 0.00044867])
labvlf_sz_avg = (labvlf_x266 + labvlf_x284 + labvlf_x307)/3
labvlf_sz_std_avg = (labvlf_std_x266 + labvlf_std_x284 + labvlf_std_x307)/3
labvlf_tp = np.array([2,3])
labvlf_sz_dir20 = np.array([np.mean((0.0046, 0.0043)), np.mean((0.0037, 0.0042))])

lf_sz = np.array([np.mean(lf_vel1[:,np.concatenate((xind2, xind1))]), np.mean(lf_vel5[:,np.concatenate((xind2, xind1))]), np.mean(lf_vel10[:,np.concatenate((xind2, xind1))]),
				  np.mean(lf_vel20[:,np.concatenate((xind2, xind1))]), np.mean(lf_vel30[:,np.concatenate((xind2, xind1))]), np.mean(lf_vel40[:,np.concatenate((xind2, xind1))])])

lf_sz_std = np.array([np.mean(np.std(lf_vel1[:,np.concatenate((xind2, xind1))], axis=1)), np.mean(np.std(lf_vel5[:,np.concatenate((xind2, xind1))], axis=1)), np.mean(np.std(lf_vel10[:,np.concatenate((xind2, xind1))], axis=1)),
				  np.mean(np.std(lf_vel20[:,np.concatenate((xind2, xind1))], axis=1)), np.mean(np.std(lf_vel30[:,np.concatenate((xind2, xind1))], axis=1)), np.mean(np.std(lf_vel40[:,np.concatenate((xind2, xind1))], axis=1))])

lf_sz_tp = np.array([np.mean(lf_vel20_15[:,np.concatenate((xind2, xind1))]), np.mean(lf_vel20[:,np.concatenate((xind2, xind1))]), np.mean(lf_vel20_25[:,np.concatenate((xind2, xind1))])])

lf_sz_tp_std = np.array([np.mean(np.std(lf_vel20_15[:,np.concatenate((xind2, xind1))], axis=1)), np.mean(np.std(lf_vel20[:,np.concatenate((xind2, xind1))], axis=1)), 
							   np.mean(np.std(lf_vel20_25[:,np.concatenate((xind2, xind1))], axis=1))])

lf_sz_inner = np.array([np.mean(lf_vel1[:,xind1]), np.mean(lf_vel5[:,xind1]), np.mean(lf_vel10[:,xind1]),
						np.mean(lf_vel20[:,xind1]), np.mean(lf_vel30[:,xind1]), np.mean(lf_vel40[:,xind1])])

lf_sz_inner_std = np.array([np.mean(np.std(lf_vel1[:,xind1], axis=1)), np.mean(np.std(lf_vel5[:,xind1], axis=1)), 
							np.mean(np.std(lf_vel10[:,xind1], axis=1)), np.mean(np.std(lf_vel20[:,xind1], axis=1)), 
							np.mean(np.std(lf_vel30[:,xind1], axis=1)), np.mean(np.std(lf_vel40[:,xind1], axis=1))])

lf_sz_inner_tp = np.array([np.mean(lf_vel20_15[:,xind1]), np.mean(lf_vel20[:,xind1]), np.mean(lf_vel20_25[:,xind1])])

lf_sz_inner_tp_std = np.array([np.mean(np.std(lf_vel20_15[:,xind1], axis=1)), np.mean(np.std(lf_vel20[:,xind1], axis=1)), 
							   np.mean(np.std(lf_vel20_25[:,xind1], axis=1))])

lf_sz_outer = np.array([np.mean(lf_vel1[:,xind2]), np.mean(lf_vel5[:,xind2]), np.mean(lf_vel10[:,xind2]),
						np.mean(lf_vel20[:,xind2]), np.mean(lf_vel30[:,xind2]), np.mean(lf_vel40[:,xind2])])

lf_sz_outer_std = np.array([np.mean(np.std(lf_vel1[:,xind2], axis=1)), np.mean(np.std(lf_vel5[:,xind2], axis=1)), 
							np.mean(np.std(lf_vel10[:,xind2], axis=1)), np.mean(np.std(lf_vel20[:,xind2], axis=1)), 
							np.mean(np.std(lf_vel30[:,xind2], axis=1)), np.mean(np.std(lf_vel40[:,xind2], axis=1))])

lf_sz_outer_tp = np.array([np.mean(lf_vel20_15[:,xind2]), np.mean(lf_vel20[:,xind2]), np.mean(lf_vel20_25[:,xind2])])

lf_sz_outer_tp_std = np.array([np.mean(np.std(lf_vel20_15[:,xind2], axis=1)), np.mean(np.std(lf_vel20[:,xind2], axis=1)), 
							   np.mean(np.std(lf_vel20_25[:,xind2], axis=1))])

lf_sz_offshore = np.array([np.mean(lf_vel1[:,xind3]), np.mean(lf_vel5[:,xind3]), np.mean(lf_vel10[:,xind3]),
						np.mean(lf_vel20[:,xind3]), np.mean(lf_vel30[:,xind3]), np.mean(lf_vel40[:,xind3])])

lf_sz_offshore_std = np.array([np.mean(np.std(lf_vel1[:,xind3], axis=1)), np.mean(np.std(lf_vel5[:,xind3], axis=1)), 
							   np.mean(np.std(lf_vel10[:,xind3], axis=1)), np.mean(np.std(lf_vel20[:,xind3], axis=1)), 
							   np.mean(np.std(lf_vel30[:,xind3], axis=1)), np.mean(np.std(lf_vel40[:,xind3], axis=1))])

lf_sz_offshore_tp = np.array([np.mean(lf_vel20_15[:,xind3]), np.mean(lf_vel20[:,xind3]), np.mean(lf_vel20_25[:,xind3])])

lf_sz_offshore_tp_std = np.array([np.mean(np.std(lf_vel20_15[:,xind3], axis=1)), np.mean(np.std(lf_vel20[:,xind3], axis=1)), 
								  np.mean(np.std(lf_vel20_25[:,xind3], axis=1))])

vlf_sz = np.array([np.mean(vlf_vel1[:,np.concatenate((xind2, xind1))]), np.mean(vlf_vel5[:,np.concatenate((xind2, xind1))]), np.mean(vlf_vel10[:,np.concatenate((xind2, xind1))]),
				  np.mean(vlf_vel20[:,np.concatenate((xind2, xind1))]), np.mean(vlf_vel30[:,np.concatenate((xind2, xind1))]), np.mean(vlf_vel40[:,np.concatenate((xind2, xind1))])])

vlf_sz_std = np.array([np.mean(np.std(vlf_vel1[:,np.concatenate((xind2, xind1))], axis=1)), np.mean(np.std(vlf_vel5[:,np.concatenate((xind2, xind1))], axis=1)), np.mean(np.std(vlf_vel10[:,np.concatenate((xind2, xind1))], axis=1)),
				  np.mean(np.std(vlf_vel20[:,np.concatenate((xind2, xind1))], axis=1)), np.mean(np.std(vlf_vel30[:,np.concatenate((xind2, xind1))], axis=1)), np.mean(np.std(vlf_vel40[:,np.concatenate((xind2, xind1))], axis=1))])

vlf_sz_tp = np.array([np.mean(vlf_vel20_15[:,np.concatenate((xind2, xind1))]), np.mean(vlf_vel20[:,np.concatenate((xind2, xind1))]), np.mean(vlf_vel20_25[:,np.concatenate((xind2, xind1))])])

vlf_sz_tp_std = np.array([np.mean(np.std(vlf_vel20_15[:,np.concatenate((xind2, xind1))], axis=1)), np.mean(np.std(vlf_vel20[:,np.concatenate((xind2, xind1))], axis=1)), 
							   np.mean(np.std(vlf_vel20_25[:,np.concatenate((xind2, xind1))], axis=1))])

vlf_sz_inner = np.array([np.mean(vlf_vel1[:,xind1]), np.mean(vlf_vel5[:,xind1]), np.mean(vlf_vel10[:,xind1]),
						np.mean(vlf_vel20[:,xind1]), np.mean(vlf_vel30[:,xind1]), np.mean(vlf_vel40[:,xind1])])

vlf_sz_inner_std = np.array([np.mean(np.std(vlf_vel1[:,xind1], axis=1)), np.mean(np.std(vlf_vel5[:,xind1], axis=1)), 
							 np.mean(np.std(vlf_vel10[:,xind1], axis=1)), np.mean(np.std(vlf_vel20[:,xind1], axis=1)), 
							 np.mean(np.std(vlf_vel30[:,xind1], axis=1)), np.mean(np.std(vlf_vel40[:,xind1], axis=1))])

vlf_sz_inner_tp = np.array([np.mean(vlf_vel20_15[:,xind1]), np.mean(vlf_vel20[:,xind1]), np.mean(vlf_vel20_25[:,xind1])])

vlf_sz_inner_tp_std = np.array([np.mean(np.std(vlf_vel20_15[:,xind1], axis=1)), np.mean(np.std(vlf_vel20[:,xind1], axis=1)), 
								np.mean(np.std(vlf_vel20_25[:,xind1], axis=1))])

vlf_sz_outer = np.array([np.mean(vlf_vel1[:,xind2]), np.mean(vlf_vel5[:,xind2]), np.mean(vlf_vel10[:,xind2]),
						np.mean(vlf_vel20[:,xind2]), np.mean(vlf_vel30[:,xind2]), np.mean(vlf_vel40[:,xind2])])

vlf_sz_outer_std = np.array([np.mean(np.std(vlf_vel1[:,xind2], axis=1)), np.mean(np.std(vlf_vel5[:,xind2], axis=1)), 
							 np.mean(np.std(vlf_vel10[:,xind2], axis=1)), np.mean(np.std(vlf_vel20[:,xind2], axis=1)), 
							 np.mean(np.std(vlf_vel30[:,xind2], axis=1)), np.mean(np.std(vlf_vel40[:,xind2], axis=1))])

vlf_sz_outer_tp = np.array([np.mean(vlf_vel20_15[:,xind2]), np.mean(vlf_vel20[:,xind2]), np.mean(vlf_vel20_25[:,xind2])])

vlf_sz_outer_tp_std = np.array([np.mean(np.std(vlf_vel20_15[:,xind2], axis=1)), np.mean(np.std(vlf_vel20[:,xind2], axis=1)), 
								np.mean(np.std(vlf_vel20_25[:,xind2], axis=1))])

vlf_sz_offshore = np.array([np.mean(vlf_vel1[:,xind3]), np.mean(vlf_vel5[:,xind3]), np.mean(vlf_vel10[:,xind3]),
						np.mean(vlf_vel20[:,xind3]), np.mean(vlf_vel30[:,xind3]), np.mean(vlf_vel40[:,xind3])])

vlf_sz_offshore_std = np.array([np.mean(np.std(vlf_vel1[:,xind3], axis=1)), np.mean(np.std(vlf_vel5[:,xind3], axis=1)), 
								np.mean(np.std(vlf_vel10[:,xind3], axis=1)), np.mean(np.std(vlf_vel20[:,xind3], axis=1)), 
								np.mean(np.std(vlf_vel30[:,xind3], axis=1)), np.mean(np.std(vlf_vel40[:,xind3], axis=1))])

vlf_sz_offshore_tp = np.array([np.mean(vlf_vel20_15[:,xind3]), np.mean(vlf_vel20[:,xind3]), np.mean(vlf_vel20_25[:,xind3])])

vlf_sz_offshore_tp_std = np.array([np.mean(np.std(vlf_vel20_15[:,xind3], axis=1)), np.mean(np.std(vlf_vel20[:,xind3], axis=1)), 
								   np.mean(np.std(vlf_vel20_25[:,xind3], axis=1))])

u_ex_sz_edge = -np.array([u_ex_cross1[sz_xind], u_ex_cross5[sz_xind], u_ex_cross10[sz_xind], 
						 u_ex_cross20[sz_xind], u_ex_cross30[sz_xind], u_ex_cross40[sz_xind]])

u_ex_sz_edge_std = np.array([np.std(u_ex1), np.std(u_ex5), np.std(u_ex10), np.std(u_ex20), np.std(u_ex30), np.std(u_ex40)])

u_ex_sz_edge_tp = -np.array([u_ex_cross20_15[sz_xind], u_ex_cross20[sz_xind], u_ex_cross20_25[sz_xind]])

u_ex_sz_edge_tp_std = np.array([np.std(u_ex20_15), np.std(u_ex20), np.std(u_ex20_25)])


#############################################

color = 'tab:grey'
color1 = '#264c5c'
color3 = '#13905a'
color6 = '#ffa600'


fig, ax = plt.subplots(ncols=6, figsize=(9.5,5.5))
lwidth = 1
msize = 5
fsize = 11

xmin = 0.08+0.15+0.015
ymin = 0.1
width = 0.2-0.15/3
height = 0.37
xoffset = 0.1
yoffset = 0.12

ax[0].errorbar(dirspread, vlf_sz, yerr=vlf_sz_std, fmt='-o', linewidth=lwidth, markersize=msize, color=color1, label=r'$\mathrm{Model}$')
ax[0].plot(labdirspread, labvlf_sz_avg, '-o', linewidth=lwidth, markersize=msize, color=color6, label=r'$\mathrm{Observations}$')
ax[0].legend(loc='upper right', fontsize=fsize, bbox_to_anchor=(-0.5, 1.03))
ax[0].set_ylabel(r'$V_{vlf}^\psi\ (\mathrm{m^2 s^{-1}})}$', fontsize=fsize)
ax[0].set_xlabel(r'$\sigma_\theta$ ($\degree$)', fontsize=fsize)
ax[0].set_position([xmin, ymin+yoffset+height, width, height])
ax[0].grid(True)
ax[0].text(0.5, 1.4*10**-2, r'$(a)$', fontsize=fsize+3)
ax[0].tick_params(axis='x', which='major', labelsize=fsize)
ax[0].tick_params(axis='y', which='major', labelsize=fsize) 
ax[0].set_xticks([0,5,10,15,20,25,30])
ax[0].set_yticks([0, 0.4*10**-2, 0.8*10**-2, 1.2*10**-2, 1.6*10**-2])
ax[0].set_ylim(0, 1.6*10**-2)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax[0].yaxis.set_major_formatter(formatter)
formatter.set_powerlimits((-1,1)) 

ax[1].errorbar(np.array([1.5, 2, 2.5]), vlf_sz_tp, yerr=vlf_sz_tp_std, fmt='-o', color=color1, linewidth=lwidth, markersize=msize)
ax[1].plot(labvlf_tp, labvlf_sz_dir20, '-o', linewidth=lwidth, markersize=msize+1, color=color6)
ax[1].grid(True)
ax[1].tick_params(axis='x', which='major', labelsize=fsize)
ax[1].tick_params(axis='y', which='major', labelsize=fsize) 
ax[1].set_xticks([1, 1.5, 2, 2.5, 3, 3.5])
ax[1].set_yticks([0, 0.4*10**-2, 0.8*10**-2, 1.2*10**-2, 1.6*10**-2])
ax[1].set_xlabel(r'$T_p$ $(\mathrm{s})$', fontsize=fsize)
ax[1].set_ylabel(r'$V_{vlf}^\psi\ (\mathrm{m^2 s^{-1}})}$', fontsize=fsize)
ax[1].set_position([xmin, ymin, width, height])
ax[1].text(1.1, 1.4*10**-2, r'$(d)$', fontsize=fsize+3)
ax[1].yaxis.set_major_formatter(formatter)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax[1].yaxis.set_major_formatter(formatter)
formatter.set_powerlimits((-1,1)) 
custom_lines = [Line2D([0], [0], linestyle='-', marker='o', markersize=msize, color=color6, lw=lwidth),
                Line2D([0], [0], linestyle='-', marker='o', markersize=msize, color=color1, lw=lwidth)]

ax[1].legend(custom_lines, [r'$\mathrm{Observations}$,' + '\n' + r'$\sigma_\theta = 16.4^\circ$' + '\n' + r'$H_s = 0.30\ \mathrm{m}$', r'$\mathrm{Model}$' + '\n' + r'$\sigma_\theta = 16.5^\circ$' + '\n' + r'$H_s = 0.25\ \mathrm{m}$'], fontsize=fsize, loc='upper right', bbox_to_anchor=(-0.5, 1.03))

ax[2].errorbar(dirspread, lf_sz, yerr=lf_sz_std, fmt='-o', linewidth=lwidth, markersize=msize, color=color1, label=r'$\mathrm{Inner\ Surf\ Zone}$')
ax[2].set_ylabel(r'$V_{lf}^\psi\ (\mathrm{m^2 s^{-1}})}$', fontsize=fsize)
ax[2].set_xlabel(r'$\sigma_\theta$ ($\degree$)', fontsize=fsize)
ax[2].grid(True)
ax[2].text(1.1, 1.4*10**-2, r'$(b)$', fontsize=fsize+3)
ax[2].tick_params(axis='x', which='major', labelsize=fsize)
ax[2].tick_params(axis='y', which='major', labelsize=fsize)
ax[2].set_xticks([0, 5, 10, 15, 20, 25, 30])
ax[2].set_yticks([0, 0.4*10**-2, 0.8*10**-2, 1.2*10**-2, 1.6*10**-2])
ax[2].set_ylim(0, 1.6*10**-2)
ax[2].set_position([xmin+xoffset+width, ymin+yoffset+height, width, height])
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax[2].yaxis.set_major_formatter(formatter)
formatter.set_powerlimits((-1,1)) 

ax[3].errorbar(np.array([1.5, 2, 2.5]), lf_sz_tp, yerr=lf_sz_tp_std, fmt='-o', color=color1, linewidth=lwidth, markersize=msize)
ax[3].grid(True)
ax[3].tick_params(axis='x', which='major', labelsize=fsize)
ax[3].tick_params(axis='y', which='major', labelsize=fsize) 
ax[3].set_xticks([1, 1.5, 2, 2.5, 3, 3.5])
ax[3].set_yticks([0, 0.4*10**-2, 0.8*10**-2, 1.2*10**-2, 1.6*10**-2])
ax[3].set_xlabel(r'$T_p$ $(\mathrm{s})$', fontsize=fsize)
ax[3].set_ylabel(r'$V_{lf}^\psi\ (\mathrm{m^2 s^{-1}})}$', fontsize=fsize)
ax[3].text(1.1, 1.4*10**-2, r'$(e)$', fontsize=fsize+3)
ax[3].set_position([xmin+xoffset+width, ymin, width, height])
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax[3].yaxis.set_major_formatter(formatter)
formatter.set_powerlimits((-1,1)) 

ax[4].errorbar(dirspread, u_ex_sz_edge, yerr=u_ex_sz_edge_std, fmt='-o', linewidth=lwidth, markersize=msize, color=color1)
ax[4].set_ylabel(r'$U_{ex}$ ($\mathrm{m s}^{-1}$)', fontsize=fsize)
ax[4].set_xlabel(r'$\sigma_\theta$ ($\degree$)', fontsize=fsize)
ax[4].grid(True)
ax[4].text(1.1, 3.5*10**-2, r'$(c)$', fontsize=fsize+3)
ax[4].tick_params(axis='x', which='major', labelsize=fsize)
ax[4].tick_params(axis='y', which='major', labelsize=fsize) 
ax[4].set_xticks([0,5,10,15,20,25,30])
ax[4].set_yticks([0, 1*10**-2, 2*10**-2, 3*10**-2, 4*10**-2])
ax[4].set_ylim(0, 4*10**-2)
ax[4].set_position([xmin+2*(xoffset+width), ymin+yoffset+height, width, height])
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax[4].yaxis.set_major_formatter(formatter)
formatter.set_powerlimits((-1,1)) 

ax[5].errorbar(np.array([1.5, 2, 2.5]), u_ex_sz_edge_tp, yerr=u_ex_sz_edge_tp_std, fmt='-o', color=color1, linewidth=lwidth, markersize=msize)
ax[5].grid(True)
ax[5].tick_params(axis='x', which='major', labelsize=fsize)
ax[5].tick_params(axis='y', which='major', labelsize=fsize) 
ax[5].set_xticks([1, 1.5, 2, 2.5, 3, 3.5])
ax[5].set_yticks([0, 1*10**-2, 2*10**-2, 3*10**-2, 4*10**-2])
ax[5].set_xlabel(r'$T_p$ $(\mathrm{s})$', fontsize=fsize)
ax[5].set_ylabel(r'$U_{ex}$ ($\mathrm{m s}^{-1}$)', fontsize=fsize)
ax[5].text(1.1, 3.5*10**-2, r'$(f)$', fontsize=fsize+3)
ax[5].set_position([xmin+2*(xoffset+width), ymin, width, height])
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax[5].yaxis.set_major_formatter(formatter)
formatter.set_powerlimits((-1,1)) 
fig.savefig(os.path.join(plotsavedir, 'lf_vlf_uex_revised.png'))
fig.savefig(os.path.join(plotsavedir, 'lf_vlf_uex_revised.jpg'))


################################################################
color1='#003f5c'
color2='#444e86'
color3='#955196'
color4='#dd5182'
color5='#ff6e54'
color6='#ffa600'
color = 'tab:grey'

xs_ind = np.argmin(np.abs(x-32.5-22))
fig, ax = plt.subplots(figsize=(8.5,3), ncols=2, sharex=True)
ax[0].plot(x[:xs_ind]-22, np.mean(lf_vel1[:,:xs_ind], axis=0), '-', linewidth=lwidth, markersize=msize, color=color1, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[0])
ax[0].plot(x[:xs_ind]-22, np.mean(lf_vel5[:,:xs_ind], axis=0), '-', linewidth=lwidth, markersize=msize, color=color2, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[1])
ax[0].plot(x[:xs_ind]-22, np.mean(lf_vel10[:,:xs_ind], axis=0), '-', linewidth=lwidth, markersize=msize, color=color3, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[2])
ax[0].plot(x[:xs_ind]-22, np.mean(lf_vel20[:,:xs_ind], axis=0), '-', linewidth=lwidth, markersize=msize, color=color4, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[3])
ax[0].plot(x[:xs_ind]-22, np.mean(lf_vel30[:,:xs_ind], axis=0), '-', linewidth=lwidth, markersize=msize, color=color5, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[4])
ax[0].plot(x[:xs_ind]-22, np.mean(lf_vel40[:,:xs_ind], axis=0), '-', linewidth=lwidth, markersize=msize, color=color6, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[5])
ax[0].legend(loc='best', fontsize=fsize, bbox_to_anchor=(-0.25, 0.95))
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax[0].yaxis.set_major_formatter(formatter)
formatter.set_powerlimits((-1,1)) 
ax[0].set_xlabel(r'$x\ (\mathrm{m})$', fontsize=fsize)
ax[0].set_ylabel(r'$V_{lf}^\psi$ $(\mathrm{m^2 s^{-1}})$', fontsize=fsize)
ax[0].tick_params(axis='x', which='major', labelsize=fsize)
ax[0].tick_params(axis='y', which='major', labelsize=fsize) 
ax[0].axvspan(31.5, 32.5, color=color, alpha=0.3)        
ax[0].text(16, 2.7*10**-2, r'$\mathrm{(a)}$', fontsize=fsize)
ax[0].axvline(xloc2-22, linestyle='--', ymin=0, ymax=1, color=color, linewidth=2, alpha=0.7)
ax[0].axvline(xloc3-22, linestyle='--', ymin=0, ymax=1, color=color, linewidth=2, alpha=0.7)
ax[0].axvline(xloc4-22, linestyle='--', ymin=0, ymax=1, color=color, linewidth=2, alpha=0.7)
ax[0].text(xloc2-22 + 0.9, 2.7*10**-2, r'$\mathrm{Inner}$', fontsize=fsize)
ax[0].text(xloc3-22 + 0.7, 2.7*10**-2, r'$\mathrm{Outer}$', fontsize=fsize)
ax[0].text(xloc4-22 + 0.2, 2.7*10**-2, r'$\mathrm{Offshore}$', fontsize=fsize)
ax[0].grid(which='major', axis='y')

ax[1].plot(x[:xs_ind]-22, -u_ex_cross1[:xs_ind], '-', linewidth=lwidth, markersize=msize, color=color1, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[0])
ax[1].plot(x[:xs_ind]-22, -u_ex_cross5[:xs_ind], '-', linewidth=lwidth, markersize=msize, color=color2, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[1])
ax[1].plot(x[:xs_ind]-22, -u_ex_cross10[:xs_ind], '-', linewidth=lwidth, markersize=msize, color=color3, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[2])
ax[1].plot(x[:xs_ind]-22, -u_ex_cross20[:xs_ind], '-', linewidth=lwidth, markersize=msize, color=color4, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[3])
ax[1].plot(x[:xs_ind]-22, -u_ex_cross30[:xs_ind], '-', linewidth=lwidth, markersize=msize, color=color5, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[4])
ax[1].plot(x[:xs_ind]-22, -u_ex_cross40[:xs_ind], '-', linewidth=lwidth, markersize=msize, color=color6, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[5])
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax[1].yaxis.set_major_formatter(formatter)
formatter.set_powerlimits((-1,1)) 
ax[1].set_xlabel(r'$x\ (\mathrm{m})$', fontsize=fsize)
ax[1].set_ylabel(r'$U_{ex}$ $\mathrm{(ms^{-1})}$', fontsize=fsize)
ax[1].grid(which='major', axis='y')
ax[1].tick_params(axis='x', which='major', labelsize=fsize)
ax[1].tick_params(axis='y', which='major', labelsize=fsize) 
ax[1].axvspan(31.5, 32.5, color=color, alpha=0.3)        
ax[1].text(16, 5.5*10**-2, r'$\mathrm{(b)}$', fontsize=fsize)
ax[1].set_xlim(15, 32.5)
ax[1].axvline(xloc2-22, linestyle='--', ymin=0, ymax=1, color=color, linewidth=2, alpha=0.7)
ax[1].axvline(xloc3-22, linestyle='--', ymin=0, ymax=1, color=color, linewidth=2, alpha=0.7)
ax[1].axvline(xloc4-22, linestyle='--', ymin=0, ymax=1, color=color, linewidth=2, alpha=0.7)
ax[1].text(xloc2-22 + 0.9, 5.5*10**-2, r'$\mathrm{Inner}$', fontsize=fsize)
ax[1].text(xloc3-22 + 0.7, 5.5*10**-2, r'$\mathrm{Outer}$', fontsize=fsize)
ax[1].text(xloc4-22 + 0.2, 5.5*10**-2, r'$\mathrm{Offshore}$', fontsize=fsize)
fig.tight_layout()
fig.savefig(os.path.join(plotsavedir, 'low_freq_u_ex_cross.png'))
fig.savefig(os.path.join(plotsavedir, 'low_freq_u_ex_cross.jpg'))
