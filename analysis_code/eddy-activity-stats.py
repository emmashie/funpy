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
dirspread = np.array([0.24,3.40,9.65,16.39,22.06,25.67])
sz = np.array([27.05, 27.55, 27.40, 27.40, 27.40, 27.40, 25.35, 27.95])+22

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

freq1, u_spec1 = welch(u_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean1 = np.mean(u_spec1, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq1, v_spec1 = welch(v_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean1 = np.mean(v_spec1, axis=1)

xind = np.argmin(np.abs(x-sz[0]))

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

freq5, u_spec5 = welch(u_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean5 = np.mean(u_spec5, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq5, v_spec5 = welch(v_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean5 = np.mean(v_spec5, axis=1)

xind = np.argmin(np.abs(x-sz[1]))

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

freq10, u_spec10 = welch(u_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean10 = np.mean(u_spec10, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq10, v_spec10 = welch(v_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean10 = np.mean(v_spec10, axis=1)

xind = np.argmin(np.abs(x-sz[2]))

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

freq20, u_spec20 = welch(u_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean20 = np.mean(u_spec20, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq20, v_spec20 = welch(v_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean20 = np.mean(v_spec20, axis=1)

xind = np.argmin(np.abs(x-sz[3]))

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

freq30, u_spec30 = welch(u_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean30 = np.mean(u_spec30, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq30, v_spec30 = welch(v_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean30 = np.mean(v_spec30, axis=1)

xind = np.argmin(np.abs(x-sz[4]))

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

freq40, u_spec40 = welch(u_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean40 = np.mean(u_spec40, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq40, v_spec40 = welch(v_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean40 = np.mean(v_spec40, axis=1)

xind = np.argmin(np.abs(x-sz[5]))

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

freq20_15, u_spec20_15 = welch(u_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean20_15 = np.mean(u_spec20_15, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq20_15, v_spec20_15 = welch(v_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean20 = np.mean(v_spec20, axis=1)

xind = np.argmin(np.abs(x-sz[3]))

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

freq20_25, u_spec20_25 = welch(u_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
u_spec_alongmean20_25 = np.mean(u_spec20_25, axis=1)

v_psi_dat = xr.open_mfdataset(vflist, combine='nested', concat_dim='time')
v_psi = v_psi_dat['v_psi']

freq20_25, v_spec20_25 = welch(v_psi, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
v_spec_alongmean20_25 = np.mean(v_spec20_25, axis=1)

xind = np.argmin(np.abs(x-sz[3]))

u_psi = ma.masked_where(u_psi>=0, u_psi)

u_ex20_25 = compute_uex(u_psi[:,:,xind], dy) 
u_ex_avg20_25 = np.mean(u_ex20_25, axis=0)
u_ex_cross20_25 = np.mean(compute_uex(u_psi, dy, yaxis=1), axis=0)

########################################

sz_ind1 = np.argmin(np.abs(x-sz[0]))
sz_ind5 = np.argmin(np.abs(x-sz[1]))
sz_ind10 = np.argmin(np.abs(x-sz[2]))
sz_ind20 = np.argmin(np.abs(x-sz[3]))
sz_ind30 = np.argmin(np.abs(x-sz[4]))
sz_ind40 = np.argmin(np.abs(x-sz[5]))
sz_ind20_15 = np.argmin(np.abs(x-sz[6]))
sz_ind20_25 = np.argmin(np.abs(x-sz[7]))

freqind_lf = np.where(freq1<0.2)[0]
freqind_vlf = np.where(freq1<0.02)[0]

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


vel_alongmean1 = np.mean(vel1, axis=1)
vel_alongmean5 = np.mean(vel5, axis=1)
vel_alongmean10 = np.mean(vel10, axis=1)
vel_alongmean20 = np.mean(vel20, axis=1)
vel_alongmean30 = np.mean(vel30, axis=1)
vel_alongmean40 = np.mean(vel40, axis=1)
vel_alongmean20_15 = np.mean(vel20_15, axis=1)
vel_alongmean20_25 = np.mean(vel20_25, axis=1)

##### FIGURES #####
lwidth = 3
msize = 10
fsize = 18

labdirspread = np.array([2.3285, 11.6920, 16.3781, 22.4739])
labvlf = np.array([1.7, 3.8, 4.9, 5.0])*10**-3

lf_sz_edge = np.array([np.sum(vel_alongmean1[freqind_lf,sz_ind1])*(freq1[1]-freq1[0]),
					   np.sum(vel_alongmean5[freqind_lf,sz_ind5])*(freq5[1]-freq5[0]),
					   np.sum(vel_alongmean10[freqind_lf,sz_ind10])*(freq10[1]-freq10[0]),
					   np.sum(vel_alongmean20[freqind_lf,sz_ind20])*(freq20[1]-freq20[0]),
					   np.sum(vel_alongmean30[freqind_lf,sz_ind30])*(freq30[1]-freq30[0]),
					   np.sum(vel_alongmean40[freqind_lf,sz_ind40])*(freq40[1]-freq40[0])])

vlf_sz_edge = np.array([np.sum(vel_alongmean1[freqind_vlf,sz_ind1])*(freq1[1]-freq1[0]),
					   np.sum(vel_alongmean5[freqind_vlf,sz_ind5])*(freq5[1]-freq5[0]),
					   np.sum(vel_alongmean10[freqind_vlf,sz_ind10])*(freq10[1]-freq10[0]),
					   np.sum(vel_alongmean20[freqind_vlf,sz_ind20])*(freq20[1]-freq20[0]),
					   np.sum(vel_alongmean30[freqind_vlf,sz_ind30])*(freq30[1]-freq30[0]),
					   np.sum(vel_alongmean40[freqind_vlf,sz_ind40])*(freq40[1]-freq40[0])])

u_ex_sz_edge = np.array([u_ex_cross1[sz_ind1], u_ex_cross5[sz_ind5], u_ex_cross10[sz_ind10], 
						 u_ex_cross20[sz_ind20], u_ex_cross30[sz_ind30], u_ex_cross40[sz_ind40]])

xind = 300
lf_inner = np.array([np.sum(vel_alongmean1[freqind_lf,xind])*(freq1[1]-freq1[0]),
					   np.sum(vel_alongmean5[freqind_lf,xind])*(freq5[1]-freq5[0]),
					   np.sum(vel_alongmean10[freqind_lf,xind])*(freq10[1]-freq10[0]),
					   np.sum(vel_alongmean20[freqind_lf,xind])*(freq20[1]-freq20[0]),
					   np.sum(vel_alongmean30[freqind_lf,xind])*(freq30[1]-freq30[0]),
					   np.sum(vel_alongmean40[freqind_lf,xind])*(freq40[1]-freq40[0])])

vlf_inner = np.array([np.sum(vel_alongmean1[freqind_vlf,xind])*(freq1[1]-freq1[0]),
					   np.sum(vel_alongmean5[freqind_vlf,xind])*(freq5[1]-freq5[0]),
					   np.sum(vel_alongmean10[freqind_vlf,xind])*(freq10[1]-freq10[0]),
					   np.sum(vel_alongmean20[freqind_vlf,xind])*(freq20[1]-freq20[0]),
					   np.sum(vel_alongmean30[freqind_vlf,xind])*(freq30[1]-freq30[0]),
					   np.sum(vel_alongmean40[freqind_vlf,xind])*(freq40[1]-freq40[0])])

u_ex_inner = np.array([u_ex_cross1[xind], u_ex_cross5[xind], u_ex_cross10[xind], 
						 u_ex_cross20[xind], u_ex_cross30[xind], u_ex_cross40[xind]])

xind = 200
lf_x25 = np.array([np.sum(vel_alongmean1[freqind_lf,xind])*(freq1[1]-freq1[0]),
					   np.sum(vel_alongmean5[freqind_lf,xind])*(freq5[1]-freq5[0]),
					   np.sum(vel_alongmean10[freqind_lf,xind])*(freq10[1]-freq10[0]),
					   np.sum(vel_alongmean20[freqind_lf,xind])*(freq20[1]-freq20[0]),
					   np.sum(vel_alongmean30[freqind_lf,xind])*(freq30[1]-freq30[0]),
					   np.sum(vel_alongmean40[freqind_lf,xind])*(freq40[1]-freq40[0])])

vlf_x25 = np.array([np.sum(vel_alongmean1[freqind_vlf,xind])*(freq1[1]-freq1[0]),
					   np.sum(vel_alongmean5[freqind_vlf,xind])*(freq5[1]-freq5[0]),
					   np.sum(vel_alongmean10[freqind_vlf,xind])*(freq10[1]-freq10[0]),
					   np.sum(vel_alongmean20[freqind_vlf,xind])*(freq20[1]-freq20[0]),
					   np.sum(vel_alongmean30[freqind_vlf,xind])*(freq30[1]-freq30[0]),
					   np.sum(vel_alongmean40[freqind_vlf,xind])*(freq40[1]-freq40[0])])

u_ex_x25 = np.array([u_ex_cross1[xind], u_ex_cross5[xind], u_ex_cross10[xind], 
						 u_ex_cross20[xind], u_ex_cross30[xind], u_ex_cross40[xind]])

color7 = '#004c6d'; color8 = '#6996b3'; color9 = '#c1e7ff'

fsize = 20
fig = plt.figure(figsize=(20,15), facecolor="white")
spec = fig.add_gridspec(2,6)

ax0 = fig.add_subplot(spec[0,:2])
#ax0.plot(dirspread, vlf_inner, '-^', linewidth=lwidth, markersize=msize, color=color7, label='x = 30 m')
ax0.plot(dirspread, vlf_sz_edge, '-^', linewidth=lwidth, markersize=msize, color='tab:grey', label=r'$\mathrm{Model}$')
ax0.plot(dirspread[3], np.sum(vel_alongmean20_15[freqind_vlf, sz_ind20_15])*(freq1[1]-freq1[0]), 'v', color=color6, markersize=msize, label=r'$T_p = 1.5\ s$')
ax0.plot(dirspread[3], np.sum(vel_alongmean20_25[freqind_vlf, sz_ind20_25])*(freq1[1]-freq1[0]), '^', color=color6, markersize=msize, label=r'$T_p = 2.5\ s$')
#ax0.plot(dirspread, vlf_x25, '-^', linewidth=lwidth, markersize=msize, color=color9, label='x = 25 m')
ax0.plot(labdirspread, labvlf, '-o', linewidth=lwidth, markersize=msize, color=color3, alpha=0.8, label=r'$\mathrm{Observations}$')
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax0.yaxis.set_major_formatter(formatter)
formatter.set_powerlimits((-1,1)) 
ax0.set_xlabel(r'$\sigma_\theta$ $(\degree)$', fontsize=fsize)
ax0.set_ylabel(r'$V_{vlf}$ $(\mathrm{m^2s^{-1}})$', fontsize=fsize)
ax0.grid(True)
ax0.tick_params(axis='x', which='major', labelsize=fsize)
ax0.tick_params(axis='y', which='major', labelsize=fsize) 
ax0.text(2, 1.3*10**-2, r'$\mathrm{(a)}$', fontsize=fsize)
ax0.set_ylim(0,1.4*10**-2)
ax0.legend(loc='best')

ax1 = fig.add_subplot(spec[0,2:4])
#ax1.plot(dirspread, lf_inner, '-^', linewidth=lwidth, markersize=msize, color=color7, label='x = 30 m')
ax1.plot(dirspread, lf_sz_edge, '-^', linewidth=lwidth, markersize=msize, color='tab:grey', label='x = 27.5 m')
#ax1.plot(dirspread, lf_x25, '-^', linewidth=lwidth, markersize=msize, color=color9, label='x = 25 m')
ax1.plot(dirspread[3], np.sum(vel_alongmean20_15[freqind_lf, sz_ind20_15])*(freq1[1]-freq1[0]), 'v', color=color6, markersize=msize, label=r'$T_p = 1.5\ s$')
ax1.plot(dirspread[3], np.sum(vel_alongmean20_25[freqind_lf, sz_ind20_25])*(freq1[1]-freq1[0]), '^', color=color6, markersize=msize, label=r'$T_p = 2.5\ s$')
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax1.yaxis.set_major_formatter(formatter)
formatter.set_powerlimits((-1,1)) 
ax1.set_xlabel(r'$\sigma_\theta$ $(\degree)$', fontsize=fsize)
ax1.set_ylabel(r'$V_{lf}$ $(\mathrm{m^2s^{-1}})$', fontsize=fsize)
ax1.grid(True)
ax1.tick_params(axis='x', which='major', labelsize=fsize)
ax1.tick_params(axis='y', which='major', labelsize=fsize) 
ax1.text(2, 1.3*10**-2, r'$\mathrm{(b)}$', fontsize=fsize)
ax1.set_ylim(0,1.4*10**-2)

ax2 = fig.add_subplot(spec[0,4:])
ax2.plot(dirspread, -u_ex_sz_edge, '-^', linewidth=lwidth, markersize=msize, color='tab:grey')
#ax2.plot(dirspread, -u_ex_inner, '-^', linewidth=lwidth, markersize=msize, color=color8)
#ax2.plot(dirspread, -u_ex_x25, '-^', linewidth=lwidth, markersize=msize, color=color9)
ax2.plot(dirspread[3], -u_ex_avg20_15, 'v', color=color6, markersize=msize, label=r'$T_p = 1.5\ s$')
ax2.plot(dirspread[3], -u_ex_avg20_25, '^', color=color6, markersize=msize, label=r'$T_p = 2.5\ s$')
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax2.yaxis.set_major_formatter(formatter)
formatter.set_powerlimits((-1,1)) 
ax2.set_xlabel(r'$\sigma_\theta$ $(\degree)$', fontsize=fsize)
ax2.set_ylabel(r'$U_{ex}$ $(\overline{U_{ex}})$ $\mathrm{(ms^{-1})}$', fontsize=fsize)
ax2.grid(True)
ax2.set_ylim(0,0.048)
ax2.tick_params(axis='x', which='major', labelsize=fsize)
ax2.tick_params(axis='y', which='major', labelsize=fsize) 
ax2.text(2, 4.5*10**-2, r'$\mathrm{(c)}$', fontsize=fsize)

ax3 = fig.add_subplot(spec[1:,:3])
ax3.plot(x-22, np.mean(lf_vel1, axis=0), '-', linewidth=lwidth, markersize=msize, color=color1, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[0])
ax3.plot(x-22, np.mean(lf_vel5, axis=0), '-', linewidth=lwidth, markersize=msize, color=color2, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[1])
ax3.plot(x-22, np.mean(lf_vel10, axis=0), '-', linewidth=lwidth, markersize=msize, color=color3, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[2])
ax3.plot(x-22, np.mean(lf_vel20, axis=0), '-', linewidth=lwidth, markersize=msize, color=color4, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[3])
ax3.plot(x-22, np.mean(lf_vel30, axis=0), '-', linewidth=lwidth, markersize=msize, color=color5, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[4])
ax3.plot(x-22, np.mean(lf_vel40, axis=0), '-', linewidth=lwidth, markersize=msize, color=color6, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[5])
#ax3.axvline(30, linestyle='--', linewidth=3, color=color7)
ax3.axvline(27.5, linestyle='--', linewidth=3, color='tab:grey')
#ax3.axvline(25, linestyle='--', linewidth=3, color=color9)
ax3.legend(loc='best', fontsize=fsize)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax3.yaxis.set_major_formatter(formatter)
formatter.set_powerlimits((-1,1)) 
ax3.set_xlabel(r'$x\ (\mathrm{m})$', fontsize=fsize)
ax3.set_ylabel(r'$V_{lf}$ $(\mathrm{m^2 s^{-1}})$', fontsize=fsize)
ax3.grid(True)
ax3.tick_params(axis='x', which='major', labelsize=fsize)
ax3.tick_params(axis='y', which='major', labelsize=fsize) 
ax3.axvspan(31.5, 35, color=color, alpha=0.3)        
ax3.text(33.5, 2.7*10**-2, r'$\mathrm{(d)}$', fontsize=fsize)

ax4 = fig.add_subplot(spec[1:,3:])
ax4.plot(x-22, -u_ex_cross1, '-', linewidth=lwidth, markersize=msize, color=color1, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[0])
ax4.plot(x-22, -u_ex_cross5, '-', linewidth=lwidth, markersize=msize, color=color2, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[1])
ax4.plot(x-22, -u_ex_cross10, '-', linewidth=lwidth, markersize=msize, color=color3, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[2])
ax4.plot(x-22, -u_ex_cross20, '-', linewidth=lwidth, markersize=msize, color=color4, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[3])
ax4.plot(x-22, -u_ex_cross30, '-', linewidth=lwidth, markersize=msize, color=color5, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[4])
ax4.plot(x-22, -u_ex_cross40, '-', linewidth=lwidth, markersize=msize, color=color6, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[5])
#ax4.axvline(30, linestyle='--', linewidth=3, color=color7)
ax4.axvline(27.5, linestyle='--', linewidth=3, color='tab:grey')
#ax4.axvline(25, linestyle='--', linewidth=3, color=color9)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax4.yaxis.set_major_formatter(formatter)
formatter.set_powerlimits((-1,1)) 
ax4.set_xlabel(r'$x\ (\mathrm{m})$', fontsize=fsize)
ax4.set_ylabel(r'$U_{ex}$ $(\overline{U_{ex}})$ $\mathrm{(ms^{-1})}$', fontsize=fsize)
ax4.grid(True)
ax4.tick_params(axis='x', which='major', labelsize=fsize)
ax4.tick_params(axis='y', which='major', labelsize=fsize) 
ax4.axvspan(31.5, 35, color=color, alpha=0.3)        
ax4.text(33.5, 5.5*10**-2, r'$\mathrm{(e)}$', fontsize=fsize)
fig.tight_layout()
fig.savefig(os.path.join(plotsavedir, 'low_freq_u_ex_dirspread.png'))
