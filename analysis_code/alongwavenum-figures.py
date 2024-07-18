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
from scipy.signal import welch

plt.ion()
#plt.style.use('ggplot')
plt.style.use('classic')

WL = 550
dx = 0.05
dy = 0.1
sz = 31.5 - 23.2

xloc1 = 31.5+22
xloc2 = (31.5 - sz*0.5)+22 
xloc3 = (31.5 - sz*1)+22
xloc4 = (31.5 - sz*1.5+22)

lwidth = 2

dirspread = np.array([0.3, 3.8, 11.2, 16.5, 21.6, 24.9])

plotsavedir = os.path.join('/gscratch', 'nearshore', 'enuss', 'lab_runs_y550', 'postprocessing', 'plots')

rundir = 'hmo25_dir1_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
u_psi1 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['u_psi']
x = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['x']
y = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['y']

xind1 = np.where((x<xloc1) & (x>=xloc2))[0]
xind2 = np.where((x<xloc2) & (x>=xloc3))[0]
xind3 = np.where((x<xloc3) & (x>=xloc4))[0]

flist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]
v_psi1 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['v_psi']

curl_psi1 = np.gradient(v_psi1, dx, axis=2) - np.gradient(u_psi1, dy, axis=1)

freq1, wavenumber1 = welch(curl_psi1, fs=1/dy, window='hann', nperseg=curl_psi1.shape[1], axis=1)
del curl_psi1

wavenum_inner1 = np.mean(np.mean(wavenumber1[:,:,xind1], axis=-1), axis=0)
wavenum_outer1 = np.mean(np.mean(wavenumber1[:,:,xind2], axis=-1), axis=0)
wavenum_offshore1 = np.mean(np.mean(wavenumber1[:,:,xind3], axis=-1), axis=0)
del wavenumber1

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx1 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry1 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

curl_fbr1 = np.gradient(fbry1, dx, axis=2) - np.gradient(fbrx1, dy, axis=1)

#freq1, fbrwavenumber1 = welch(curl_fbr1, fs=1/dy, window='hann', nperseg=curl_fbr1.shape[1], axis=1)
freq1_fbr, fbrwavenumber1 = welch(curl_fbr1, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr1

fbrwavenum_sz1 = np.mean(np.mean(fbrwavenumber1[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
fbrwavenum_inner1 = np.mean(np.mean(fbrwavenumber1[:,:,xind1], axis=-1), axis=0)
fbrwavenum_outer1 = np.mean(np.mean(fbrwavenumber1[:,:,xind2], axis=-1), axis=0)
fbrwavenum_offshore1 = np.mean(np.mean(fbrwavenumber1[:,:,xind3], axis=-1), axis=0)
del fbrwavenumber1

##################################################################
rundir = 'hmo25_dir5_tp2_ntheta15'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
u_psi5 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['u_psi']

flist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]
v_psi5 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['v_psi']

curl_psi5 = np.gradient(v_psi5, dx, axis=2) - np.gradient(u_psi5, dy, axis=1)

freq5, wavenumber5 = welch(curl_psi5, fs=1/dy, window='hann', nperseg=curl_psi5.shape[1], axis=1)
del curl_psi5

wavenum_inner5 = np.mean(np.mean(wavenumber5[:,:,xind1], axis=-1), axis=0)
wavenum_outer5 = np.mean(np.mean(wavenumber5[:,:,xind2], axis=-1), axis=0)
wavenum_offshore5 = np.mean(np.mean(wavenumber5[:,:,xind3], axis=-1), axis=0)
del wavenumber5

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx5 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry5 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

curl_fbr5 = np.gradient(fbry5, dx, axis=2) - np.gradient(fbrx5, dy, axis=1)

#freq5, fbrwavenumber5 = welch(curl_fbr5, fs=1/dy, window='hann', nperseg=curl_fbr5.shape[1], axis=1)
freq5_fbr, fbrwavenumber5 = welch(curl_fbr5, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr5

fbrwavenum_sz5 = np.mean(np.mean(fbrwavenumber5[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
fbrwavenum_inner5 = np.mean(np.mean(fbrwavenumber5[:,:,xind1], axis=-1), axis=0)
fbrwavenum_outer5 = np.mean(np.mean(fbrwavenumber5[:,:,xind2], axis=-1), axis=0)
fbrwavenum_offshore5 = np.mean(np.mean(fbrwavenumber5[:,:,xind3], axis=-1), axis=0)
del fbrwavenumber5

#####################################################################################
rundir = 'hmo25_dir10_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
u_psi10 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['u_psi']

flist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]
v_psi10 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['v_psi']

curl_psi10 = np.gradient(v_psi10, dx, axis=2) - np.gradient(u_psi10, dy, axis=1)

freq10, wavenumber10 = welch(curl_psi10, fs=1/dy, window='hann', nperseg=curl_psi10.shape[1], axis=1)
del curl_psi10

wavenum_inner10 = np.mean(np.mean(wavenumber10[:,:,xind1], axis=-1), axis=0)
wavenum_outer10 = np.mean(np.mean(wavenumber10[:,:,xind2], axis=-1), axis=0)
wavenum_offshore10 = np.mean(np.mean(wavenumber10[:,:,xind3], axis=-1), axis=0)
del wavenumber10

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx10 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry10 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

curl_fbr10 = np.gradient(fbry10, dx, axis=2) - np.gradient(fbrx10, dy, axis=1)

#freq10, fbrwavenumber10 = welch(curl_fbr10, fs=1/dy, window='hann', nperseg=curl_fbr10.shape[1], axis=1)
freq10_fbr, fbrwavenumber10 = welch(curl_fbr10, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr10

fbrwavenum_sz10 = np.mean(np.mean(fbrwavenumber10[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
fbrwavenum_inner10 = np.mean(np.mean(fbrwavenumber10[:,:,xind1], axis=-1), axis=0)
fbrwavenum_outer10 = np.mean(np.mean(fbrwavenumber10[:,:,xind2], axis=-1), axis=0)
fbrwavenum_offshore10 = np.mean(np.mean(fbrwavenumber10[:,:,xind3], axis=-1), axis=0)
del fbrwavenumber10

#####################################################################################
rundir = 'hmo25_dir20_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
u_psi20 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['u_psi']

flist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]
v_psi20 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['v_psi']

curl_psi20 = np.gradient(v_psi20, dx, axis=2) - np.gradient(u_psi20, dy, axis=1)

freq20, wavenumber20 = welch(curl_psi20, fs=1/dy, window='hann', nperseg=curl_psi20.shape[1], axis=1)
del curl_psi20

wavenum_inner20 = np.mean(np.mean(wavenumber20[:,:,xind1], axis=-1), axis=0)
wavenum_outer20 = np.mean(np.mean(wavenumber20[:,:,xind2], axis=-1), axis=0)
wavenum_offshore20 = np.mean(np.mean(wavenumber20[:,:,xind3], axis=-1), axis=0)
del wavenumber20

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx20 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry20 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

curl_fbr20 = np.gradient(fbry20, dx, axis=2) - np.gradient(fbrx20, dy, axis=1)

#freq20, fbrwavenumber20 = welch(curl_fbr20, fs=1/dy, window='hann', nperseg=curl_fbr20.shape[1], axis=1)
freq20_fbr, fbrwavenumber20 = welch(curl_fbr20, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr20

fbrwavenum_sz20 = np.mean(np.mean(fbrwavenumber20[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
fbrwavenum_inner20 = np.mean(np.mean(fbrwavenumber20[:,:,xind1], axis=-1), axis=0)
fbrwavenum_outer20 = np.mean(np.mean(fbrwavenumber20[:,:,xind2], axis=-1), axis=0)
fbrwavenum_offshore20 = np.mean(np.mean(fbrwavenumber20[:,:,xind3], axis=-1), axis=0)
del fbrwavenumber20

#####################################################################################
rundir = 'hmo25_dir30_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
u_psi30 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['u_psi']

flist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]
v_psi30 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['v_psi']

curl_psi30 = np.gradient(v_psi30, dx, axis=2) - np.gradient(u_psi30, dy, axis=1)

freq30, wavenumber30 = welch(curl_psi30, fs=1/dy, window='hann', nperseg=curl_psi30.shape[1], axis=1)
del curl_psi30

wavenum_inner30 = np.mean(np.mean(wavenumber30[:,:,xind1], axis=-1), axis=0)
wavenum_outer30 = np.mean(np.mean(wavenumber30[:,:,xind2], axis=-1), axis=0)
wavenum_offshore30 = np.mean(np.mean(wavenumber30[:,:,xind3], axis=-1), axis=0)
del wavenumber30

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx30 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry30 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

curl_fbr30 = np.gradient(fbry30, dx, axis=2) - np.gradient(fbrx30, dy, axis=1)

#freq30, fbrwavenumber30 = welch(curl_fbr30, fs=1/dy, window='hann', nperseg=curl_fbr30.shape[1], axis=1)
freq30_fbr, fbrwavenumber30 = welch(curl_fbr30, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr30

fbrwavenum_sz30 = np.mean(np.mean(fbrwavenumber30[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
fbrwavenum_inner30 = np.mean(np.mean(fbrwavenumber30[:,:,xind1], axis=-1), axis=0)
fbrwavenum_outer30 = np.mean(np.mean(fbrwavenumber30[:,:,xind2], axis=-1), axis=0)
fbrwavenum_offshore30 = np.mean(np.mean(fbrwavenumber30[:,:,xind3], axis=-1), axis=0)
del fbrwavenumber30

#####################################################################################
rundir = 'hmo25_dir40_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
u_psi40 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['u_psi']

flist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]
v_psi40 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['v_psi']

curl_psi40 = np.gradient(v_psi40, dx, axis=2) - np.gradient(u_psi40, dy, axis=1)

freq40, wavenumber40 = welch(curl_psi40, fs=1/dy, window='hann', nperseg=curl_psi40.shape[1], axis=1)
del curl_psi40

wavenum_inner40 = np.mean(np.mean(wavenumber40[:,:,xind1], axis=-1), axis=0)
wavenum_outer40 = np.mean(np.mean(wavenumber40[:,:,xind2], axis=-1), axis=0)
wavenum_offshore40 = np.mean(np.mean(wavenumber40[:,:,xind3], axis=-1), axis=0)
del wavenumber40

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx40 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry40 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

curl_fbr40 = np.gradient(fbry40, dx, axis=2) - np.gradient(fbrx40, dy, axis=1)

#freq40, fbrwavenumber40 = welch(curl_fbr40, fs=1/dy, window='hann', nperseg=curl_fbr40.shape[1], axis=1)
freq40_fbr, fbrwavenumber40 = welch(curl_fbr40, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr40

fbrwavenum_sz40 = np.mean(np.mean(fbrwavenumber40[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
fbrwavenum_inner40 = np.mean(np.mean(fbrwavenumber40[:,:,xind1], axis=-1), axis=0)
fbrwavenum_outer40 = np.mean(np.mean(fbrwavenumber40[:,:,xind2], axis=-1), axis=0)
fbrwavenum_offshore40 = np.mean(np.mean(fbrwavenumber40[:,:,xind3], axis=-1), axis=0)
del fbrwavenumber40

#####################################################################################
rundir = 'hmo25_dir20_tp15'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
u_psi20 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['u_psi']

flist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]
v_psi20 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['v_psi']

curl_psi20 = np.gradient(v_psi20, dx, axis=2) - np.gradient(u_psi20, dy, axis=1)

freq20_15, wavenumber20_15 = welch(curl_psi20, fs=1/dy, window='hann', nperseg=curl_psi20.shape[1], axis=1)
del curl_psi20

wavenum_inner20_15 = np.mean(np.mean(wavenumber20_15[:,:,xind1], axis=-1), axis=0)
wavenum_outer20_15 = np.mean(np.mean(wavenumber20_15[:,:,xind2], axis=-1), axis=0)
wavenum_offshore20_15 = np.mean(np.mean(wavenumber20_15[:,:,xind3], axis=-1), axis=0)
del wavenumber20_15

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx20_15 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry20_15 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

curl_fbr20_15 = np.gradient(fbry20_15, dx, axis=2) - np.gradient(fbrx20_15, dy, axis=1)

#freq20_15, fbrwavenumber20_15 = welch(curl_fbr20_15, fs=1/dy, window='hann', nperseg=curl_fbr20_15.shape[1], axis=1)
freq20_15_fbr, fbrwavenumber20_15 = welch(curl_fbr20_15, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr20_15

fbrwavenum_sz20_15 = np.mean(np.mean(fbrwavenumber20_15[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
fbrwavenum_inner20_15 = np.mean(np.mean(fbrwavenumber20_15[:,:,xind1], axis=-1), axis=0)
fbrwavenum_outer20_15 = np.mean(np.mean(fbrwavenumber20_15[:,:,xind2], axis=-1), axis=0)
fbrwavenum_offshore20_15 = np.mean(np.mean(fbrwavenumber20_15[:,:,xind3], axis=-1), axis=0)
del fbrwavenumber20_15

#####################################################################################
rundir = 'hmo25_dir20_tp25'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'u_psi_1.nc'), os.path.join(fdir, 'u_psi_2.nc'), os.path.join(fdir, 'u_psi_3.nc'), os.path.join(fdir, 'u_psi_4.nc')]
u_psi20 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['u_psi']

flist = [os.path.join(fdir, 'v_psi_1.nc'), os.path.join(fdir, 'v_psi_2.nc'), os.path.join(fdir, 'v_psi_3.nc'), os.path.join(fdir, 'v_psi_4.nc')]
v_psi20 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['v_psi']

curl_psi20 = np.gradient(v_psi20, dx, axis=2) - np.gradient(u_psi20, dy, axis=1)

freq20_25, wavenumber20_25 = welch(curl_psi20, fs=1/dy, window='hann', nperseg=curl_psi20.shape[1], axis=1)
del curl_psi20

wavenum_inner20_25 = np.mean(np.mean(wavenumber20_25[:,:,xind1], axis=-1), axis=0)
wavenum_outer20_25 = np.mean(np.mean(wavenumber20_25[:,:,xind2], axis=-1), axis=0)
wavenum_offshore20_25 = np.mean(np.mean(wavenumber20_25[:,:,xind3], axis=-1), axis=0)
del wavenumber20_25

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx20_25 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry20_25 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

curl_fbr20_25 = np.gradient(fbry20_25, dx, axis=2) - np.gradient(fbrx20_25, dy, axis=1)

#freq20_25, fbrwavenumber20_25 = welch(curl_fbr20_25, fs=1/dy, window='hann', nperseg=curl_fbr20_25.shape[1], axis=1)
freq20_25_fbr, fbrwavenumber20_25 = welch(curl_fbr20_25, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr20_25

fbrwavenum_sz20_25 = np.mean(np.mean(fbrwavenumber20_25[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
fbrwavenum_inner20_25 = np.mean(np.mean(fbrwavenumber20_25[:,:,xind1], axis=-1), axis=0)
fbrwavenum_outer20_25 = np.mean(np.mean(fbrwavenumber20_25[:,:,xind2], axis=-1), axis=0)
fbrwavenum_offshore20_25 = np.mean(np.mean(fbrwavenumber20_25[:,:,xind3], axis=-1), axis=0)
del fbrwavenumber20_25

###################
color1='#003f5c'
color2='#444e86'
color3='#955196'
color4='#dd5182'
color5='#ff6e54'
color6='#ffa600'

fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(12,8), sharex=True, sharey=True)
ax[0,0].loglog(freq1, wavenum_inner1, linewidth=lwidth, color=color1, label=r'$\sigma_\theta = %.1f$' % dirspread[0])
ax[0,0].loglog(freq5, wavenum_inner5, linewidth=lwidth, color=color2, label=r'$\sigma_\theta = %.1f$' % dirspread[1])
ax[0,0].loglog(freq10, wavenum_inner10, linewidth=lwidth, color=color3, label=r'$\sigma_\theta = %.1f$' % dirspread[2])
ax[0,0].loglog(freq20, wavenum_inner20, linewidth=lwidth, color=color4, label=r'$\sigma_\theta = %.1f$' % dirspread[3])
ax[0,0].loglog(freq30, wavenum_inner30, linewidth=lwidth, color=color5, label=r'$\sigma_\theta = %.1f$' % dirspread[4])
ax[0,0].loglog(freq40, wavenum_inner40, linewidth=lwidth, color=color6, label=r'$\sigma_\theta = %.1f$' % dirspread[5])
ax[0,0].set_ylim(10**-6, 1)
ax[0,0].set_ylabel(r'$S_{\omega \omega}$ ($s^{-2} m$)')
ax[0,0].set_title(r'$\mathrm{Inner\ surf\ zone}$ $\langle x$ = $%.1f$ - $%.1f \rangle $' % ((xloc1-22), (xloc2-22)))
ax[0,0].grid(True)
ax[0,0].text(2.8, 0.3, r'$\mathrm{(a)}$', fontsize=16)
ax[0,0].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')

ax[0,1].loglog(freq1, wavenum_outer1, linewidth=lwidth, color=color1, label=r'$\sigma_\theta = %.1f$' % dirspread[0])
ax[0,1].loglog(freq5, wavenum_outer5, linewidth=lwidth, color=color2, label=r'$\sigma_\theta = %.1f$' % dirspread[1])
ax[0,1].loglog(freq10, wavenum_outer10, linewidth=lwidth, color=color3, label=r'$\sigma_\theta = %.1f$' % dirspread[2])
ax[0,1].loglog(freq20, wavenum_outer20, linewidth=lwidth, color=color4, label=r'$\sigma_\theta = %.1f$' % dirspread[3])
ax[0,1].loglog(freq30, wavenum_outer30, linewidth=lwidth, color=color5, label=r'$\sigma_\theta = %.1f$' % dirspread[4])
ax[0,1].loglog(freq40, wavenum_outer40, linewidth=lwidth, color=color6, label=r'$\sigma_\theta = %.1f$' % dirspread[5])
ax[0,1].set_ylim(10**-6, 1)
ax[0,1].set_title(r'$\mathrm{Outer\ surf\ zone}$ $\langle x$ = $%.1f$ - $%.1f \rangle $' % ((xloc2-22), (xloc3-22)))
ax[0,1].grid(True)
ax[0,1].text(2.8, 0.3, r'$\mathrm{(b)}$', fontsize=16)
ax[0,1].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')

ax[0,2].loglog(freq1, wavenum_offshore1, linewidth=lwidth, color=color1, label=r'$\sigma_\theta = %.1f$' % dirspread[0])
ax[0,2].loglog(freq5, wavenum_offshore5, linewidth=lwidth, color=color2, label=r'$\sigma_\theta = %.1f$' % dirspread[1])
ax[0,2].loglog(freq10, wavenum_offshore10, linewidth=lwidth, color=color3, label=r'$\sigma_\theta = %.1f$' % dirspread[2])
ax[0,2].loglog(freq20, wavenum_offshore20, linewidth=lwidth, color=color4, label=r'$\sigma_\theta = %.1f$' % dirspread[3])
ax[0,2].loglog(freq30, wavenum_offshore30, linewidth=lwidth, color=color5, label=r'$\sigma_\theta = %.1f$' % dirspread[4])
ax[0,2].loglog(freq40, wavenum_offshore40, linewidth=lwidth, color=color6, label=r'$\sigma_\theta = %.1f$' % dirspread[5])
ax[0,2].set_xlim(freq1[1], 5)
ax[0,2].set_ylim(10**-6, 1)
ax[0,2].set_title(r'$\mathrm{Offshore}$ $\langle x$ = $%.1f$ - $%.1f \rangle $' % ((xloc3-22), (xloc4-22)))
ax[0,2].grid(True)
ax[0,2].text(2.8, 0.3, r'$\mathrm{(c)}$', fontsize=16)
ax[0,2].legend(loc='center right', bbox_to_anchor=(1.7, 0.5))
ax[0,2].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')


ax[1,0].loglog(freq20_15, wavenum_inner20_15, linewidth=lwidth, color=color1, label=r'$T_p$ = 1.5 $\mathrm{s}$')
ax[1,0].loglog(freq20, wavenum_inner20, linewidth=lwidth, color=color4, label=r'$T_p$ = 2.0 $\mathrm{s}$')
ax[1,0].loglog(freq20_25, wavenum_inner20_25, linewidth=lwidth, color=color6, label=r'$T_p$ = 2.5 $\mathrm{s}$')
ax[1,0].set_ylim(10**-6, 1)
ax[1,0].set_ylabel(r'$S_{\omega \omega}$ ($s^{-2} m$)')
ax[1,0].set_xlabel(r'$k_y$ ($m^{-1}$)')
ax[1,0].grid(True)
ax[1,0].text(2.8, 0.3, r'$\mathrm{(d)}$', fontsize=16)
ax[1,0].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')

ax[1,1].loglog(freq20_15, wavenum_outer20_15, linewidth=lwidth, color=color1, label=r'$T_p$ = 1.5 $\mathrm{s}$')
ax[1,1].loglog(freq20, wavenum_outer20, linewidth=lwidth, color=color4, label=r'$T_p$ = 2.0 $\mathrm{s}$')
ax[1,1].loglog(freq20_25, wavenum_outer20_25, linewidth=lwidth, color=color6, label=r'$T_p$ = 2.5 $\mathrm{s}$')
ax[1,1].set_ylim(10**-6, 1)
ax[1,1].set_ylabel(r'$S_{\omega \omega}$ ($s^{-2} m$)')
ax[1,1].set_xlabel(r'$k_y$ ($m^{-1}$)')
ax[1,1].grid(True)
ax[1,1].text(2.8, 0.3, r'$\mathrm{(e)}$', fontsize=16)
ax[1,1].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')

ax[1,2].loglog(freq20_15, wavenum_offshore20_15, linewidth=lwidth, color=color1, label=r'$T_p$ = $1.5$ $\mathrm{s}$')
ax[1,2].loglog(freq20, wavenum_offshore20, linewidth=lwidth, color=color4, label=r'$T_p$ = $2.0$ $\mathrm{s}$')
ax[1,2].loglog(freq20_25, wavenum_offshore20_25, linewidth=lwidth, color=color6, label=r'$T_p$ = $2.5$ $\mathrm{s}$')
ax[1,2].set_ylim(10**-6, 1)
ax[1,2].set_ylabel(r'$S_{\omega \omega}$ ($s^{-2} m$)')
ax[1,2].set_xlabel(r'$k_y$ ($m^{-1}$)')
ax[1,2].grid(True)
ax[1,2].text(2.8, 0.3, r'$\mathrm{(f)}$', fontsize=16)
ax[1,2].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')
ax[1,2].legend(loc='center right', bbox_to_anchor=(1.7, 0.5))

fig.tight_layout()
fig.savefig(os.path.join(plotsavedir, 'along_wavenum_zones.png'))
fig.savefig(os.path.join(plotsavedir, 'along_wavenum_zones.jpg'))

########################################################################
fig, ax = plt.subplots(ncols=2, figsize=(11,4.5), sharex=True, sharey=True)
ax[0].loglog(freq1_fbr, fbrwavenum_sz1, linewidth=lwidth, color=color1, label=r'$\sigma_\theta = %.1f$' % dirspread[0])
ax[0].loglog(freq5_fbr, fbrwavenum_sz5, linewidth=lwidth, color=color2, label=r'$\sigma_\theta = %.1f$' % dirspread[1])
ax[0].loglog(freq10_fbr, fbrwavenum_sz10, linewidth=lwidth, color=color3, label=r'$\sigma_\theta = %.1f$' % dirspread[2])
ax[0].loglog(freq20_fbr, fbrwavenum_sz20, linewidth=lwidth, color=color4, label=r'$\sigma_\theta = %.1f$' % dirspread[3])
ax[0].loglog(freq30_fbr, fbrwavenum_sz30, linewidth=lwidth, color=color5, label=r'$\sigma_\theta = %.1f$' % dirspread[4])
ax[0].loglog(freq40_fbr, fbrwavenum_sz40, linewidth=lwidth, color=color6, label=r'$\sigma_\theta = %.1f$' % dirspread[5])
ax[0].set_ylim(10**-2, 10)
ax[0].set_ylabel(r'$S_{\nabla \times \bf{F_{br}}}$ ($ms^{-4}$)')
ax[0].set_xlabel(r'$k_y$ ($m^{-1}$)')
ax[0].grid(True)
ax[0].text(2.5, 3, r'$\mathrm{(a)}$', fontsize=16)
ax[0].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')
ax[0].legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

ax[1].loglog(freq20_15_fbr, fbrwavenum_sz20_15, linewidth=lwidth, color=color1, label=r'$T_p$ = 1.5 $\mathrm{s}$')
ax[1].loglog(freq20_fbr, fbrwavenum_sz20, linewidth=lwidth, color=color4, label=r'$T_p$ = 2.0 $\mathrm{s}$')
ax[1].loglog(freq20_25_fbr, fbrwavenum_sz20_25, linewidth=lwidth, color=color6, label=r'$T_p$ = 2.5 $\mathrm{s}$')
ax[1].set_ylim(10**-2, 10)
ax[1].set_ylabel(r'$S_{\nabla \times \bf{F_{br}}}$ ($ms^{-4}$)')
ax[1].set_xlabel(r'$k_y$ ($m^{-1}$)')
ax[1].grid(True)
ax[1].text(2.5, 3, r'$\mathrm{(b)}$', fontsize=16)
ax[1].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')
ax[1].legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

fig.tight_layout()
fig.savefig(os.path.join(plotsavedir, 'along_fbrwavenum.png'))


fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(8,4))
df = freq1_fbr[1]-freq1_fbr[0]
curl_fbr = np.array([np.sum(fbrwavenum_sz1), np.sum(fbrwavenum_sz5), np.sum(fbrwavenum_sz10), np.sum(fbrwavenum_sz20), 
					 np.sum(fbrwavenum_sz30), np.sum(fbrwavenum_sz40)])*df
ax[0].plot(dirspread, curl_fbr, 'o-', color=color1)
ax[0].set_ylabel(r'Integrated $S_{\nabla \times \bf{F_{br}}}$')
ax[0].set_xlabel(r'$\sigma_\theta$')
ax[0].grid(True)
curl_fbr = np.array([np.sum(fbrwavenum_sz20_15), np.sum(fbrwavenum_sz20), np.sum(fbrwavenum_sz20_25)])*df
ax[1].plot(np.array([1.5, 2, 2.5]), curl_fbr, 'o-', color=color1)
ax[1].set_xlabel(r'$T_p$')
ax[1].grid(True)
fig.tight_layout()
fig.savefig(os.path.join(plotsavedir, 'integrated_along_wavenum.png'))

fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(12,8), sharex=True, sharey=True)
ax[0,0].loglog(freq1_fbr, fbrwavenum_inner1, linewidth=lwidth, color=color1, label=r'$\sigma_\theta = %.1f$' % dirspread[0])
ax[0,0].loglog(freq5_fbr, fbrwavenum_inner5, linewidth=lwidth, color=color2, label=r'$\sigma_\theta = %.1f$' % dirspread[1])
ax[0,0].loglog(freq10_fbr, fbrwavenum_inner10, linewidth=lwidth, color=color3, label=r'$\sigma_\theta = %.1f$' % dirspread[2])
ax[0,0].loglog(freq20_fbr, fbrwavenum_inner20, linewidth=lwidth, color=color4, label=r'$\sigma_\theta = %.1f$' % dirspread[3])
ax[0,0].loglog(freq30_fbr, fbrwavenum_inner30, linewidth=lwidth, color=color5, label=r'$\sigma_\theta = %.1f$' % dirspread[4])
ax[0,0].loglog(freq40_fbr, fbrwavenum_inner40, linewidth=lwidth, color=color6, label=r'$\sigma_\theta = %.1f$' % dirspread[5])
ax[0,0].set_ylim(10**-5, 10)
ax[0,0].set_ylabel(r'$S_{\omega \omega}$ ($s^{-2} m$)')
ax[0,0].set_title(r'$\mathrm{Inner\ surf\ zone}$ $\langle x$ = $%.1f$ - $%.1f \rangle $' % ((xloc1-22), (xloc2-22)))
ax[0,0].grid(True)
ax[0,0].text(2.5, 3, r'$\mathrm{(a)}$', fontsize=16)
ax[0,0].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')

ax[0,1].loglog(freq1_fbr, fbrwavenum_outer1, linewidth=lwidth, color=color1, label=r'$\sigma_\theta = %.1f$' % dirspread[0])
ax[0,1].loglog(freq5_fbr, fbrwavenum_outer5, linewidth=lwidth, color=color2, label=r'$\sigma_\theta = %.1f$' % dirspread[1])
ax[0,1].loglog(freq10_fbr, fbrwavenum_outer10, linewidth=lwidth, color=color3, label=r'$\sigma_\theta = %.1f$' % dirspread[2])
ax[0,1].loglog(freq20_fbr, fbrwavenum_outer20, linewidth=lwidth, color=color4, label=r'$\sigma_\theta = %.1f$' % dirspread[3])
ax[0,1].loglog(freq30_fbr, fbrwavenum_outer30, linewidth=lwidth, color=color5, label=r'$\sigma_\theta = %.1f$' % dirspread[4])
ax[0,1].loglog(freq40_fbr, fbrwavenum_outer40, linewidth=lwidth, color=color6, label=r'$\sigma_\theta = %.1f$' % dirspread[5])
ax[0,1].set_ylim(10**-5, 10)
ax[0,1].set_title(r'$\mathrm{Outer\ surf\ zone}$ $\langle x$ = $%.1f$ - $%.1f \rangle $' % ((xloc2-22), (xloc3-22)))
ax[0,1].grid(True)
ax[0,1].text(2.5, 3, r'$\mathrm{(b)}$', fontsize=16)
ax[0,1].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')

ax[0,2].loglog(freq1_fbr, fbrwavenum_offshore1, linewidth=lwidth, color=color1, label=r'$\sigma_\theta = %.1f$' % dirspread[0])
ax[0,2].loglog(freq5_fbr, fbrwavenum_offshore5, linewidth=lwidth, color=color2, label=r'$\sigma_\theta = %.1f$' % dirspread[1])
ax[0,2].loglog(freq10_fbr, fbrwavenum_offshore10, linewidth=lwidth, color=color3, label=r'$\sigma_\theta = %.1f$' % dirspread[2])
ax[0,2].loglog(freq20_fbr, fbrwavenum_offshore20, linewidth=lwidth, color=color4, label=r'$\sigma_\theta = %.1f$' % dirspread[3])
ax[0,2].loglog(freq30_fbr, fbrwavenum_offshore30, linewidth=lwidth, color=color5, label=r'$\sigma_\theta = %.1f$' % dirspread[4])
ax[0,2].loglog(freq40_fbr, fbrwavenum_offshore40, linewidth=lwidth, color=color6, label=r'$\sigma_\theta = %.1f$' % dirspread[5])
ax[0,2].set_xlim(freq1[1], 5)
ax[0,2].set_ylim(10**-5, 10)
ax[0,2].set_title(r'$\mathrm{Offshore}$ $\langle x$ = $%.1f$ - $%.1f \rangle $' % ((xloc3-22), (xloc4-22)))
ax[0,2].grid(True)
ax[0,2].text(2.5, 3, r'$\mathrm{(c)}$', fontsize=16)
ax[0,2].legend(loc='center right', bbox_to_anchor=(1.7, 0.5))
ax[0,2].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')


ax[1,0].loglog(freq20_15_fbr, fbrwavenum_inner20_15, linewidth=lwidth, color=color1, label=r'$T_p$ = 1.5 $\mathrm{s}$')
ax[1,0].loglog(freq20_fbr, fbrwavenum_inner20, linewidth=lwidth, color=color4, label=r'$T_p$ = 2.0 $\mathrm{s}$')
ax[1,0].loglog(freq20_25_fbr, fbrwavenum_inner20_25, linewidth=lwidth, color=color6, label=r'$T_p$ = 2.5 $\mathrm{s}$')
ax[1,0].set_ylim(10**-5, 10)
ax[1,0].set_ylabel(r'$S_{\omega \omega}$ ($s^{-2} m$)')
ax[1,0].set_xlabel(r'$k_y$ ($m^{-1}$)')
ax[1,0].grid(True)
ax[1,0].text(2.5, 3, r'$\mathrm{(d)}$', fontsize=16)
ax[1,0].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')

ax[1,1].loglog(freq20_15_fbr, fbrwavenum_outer20_15, linewidth=lwidth, color=color1, label=r'$T_p$ = 1.5 $\mathrm{s}$')
ax[1,1].loglog(freq20_fbr, fbrwavenum_outer20, linewidth=lwidth, color=color4, label=r'$T_p$ = 2.0 $\mathrm{s}$')
ax[1,1].loglog(freq20_25_fbr, fbrwavenum_outer20_25, linewidth=lwidth, color=color6, label=r'$T_p$ = 2.5 $\mathrm{s}$')
ax[1,1].set_ylim(10**-5, 10)
ax[1,1].set_ylabel(r'$S_{\omega \omega}$ ($s^{-2} m$)')
ax[1,1].set_xlabel(r'$k_y$ ($m^{-1}$)')
ax[1,1].grid(True)
ax[1,1].text(2.5, 3, r'$\mathrm{(e)}$', fontsize=16)
ax[1,1].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')

ax[1,2].loglog(freq20_15_fbr, fbrwavenum_offshore20_15, linewidth=lwidth, color=color1, label=r'$T_p$ = $1.5$ $\mathrm{s}$')
ax[1,2].loglog(freq20_fbr, fbrwavenum_offshore20, linewidth=lwidth, color=color4, label=r'$T_p$ = $2.0$ $\mathrm{s}$')
ax[1,2].loglog(freq20_25_fbr, fbrwavenum_offshore20_25, linewidth=lwidth, color=color6, label=r'$T_p$ = $2.5$ $\mathrm{s}$')
ax[1,2].set_ylim(10**-5, 10)
ax[1,2].set_ylabel(r'$S_{\omega \omega}$ ($s^{-2} m$)')
ax[1,2].set_xlabel(r'$k_y$ ($m^{-1}$)')
ax[1,2].grid(True)
ax[1,2].text(2.5, 3, r'$\mathrm{(f)}$', fontsize=16)
ax[1,2].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')
ax[1,2].legend(loc='center right', bbox_to_anchor=(1.7, 0.5))

fig.tight_layout()
fig.savefig(os.path.join(plotsavedir, 'along_fbrwavenum_zones.png'))




#### testing fbr wavenumber spectra
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
from scipy.signal import welch

plt.ion()
#plt.style.use('ggplot')
plt.style.use('classic')

WL = 550
dx = 0.05
dy = 0.1
sz = 31.5 - 23.2

xloc1 = 31.5+22
xloc2 = (32.5 - sz*0.5)+22 
xloc3 = (32.5 - sz*1)+22
xloc4 = (32.5 - sz*1.5+22)

lwidth = 2

dirspread = np.array([0.22,3.2,9.5,16.5,22.0,25.5])

plotsavedir = os.path.join('/gscratch', 'nearshore', 'enuss', 'lab_runs_y550', 'postprocessing', 'plots')

rundir = 'hmo25_dir1_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx1 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']
x = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['x']
y = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['y']

xind1 = np.where((x<xloc1) & (x>=xloc2))[0]
xind2 = np.where((x<xloc2) & (x>=xloc3))[0]
xind3 = np.where((x<xloc3) & (x>=xloc4))[0]

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry1 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

fbrx1, x, y = mod_utils.load_masked_variable('fbrx', os.path.join(fdir, 'fbrx_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))
fbry1, x, y = mod_utils.load_masked_variable('fbry', os.path.join(fdir, 'fbry_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))

start = 1500
fbrx1_bar = np.zeros(fbrx1[start:,:,:].shape)
fbry1_bar = np.zeros(fbry1[start:,:,:].shape)

for i in range(start, start+len(fbrx1[start:,:,:])):
	fbrx1_bar = mod_utils.spatially_avg(fbrx1[i,:,:], x, y)
	fbry1_bar = mod_utils.spatially_avg(fbry1[i,:,:], x, y)

curl_fbr1 = np.gradient(fbry1_bar, dx, axis=2) - np.gradient(fbrx1_bar, dy, axis=1)
#curl_fbr1 = np.gradient(fbry1, dx, axis=2) - np.gradient(fbrx1, dy, axis=1)

#freq1, fbrwavenumber1 = welch(curl_fbr1, fs=1/dy, window='hann', nperseg=curl_fbr1.shape[1], axis=1)
freq1_fbr, fbrwavenumber1 = welch(curl_fbr1, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr1

fbrwavenum_sz1 = np.mean(np.mean(fbrwavenumber1[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
del fbrwavenumber1

##################################################################
rundir = 'hmo25_dir5_tp2_ntheta15'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx5 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry5 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

fbrx5, x, y = mod_utils.load_masked_variable('fbrx', os.path.join(fdir, 'fbrx_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))
fbry5, x, y = mod_utils.load_masked_variable('fbry', os.path.join(fdir, 'fbry_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))

start = 1500
fbrx5_bar = np.zeros(fbrx5[start:,:,:].shape)
fbry5_bar = np.zeros(fbry5[start:,:,:].shape)

for i in range(start, start+len(fbrx5[start:,:,:])):
	fbrx5_bar = mod_utils.spatially_avg(fbrx5[i,:,:], x, y)
	fbry5_bar = mod_utils.spatially_avg(fbry5[i,:,:], x, y)

curl_fbr5 = np.gradient(fbry5_bar, dx, axis=2) - np.gradient(fbrx5_bar, dy, axis=1)
#curl_fbr5 = np.gradient(fbry5, dx, axis=2) - np.gradient(fbrx5, dy, axis=1)

#freq5, fbrwavenumber5 = welch(curl_fbr5, fs=1/dy, window='hann', nperseg=curl_fbr5.shape[1], axis=1)
freq5_fbr, fbrwavenumber5 = welch(curl_fbr5, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr5

fbrwavenum_sz5 = np.mean(np.mean(fbrwavenumber5[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
del fbrwavenumber5

#####################################################################################
rundir = 'hmo25_dir10_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx10 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry10 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

fbrx10, x, y = mod_utils.load_masked_variable('fbrx', os.path.join(fdir, 'fbrx_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))
fbry10, x, y = mod_utils.load_masked_variable('fbry', os.path.join(fdir, 'fbry_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))

start = 1500
fbrx10_bar = np.zeros(fbrx10[start:,:,:].shape)
fbry10_bar = np.zeros(fbry10[start:,:,:].shape)

for i in range(start, start+len(fbrx10[start:,:,:])):
	fbrx10_bar = mod_utils.spatially_avg(fbrx10[i,:,:], x, y)
	fbry10_bar = mod_utils.spatially_avg(fbry10[i,:,:], x, y)

curl_fbr10 = np.gradient(fbry10_bar, dx, axis=2) - np.gradient(fbrx10_bar, dy, axis=1)
#curl_fbr10 = np.gradient(fbry10, dx, axis=2) - np.gradient(fbrx10, dy, axis=1)

#freq10, fbrwavenumber10 = welch(curl_fbr10, fs=1/dy, window='hann', nperseg=curl_fbr10.shape[1], axis=1)
freq10_fbr, fbrwavenumber10 = welch(curl_fbr10, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr10

fbrwavenum_sz10 = np.mean(np.mean(fbrwavenumber10[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
del fbrwavenumber10

#####################################################################################
rundir = 'hmo25_dir20_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx20 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry20 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

fbrx20, x, y = mod_utils.load_masked_variable('fbrx', os.path.join(fdir, 'fbrx_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))
fbry20, x, y = mod_utils.load_masked_variable('fbry', os.path.join(fdir, 'fbry_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))

start = 1500
fbrx20_bar = np.zeros(fbrx20[start:,:,:].shape)
fbry20_bar = np.zeros(fbry20[start:,:,:].shape)

for i in range(start, start+len(fbrx20[start:,:,:])):
	fbrx20_bar = mod_utils.spatially_avg(fbrx20[i,:,:], x, y)
	fbry20_bar = mod_utils.spatially_avg(fbry20[i,:,:], x, y)

curl_fbr20 = np.gradient(fbry20_bar, dx, axis=2) - np.gradient(fbrx20_bar, dy, axis=1)
#curl_fbr20 = np.gradient(fbry20, dx, axis=2) - np.gradient(fbrx20, dy, axis=1)

#freq20, fbrwavenumber20 = welch(curl_fbr20, fs=1/dy, window='hann', nperseg=curl_fbr20.shape[1], axis=1)
freq20_fbr, fbrwavenumber20 = welch(curl_fbr20, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr20

fbrwavenum_sz20 = np.mean(np.mean(fbrwavenumber20[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
del fbrwavenumber20

#####################################################################################
rundir = 'hmo25_dir30_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx30 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry30 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

fbrx30, x, y = mod_utils.load_masked_variable('fbrx', os.path.join(fdir, 'fbrx_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))
fbry30, x, y = mod_utils.load_masked_variable('fbry', os.path.join(fdir, 'fbry_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))

start = 1500
fbrx30_bar = np.zeros(fbrx30[start:,:,:].shape)
fbry30_bar = np.zeros(fbry30[start:,:,:].shape)

for i in range(start, start+len(fbrx30[start:,:,:])):
	fbrx30_bar = mod_utils.spatially_avg(fbrx30[i,:,:], x, y)
	fbry30_bar = mod_utils.spatially_avg(fbry30[i,:,:], x, y)

curl_fbr30 = np.gradient(fbry30_bar, dx, axis=2) - np.gradient(fbrx30_bar, dy, axis=1)
#curl_fbr30 = np.gradient(fbry30, dx, axis=2) - np.gradient(fbrx30, dy, axis=1)

#freq30, fbrwavenumber30 = welch(curl_fbr30, fs=1/dy, window='hann', nperseg=curl_fbr30.shape[1], axis=1)
freq30_fbr, fbrwavenumber30 = welch(curl_fbr30, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr30

fbrwavenum_sz30 = np.mean(np.mean(fbrwavenumber30[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
del fbrwavenumber30

#####################################################################################
rundir = 'hmo25_dir40_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx40 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry40 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

fbrx40, x, y = mod_utils.load_masked_variable('fbrx', os.path.join(fdir, 'fbrx_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))
fbry40, x, y = mod_utils.load_masked_variable('fbry', os.path.join(fdir, 'fbry_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))

start = 1500
fbrx40_bar = np.zeros(fbrx40[start:,:,:].shape)
fbry40_bar = np.zeros(fbry40[start:,:,:].shape)

for i in range(start, start+len(fbrx40[start:,:,:])):
	fbrx40_bar = mod_utils.spatially_avg(fbrx40[i,:,:], x, y)
	fbry40_bar = mod_utils.spatially_avg(fbry40[i,:,:], x, y)

curl_fbr40 = np.gradient(fbry40_bar, dx, axis=2) - np.gradient(fbrx40_bar, dy, axis=1)
#curl_fbr40 = np.gradient(fbry40, dx, axis=2) - np.gradient(fbrx40, dy, axis=1)

#freq40, fbrwavenumber40 = welch(curl_fbr40, fs=1/dy, window='hann', nperseg=curl_fbr40.shape[1], axis=1)
freq40_fbr, fbrwavenumber40 = welch(curl_fbr40, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr40

fbrwavenum_sz40 = np.mean(np.mean(fbrwavenumber40[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
del fbrwavenumber40

#####################################################################################
rundir = 'hmo25_dir20_tp15'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx20_15 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry20_15 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

fbrx20_15, x, y = mod_utils.load_masked_variable('fbrx', os.path.join(fdir, 'fbrx_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))
fbry20_15, x, y = mod_utils.load_masked_variable('fbry', os.path.join(fdir, 'fbry_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))

start = 1500
fbrx20_15_bar = np.zeros(fbrx20_15[start:,:,:].shape)
fbry20_15_bar = np.zeros(fbry20_15[start:,:,:].shape)

for i in range(start, start+len(fbrx20_15[start:,:,:])):
	fbrx20_15_bar = mod_utils.spatially_avg(fbrx20_15[i,:,:], x, y)
	fbry20_15_bar = mod_utils.spatially_avg(fbry20_15[i,:,:], x, y)

curl_fbr20_15 = np.gradient(fbry20_15_bar, dx, axis=2) - np.gradient(fbrx20_15_bar, dy, axis=1)
#curl_fbr20_15 = np.gradient(fbry20_15, dx, axis=2) - np.gradient(fbrx20_15, dy, axis=1)

#freq20_15, fbrwavenumber20_15 = welch(curl_fbr20_15, fs=1/dy, window='hann', nperseg=curl_fbr20_15.shape[1], axis=1)
freq20_15_fbr, fbrwavenumber20_15 = welch(curl_fbr20_15, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr20_15

fbrwavenum_sz20_15 = np.mean(np.mean(fbrwavenumber20_15[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
del fbrwavenumber20_15

#####################################################################################
rundir = 'hmo25_dir20_tp25_unfinished'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

flist = [os.path.join(fdir, 'fbrx_1.nc'), os.path.join(fdir, 'fbrx_2.nc'), os.path.join(fdir, 'fbrx_3.nc'), os.path.join(fdir, 'fbrx_4.nc')]
fbrx20_25 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']

flist = [os.path.join(fdir, 'fbry_1.nc'), os.path.join(fdir, 'fbry_2.nc'), os.path.join(fdir, 'fbry_3.nc'), os.path.join(fdir, 'fbry_4.nc')]
fbry20_25 = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

fbrx20_25, x, y = mod_utils.load_masked_variable('fbrx', os.path.join(fdir, 'fbrx_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))
fbry20_25, x, y = mod_utils.load_masked_variable('fbry', os.path.join(fdir, 'fbry_*.nc'), 'mask', os.path.join(fdir, 'mask_*.nc'))

start = 1500
fbrx20_25_bar = np.zeros(fbrx20_25[start:,:,:].shape)
fbry20_25_bar = np.zeros(fbry20_25[start:,:,:].shape)

for i in range(start, start+len(fbrx20_25[start:,:,:])):
	fbrx20_25_bar = mod_utils.spatially_avg(fbrx20_25[i,:,:], x, y)
	fbry20_25_bar = mod_utils.spatially_avg(fbry20_25[i,:,:], x, y)

curl_fbr20_25 = np.gradient(fbry20_25_bar, dx, axis=2) - np.gradient(fbrx20_25_bar, dy, axis=1)
#curl_fbr20_25 = np.gradient(fbry20_25, dx, axis=2) - np.gradient(fbrx20_25, dy, axis=1)

#freq20_25, fbrwavenumber20_25 = welch(curl_fbr20_25, fs=1/dy, window='hann', nperseg=curl_fbr20_25.shape[1], axis=1)
freq20_25_fbr, fbrwavenumber20_25 = welch(curl_fbr20_25, fs=1/dy, window='hann', nperseg=WL, axis=1)
del curl_fbr20_25

fbrwavenum_sz20_25 = np.mean(np.mean(fbrwavenumber20_25[:,:,np.concatenate((xind2, xind1))], axis=-1), axis=0)
del fbrwavenumber20_25

###################
color1='#003f5c'
color2='#444e86'
color3='#955196'
color4='#dd5182'
color5='#ff6e54'
color6='#ffa600'

########################################################################
fig, ax = plt.subplots(ncols=2, figsize=(11,4.5), sharex=True, sharey=True)
ax[0].loglog(freq1_fbr, fbrwavenum_sz1, linewidth=lwidth, color=color1, label=r'$\sigma_\theta = %.1f$' % dirspread[0])
ax[0].loglog(freq5_fbr, fbrwavenum_sz5, linewidth=lwidth, color=color2, label=r'$\sigma_\theta = %.1f$' % dirspread[1])
ax[0].loglog(freq10_fbr, fbrwavenum_sz10, linewidth=lwidth, color=color3, label=r'$\sigma_\theta = %.1f$' % dirspread[2])
ax[0].loglog(freq20_fbr, fbrwavenum_sz20, linewidth=lwidth, color=color4, label=r'$\sigma_\theta = %.1f$' % dirspread[3])
ax[0].loglog(freq30_fbr, fbrwavenum_sz30, linewidth=lwidth, color=color5, label=r'$\sigma_\theta = %.1f$' % dirspread[4])
ax[0].loglog(freq40_fbr, fbrwavenum_sz40, linewidth=lwidth, color=color6, label=r'$\sigma_\theta = %.1f$' % dirspread[5])
ax[0].set_ylim(10**-2, 10)
ax[0].set_ylabel(r'$S_{\nabla \times \bf{F_{br}}}$ ($ms^{-4}$)')
ax[0].set_xlabel(r'$k_y$ ($m^{-1}$)')
ax[0].grid(True)
ax[0].text(2.5, 3, r'$\mathrm{(a)}$', fontsize=16)
ax[0].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')
ax[0].legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

ax[1].loglog(freq20_15_fbr, fbrwavenum_sz20_15, linewidth=lwidth, color=color1, label=r'$T_p$ = 1.5 $\mathrm{s}$')
ax[1].loglog(freq20_fbr, fbrwavenum_sz20, linewidth=lwidth, color=color4, label=r'$T_p$ = 2.0 $\mathrm{s}$')
ax[1].loglog(freq20_25_fbr, fbrwavenum_sz20_25, linewidth=lwidth, color=color6, label=r'$T_p$ = 2.5 $\mathrm{s}$')
ax[1].set_ylim(10**-2, 10)
ax[1].set_ylabel(r'$S_{\nabla \times \bf{F_{br}}}$ ($ms^{-4}$)')
ax[1].set_xlabel(r'$k_y$ ($m^{-1}$)')
ax[1].grid(True)
ax[1].text(2.5, 3, r'$\mathrm{(b)}$', fontsize=16)
ax[1].axvline(1/sz, linestyle='--', linewidth=lwidth, color='tab:grey')
ax[1].legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

fig.tight_layout()
fig.savefig(os.path.join(plotsavedir, 'along_fbrwavenum_averaged.png'))

