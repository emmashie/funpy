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

#WL = 128*2; OL = 64*2
dx = 0.05
dy = 0.1
sz = 31.5 - 23.5

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
curl_psi10

wavenum_inner10 = np.mean(np.mean(wavenumber10[:,:,xind1], axis=-1), axis=0)
wavenum_outer10 = np.mean(np.mean(wavenumber10[:,:,xind2], axis=-1), axis=0)
wavenum_offshore10 = np.mean(np.mean(wavenumber10[:,:,xind3], axis=-1), axis=0)

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
ax[0,2].axvline(1/4, linestyle='--', linewidth=lwidth, color='tab:grey')


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