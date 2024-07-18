import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import pandas as pd
import xarray as xr
import funpy.postprocess as fp
import funpy.model_utils as mod_utils

rundir = 'lab_runs_y550'
outputdir = 'output_hmo25_dir20_tp25'
fdir = os.path.join('/gscratch', 'nearshore', 'enuss', rundir, outputdir)
savefulldir = os.path.join('/gscratch', 'nearshore', 'enuss', rundir, 'postprocessing', 'compiled_' + outputdir, 'full_netcdfs')
savedir = os.path.join('/gscratch', 'nearshore', 'enuss', rundir, 'postprocessing', 'compiled_' + outputdir, 'lab_netcdfs')

if not os.path.exists(savedir):
    os.makedirs(savedir)

if not os.path.exists(savefulldir):
    os.makedirs(savefulldir)

if not os.path.exists(os.path.join(fdir, 'compiled' + outputdir, 'dep.out')):
    os.system('cp /gscratch/nearshore/enuss/'+rundir+'/'+outputdir+'/dep.out /gscratch/nearshore/enuss/'+rundir+'/postprocessing/compiled_'+outputdir)

dx = 0.05
dy = 0.1
dt = 0.2

xmin = 37.0
xmax = 57.0

nchunks = 5

def subset_lab(fdir, savedir, var, nchunks=1, xmin=37.0, xmax=57.0):
    for i in range(nchunks):
        df = xr.open_dataset(os.path.join(fdir, var+'_%d.nc' % i))
        xslice = slice(xmin, xmax)
        new_df = df.sel(x=xslice)
        new_df.to_netcdf(os.path.join(savedir, var+'_%d.nc' % i), 'w')

def rms_calc(fdir, dx=0.05, dy=0.1):
    u_psi_dat = xr.open_mfdataset(os.path.join(fdir, 'u_psi_*.nc'), combine='nested', concat_dim='time')
    u_phi_dat = xr.open_mfdataset(os.path.join(fdir, 'u_phi_*.nc'), combine='nested', concat_dim='time')
    u_dat = xr.open_mfdataset([os.path.join(fdir, 'u_0.nc'), os.path.join(fdir, 'u_1.nc'), os.path.join(fdir, 'u_2.nc'), os.path.join(fdir, 'u_3.nc'), os.path.join(fdir, 'u_4.nc')], combine='nested', concat_dim='time')

    x = u_dat['x']
    y = u_dat['y']

    u_rec = u_psi_dat['u_psi'] + u_phi_dat['u_phi']
    u = u_dat['u']

    xend = int(54/dx)

    f = open(os.path.join(fdir, 'rms_u.txt'), 'w')
    N = len(x[:xend])*len(y)

    for t in range(len(u)):
        rms = np.sqrt(np.nansum((u[t,:,:xend]-u_rec[t,:,:xend])**2)/N)
        f.write('%f' % rms)
        f.write('\n')
    f.close()

    del u_psi_dat, u_phi_dat, u_dat, u_rec, u
    v_psi_dat = xr.open_mfdataset(os.path.join(fdir, 'v_psi_*.nc'), combine='nested', concat_dim='time')
    v_phi_dat = xr.open_mfdataset(os.path.join(fdir, 'v_phi_*.nc'), combine='nested', concat_dim='time')
    v_dat = xr.open_mfdataset([os.path.join(fdir, 'v_0.nc'), os.path.join(fdir, 'v_1.nc'), os.path.join(fdir, 'v_2.nc'), os.path.join(fdir, 'v_3.nc'), os.path.join(fdir, 'v_4.nc')], combine='nested', concat_dim='time')

    x = v_dat['x']
    y = v_dat['y']

    v_rec = v_psi_dat['v_psi'] + v_phi_dat['v_phi']
    v = v_dat['v']

    f = open(os.path.join(fdir, 'rms_v.txt'), 'w')
    N = len(x[:xend])*len(y)

    for t in range(len(v)):
        rms = np.sqrt(np.nansum((v[t,:,:xend]-v_rec[t,:,:xend])**2)/N)
        f.write('%f' % rms)
        f.write('\n')
    f.close()


fp.output2netcdf(fdir, savefulldir, dx, dy, dt, 'eta', nchunks=nchunks)
fp.output2netcdf(fdir, savefulldir, dx, dy, dt, 'u', nchunks=nchunks)
fp.output2netcdf(fdir, savefulldir, dx, dy, dt, 'v', nchunks=nchunks)
fp.output2netcdf(fdir, savefulldir, dx, dy, dt, 'mask', nchunks=nchunks)
fp.output2netcdf(fdir, savefulldir, dx, dy, dt, 'nubrk', nchunks=nchunks)
fp.veldec2netcdf(savefulldir, nchunks=nchunks)

dep = np.loadtxt(os.path.join(fdir, 'dep.out'))
new_dep = dep[:,int(xmin/dx):int(xmax/dx)+1]
np.savetxt(os.path.join(savedir, 'dep.out'), new_dep)

subset_lab(savefulldir, savedir, 'eta', nchunks=nchunks)
subset_lab(savefulldir, savedir, 'mask', nchunks=nchunks)
subset_lab(savefulldir, savedir, 'u', nchunks=nchunks)
subset_lab(savefulldir, savedir, 'v', nchunks=nchunks)
subset_lab(savefulldir, savedir, 'nubrk', nchunks=nchunks)
subset_lab(savefulldir, savedir, 'u_phi', nchunks=nchunks)
subset_lab(savefulldir, savedir, 'v_phi', nchunks=nchunks)
subset_lab(savefulldir, savedir, 'u_psi', nchunks=nchunks)
subset_lab(savefulldir, savedir, 'v_psi', nchunks=nchunks)

fp.vorticity2netcdf(savedir, nchunks=nchunks)
fp.fbr2netcdf(savedir, nchunks=nchunks)
fp.crest2netcdf(savedir, nchunks=nchunks)

def compute_creststats(x, y, crests, fbr):
	crestlen_total = []
	crestpertime = []
	crestend_min_x_total = []
	crestend_max_x_total = []
	crestend_min_y_total = []
	crestend_max_y_total = []
	crest_total_fbr_std = []
	crest_total_fbr_abs = []
	crest_total_fbr_sq = []
	crest_total_fbr_mean = [] 
	crestends = []

	T = len(fbr)
	for t in range(T):
		#num = int(np.max(crests[t,:,:]).values)
		num = int(np.max(crests[t,:,:]))
		crestend_min_x, crestend_max_x, crestend_min_y, crestend_max_y, crestlen, crestlen, crestfbr_std, crestfbr_abs, crestfbr_sq, crestfbr_mean = mod_utils.calc_crestlen_fbr(x, y, num, crests[t,:,:], fbr[t,:,:])
		crestends.append(num*2)
		for j in range(len(crestlen)):
			crestlen_total.append(crestlen[j])
			crestend_min_x_total.append(crestend_min_x[j])
			crestend_min_y_total.append(crestend_min_y[j])
			crestend_max_x_total.append(crestend_max_x[j])
			crestend_max_y_total.append(crestend_max_y[j])
			crestpertime.append(t)
			crest_total_fbr_std.append(crestfbr_std[j])
			crest_total_fbr_abs.append(crestfbr_abs[j])
			crest_total_fbr_sq.append(crestfbr_sq[j])
			crest_total_fbr_mean.append(crestfbr_mean[j])
	return crestlen_total, crestend_min_x_total, crestend_min_y_total, crestend_max_x_total, crestend_max_y_total, crest_total_fbr_std, crest_total_fbr_abs, crest_total_fbr_sq, crest_total_fbr_mean, crestends, crestpertime

def crest_stats_txt(filename, crestlen, minx, miny, maxx, maxy, fbrstd, fbrabs, fbrsq, fbrmean, time):
	f = open(filename, 'w')
	f.write("Crest length, Min x position, Min y position, Max x posiition, Max y position, std(fbr), abs(fbr), sq(fbr), number of ends, time index\n")
	for i in range(len(crestlen)):
		f.write('%f, ' % crestlen[i])
		f.write('%f, ' % minx[i])
		f.write('%f, ' % miny[i])
		f.write('%f, ' % maxx[i])
		f.write('%f, ' % maxy[i])
		f.write('%f, ' % fbrstd[i])
		f.write('%f, ' % fbrabs[i])
		f.write('%f, ' % fbrsq[i])
		f.write('%f, ' % fbrmean[i])
		f.write('%f' % time[i])
		f.write('\n')
	f.close()


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


def to_textfile(filename, time, crestlen, avgx, minx, maxx, miny, maxy, fbr, area):
	f = open(filename, "w")
	for i in range(len(time)):
		f.write('%f, ' % time[i])
		f.write('%f, ' % crestlen[i])
		f.write('%f, ' % avgx[i])
		f.write('%f, ' % minx[i])
		f.write('%f, ' % maxx[i])
		f.write('%f, ' % miny[i])
		f.write('%f, ' % maxy[i])
		f.write('%f, ' % fbr[i])
		f.write('%f' % area[i])
		f.write('\n')
	f.close()
	return

x = xr.open_mfdataset(os.path.join(savedir, 'fbrx_*.nc'), combine='nested', concat_dim='time')['x']
y = xr.open_mfdataset(os.path.join(savedir, 'fbrx_*.nc'), combine='nested', concat_dim='time')['y']

flist = [os.path.join(savedir, 'fbrx_1.nc'), os.path.join(savedir, 'fbrx_2.nc'), os.path.join(savedir, 'fbrx_3.nc'), os.path.join(savedir, 'fbrx_4.nc')]
fbrx = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbrx']
flist = [os.path.join(savedir, 'fbry_1.nc'), os.path.join(savedir, 'fbry_2.nc'), os.path.join(savedir, 'fbry_3.nc'), os.path.join(savedir, 'fbry_4.nc')]
fbry = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['fbry']

fbr = np.gradient(fbry, 0.05, axis=2) - np.gradient(fbrx, 0.1, axis=1)
del fbrx, fbry

flist = [os.path.join(savedir, 'crest_1.nc'), os.path.join(savedir, 'crest_2.nc'), os.path.join(savedir, 'crest_3.nc'), os.path.join(savedir, 'crest_4.nc')]
crests = xr.open_mfdataset(flist, combine='nested', concat_dim='time')['labels']

if 0:
	crestlen_total, crestend_min_x_total, crestend_min_y_total, crestend_max_x_total, crestend_max_y_total, crest_total_fbr_std, crest_total_fbr_abs, crest_total_fbr_sq, crest_total_fbr_mean, crestends, crestpertime  = compute_creststats(x, y, crests, fbr)

	crest_stats_txt(os.path.join(savedir, 'crest_stats.txt'), crestlen_total, crestend_min_x_total, crestend_min_y_total, crestend_max_x_total, crestend_max_y_total, crest_total_fbr_std, crest_total_fbr_abs, crest_total_fbr_sq, crest_total_fbr_mean, crestpertime)
	np.savetxt(os.path.join(savedir, 'crest_ends.txt'), crestends, delimiter=',')

time_all, crestlen_all, avgx_all, minx_all, maxx_all, miny_all, maxy_all, \
fbr_std_all, fbr_abs_all, fbr_sq_all, fbr_mean_all, crestends_all, areacrest_all = \
    load_creststats(fdir, crests, os.path.join(savedir, 'crest_stats.txt'), os.path.join(savedir, 'crest_ends.txt'))

to_textfile(os.path.join(savedir, 'crest_info_all.txt'), time_all, crestlen_all, avgx_all, \
			minx_all, maxx_all, miny_all, maxy_all, fbr_abs_all, areacrest_all)

## subset crests and fbr to only lab regions
# lab 1
labwidth = 26.5
ystart = 0.5
yend = ystart+labwidth
yind = np.where((y>ystart)&(y<=yend))[0]

y1 = y[yind]
fbr1 = fbr[:,yind,:]
crests1 = crests.sel(y=slice(ystart, yend-1))

if 0:
	crestlen_total, crestend_min_x_total, crestend_min_y_total, crestend_max_x_total, crestend_max_y_total, crest_total_fbr_std, crest_total_fbr_abs, crest_total_fbr_sq, crest_total_fbr_mean, crestends, crestpertime  = compute_creststats(x, y1, crests1, fbr1)

	crest_stats_txt(os.path.join(savedir, 'crest_stats_lab1.txt'), crestlen_total, crestend_min_x_total, crestend_min_y_total, crestend_max_x_total, crestend_max_y_total, crest_total_fbr_std, crest_total_fbr_abs, crest_total_fbr_sq, crest_total_fbr_mean, crestpertime)
	np.savetxt(os.path.join(savedir, 'crest_ends_lab1.txt'), crestends, delimiter=',')

time_all, crestlen_all, avgx_all, minx_all, maxx_all, miny_all, maxy_all, \
fbr_std_all, fbr_abs_all, fbr_sq_all, fbr_mean_all, crestends_all, areacrest_all = \
    load_creststats(savedir, crests, os.path.join(savedir, 'crest_stats_lab1.txt'), os.path.join(savedir, 'crest_ends_lab1.txt'))

to_textfile(os.path.join(savedir, 'crest_info_lab1.txt'), time_all, crestlen_all, avgx_all, \
			minx_all, maxx_all, miny_all, maxy_all, fbr_abs_all, areacrest_all)

# lab 2
ystart = 28
yend = ystart+labwidth
yind = np.where((y>ystart)&(y<=yend))[0]

fbr1 = fbr[:,yind,:]
crests1 = crests.sel(y=slice(ystart, yend-1))

if 0:
	crestlen_total, crestend_min_x_total, crestend_min_y_total, crestend_max_x_total, crestend_max_y_total, crest_total_fbr_std, crest_total_fbr_abs, crest_total_fbr_sq, crest_total_fbr_mean, crestends, crestpertime  = compute_creststats(x, y1, crests1, fbr1)

	crest_stats_txt(os.path.join(savedir, 'crest_stats_lab2.txt'), crestlen_total, crestend_min_x_total, crestend_min_y_total, crestend_max_x_total, crestend_max_y_total, crest_total_fbr_std, crest_total_fbr_abs, crest_total_fbr_sq, crest_total_fbr_mean, crestpertime)
	np.savetxt(os.path.join(savedir, 'crest_ends_lab2.txt'), crestends, delimiter=',')

time_all, crestlen_all, avgx_all, minx_all, maxx_all, miny_all, maxy_all, \
fbr_std_all, fbr_abs_all, fbr_sq_all, fbr_mean_all, crestends_all, areacrest_all = \
    load_creststats(savedir, crests, os.path.join(savedir, 'crest_stats_lab2.txt'), os.path.join(savedir, 'crest_ends_lab2.txt'))

to_textfile(os.path.join(savedir, 'crest_info_lab2.txt'), time_all, crestlen_all, avgx_all, \
			minx_all, maxx_all, miny_all, maxy_all, fbr_abs_all, areacrest_all)

rms_calc(savedir)

