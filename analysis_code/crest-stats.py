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

start = datetime.datetime.now()

plt.ion()
#plt.style.use('ggplot')
plt.style.use('classic')

dx = 0.05
dy = 0.1
arealim = 0.375 
plotsavedir = os.path.join('/gscratch', 'nearshore', 'enuss', 'lab_runs_y550', 'postprocessing', 'plots')

#####
labwidth = 26.5
ystart1 = 0.5
yend1 = ystart1+labwidth

ystart2 = 28
yend2 = ystart2+labwidth

#####

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

def find_finite(var):
    return var[np.where(np.isfinite(var)==True)]

def remove_small_crests(var, areas, arealim):
    return var[areas>arealim]

def restrict_cross_crests(var, avgx, sz=15+22, shore=31.5+22):
    return var[(avgx>sz)&(avgx<shore)]

def subset_time(var, time, tstart=0, tend=10*60/0.2):
    return var[(time>tstart)&(time<tend)]

def subsample_by_area(time, area, crestlen, fbr_abs, avgx, miny, maxy, arealim=0.375):
    time_sub = remove_small_crests(time, area, arealim)
    crestlen_sub = remove_small_crests(crestlen, area, arealim)
    fbr_abs_sub = remove_small_crests(fbr_abs, area, arealim)
    avgx_sub = remove_small_crests(avgx, area, arealim)
    miny_sub = remove_small_crests(miny, area, arealim)
    maxy_sub = remove_small_crests(maxy, area, arealim)
    return time_sub, crestlen_sub, fbr_abs_sub, avgx_sub, miny_sub, maxy_sub 

def subsample_by_time(tstart, tend, time, crestlen, fbr_abs, avgx):
    time_sub = subset_time(time, time, tstart, tend)
    crestlen_sub = subset_time(crestlen, time, tstart, tend)
    fbr_abs_sub = subset_time(fbr_abs, time, tstart, tend)
    avgx_sub = subset_time(avgx, time, tstart, tend)
    return time_sub, crestlen_sub, fbr_abs_sub, avgx_sub

def crest_ends(time, crest_ymin, crest_ymax, ymin, ymax, dy=0.2):
    crestends = []
    for i in range(len(time)):
        if crest_ymin[i] <= ymin - dy:
            crestends.append(1)
        if crest_ymax[i] >= ymax - dy:
            crestends.append(1)
        else:
            crestends.append(2)
    return np.asarray(crestends)

def var_pertime(var, time):
    T = np.max(time)
    Tmin = np.min(time)
    var_sum = []
    for i in range(int(Tmin), int(T)):
        ind = np.where(time==i)[0]
        if len(ind)==0:
            var_sum.append(0)
        else:
            tmp = var[ind]
            var_sum.append(np.nansum(tmp))
    return np.asarray(var_sum)

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

#####
#shore = np.array([32.9, 32.75, 32.55, 32.6, 32.6, 32.7, 32.65, 32.7])+22
shore = np.ones(8)*31.5+22

rundir = 'hmo25_dir1_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

time_dir0_all, crestlen_dir0_all, avgx_dir0_all, minx_dir0_all, maxx_dir0_all, miny_dir0_all, maxy_dir0_all, fbr_abs_dir0_all, areacrest_dir0_all = np.loadtxt(os.path.join(fdir, 'crest_info_all.txt'), delimiter=',').T
time_dir0_lab1, crestlen_dir0_lab1, avgx_dir0_lab1, minx_dir0_lab1, maxx_dir0_lab1, miny_dir0_lab1, maxy_dir0_lab1, fbr_abs_dir0_lab1, areacrest_dir0_lab1 = np.loadtxt(os.path.join(fdir, 'crest_info_lab1.txt'), delimiter=',').T
time_dir0_lab2, crestlen_dir0_lab2, avgx_dir0_lab2, minx_dir0_lab2, maxx_dir0_lab2, miny_dir0_lab2, maxy_dir0_lab2, fbr_abs_dir0_lab2, areacrest_dir0_lab2 = np.loadtxt(os.path.join(fdir, 'crest_info_lab2.txt'), delimiter=',').T

######## process crest stats ##########
# restrict cross-shore region
time_dir0_all_cross = restrict_cross_crests(time_dir0_all, avgx_dir0_all, shore=shore[0])
areacrest_dir0_all_cross = restrict_cross_crests(areacrest_dir0_all, avgx_dir0_all, shore=shore[0])
crestlen_dir0_all_cross = restrict_cross_crests(crestlen_dir0_all, avgx_dir0_all, shore=shore[0])
fbr_abs_dir0_all_cross = restrict_cross_crests(fbr_abs_dir0_all, avgx_dir0_all, shore=shore[0])
avgx_dir0_all_cross = restrict_cross_crests(avgx_dir0_all, avgx_dir0_all, shore=shore[0])
miny_dir0_all_cross = restrict_cross_crests(miny_dir0_all, avgx_dir0_all, shore=shore[0])
maxy_dir0_all_cross = restrict_cross_crests(maxy_dir0_all, avgx_dir0_all, shore=shore[0])

time_dir0_lab1_cross = restrict_cross_crests(time_dir0_lab1, avgx_dir0_lab1, shore=shore[0])
areacrest_dir0_lab1_cross = restrict_cross_crests(areacrest_dir0_lab1, avgx_dir0_lab1, shore=shore[0])
crestlen_dir0_lab1_cross = restrict_cross_crests(crestlen_dir0_lab1, avgx_dir0_lab1, shore=shore[0])
fbr_abs_dir0_lab1_cross = restrict_cross_crests(fbr_abs_dir0_lab1, avgx_dir0_lab1, shore=shore[0])
avgx_dir0_lab1_cross = restrict_cross_crests(avgx_dir0_lab1, avgx_dir0_lab1, shore=shore[0])
miny_dir0_lab1_cross = restrict_cross_crests(miny_dir0_lab1, avgx_dir0_lab1, shore=shore[0])
maxy_dir0_lab1_cross = restrict_cross_crests(maxy_dir0_lab1, avgx_dir0_lab1, shore=shore[0])

time_dir0_lab2_cross = restrict_cross_crests(time_dir0_lab2, avgx_dir0_lab2, shore=shore[0])
areacrest_dir0_lab2_cross = restrict_cross_crests(areacrest_dir0_lab2, avgx_dir0_lab2, shore=shore[0])
crestlen_dir0_lab2_cross = restrict_cross_crests(crestlen_dir0_lab2, avgx_dir0_lab2, shore=shore[0])
fbr_abs_dir0_lab2_cross = restrict_cross_crests(fbr_abs_dir0_lab2, avgx_dir0_lab2, shore=shore[0])
avgx_dir0_lab2_cross = restrict_cross_crests(avgx_dir0_lab2, avgx_dir0_lab2, shore=shore[0])
miny_dir0_lab2_cross = restrict_cross_crests(miny_dir0_lab2, avgx_dir0_lab2, shore=shore[0])
maxy_dir0_lab2_cross = restrict_cross_crests(maxy_dir0_lab2, avgx_dir0_lab2, shore=shore[0])

# remove crests below area threshold 
time_dir0_all_subarea, crestlen_dir0_all_subarea, fbr_abs_dir0_all_subarea, avgx_dir0_all_subarea, miny_dir0_all_subarea, maxy_dir0_all_subarea = \
    subsample_by_area(time_dir0_all_cross, areacrest_dir0_all_cross, crestlen_dir0_all_cross, fbr_abs_dir0_all_cross, avgx_dir0_all_cross, miny_dir0_all_cross, maxy_dir0_all_cross)
time_dir0_lab1_subarea, crestlen_dir0_lab1_subarea, fbr_abs_dir0_lab1_subarea, avgx_dir0_lab1_subarea, miny_dir0_lab1_subarea, maxy_dir0_lab1_subarea = \
    subsample_by_area(time_dir0_lab1_cross, areacrest_dir0_lab1_cross, crestlen_dir0_lab1_cross, fbr_abs_dir0_lab1_cross, avgx_dir0_lab1_cross, miny_dir0_lab1_cross, maxy_dir0_lab1_cross)
time_dir0_lab2_subarea, crestlen_dir0_lab2_subarea, fbr_abs_dir0_lab2_subarea, avgx_dir0_lab2_subarea, miny_dir0_lab2_subarea, maxy_dir0_lab2_subarea = \
    subsample_by_area(time_dir0_lab2_cross, areacrest_dir0_lab2_cross, crestlen_dir0_lab2_cross, fbr_abs_dir0_lab2_cross, avgx_dir0_lab2_cross, miny_dir0_lab2_cross, maxy_dir0_lab2_cross)

crestends_dir0_all = crest_ends(time_dir0_all_subarea, miny_dir0_all_subarea, maxy_dir0_all_subarea, ymin=0, ymax=55)
crestends_dir0_lab1 = crest_ends(time_dir0_lab1_subarea, miny_dir0_lab1_subarea, maxy_dir0_lab1_subarea, ymin=0.6, ymax=26.1)
crestends_dir0_lab2 = crest_ends(time_dir0_lab2_subarea, miny_dir0_lab2_subarea, maxy_dir0_lab2_subarea, ymin=0.6, ymax=26.1)

crestends_pertime_dir0_all = var_pertime(crestends_dir0_all, time_dir0_all_subarea)
crestends_pertime_dir0_lab1 = var_pertime(crestends_dir0_lab1, time_dir0_lab1_subarea)
crestends_pertime_dir0_lab2 = var_pertime(crestends_dir0_lab2, time_dir0_lab2_subarea)

fbr_abs_pertime_dir0_all = var_pertime(fbr_abs_dir0_all_subarea, time_dir0_all_subarea)
fbr_abs_pertime_dir0_lab1 = var_pertime(fbr_abs_dir0_lab1_subarea, time_dir0_lab1_subarea)
fbr_abs_pertime_dir0_lab2 = var_pertime(fbr_abs_dir0_lab2_subarea, time_dir0_lab2_subarea)

# calculate stats for first 10 min window
tstart = 0; tend = 600
time_dir0_all_t1, crestlen_dir0_all_t1, fbr_abs_dir0_all_t1, avgx_dir0_all_t1 = \
    subsample_by_time(tstart, tend, time_dir0_all_subarea, crestlen_dir0_all_subarea, fbr_abs_dir0_all_subarea, avgx_dir0_all_subarea)
time_dir0_lab1_t1, crestlen_dir0_lab1_t1, fbr_abs_dir0_lab1_t1, avgx_dir0_lab1_t1 = \
    subsample_by_time(tstart, tend, time_dir0_lab1_subarea, crestlen_dir0_lab1_subarea, fbr_abs_dir0_lab1_subarea, avgx_dir0_lab1_subarea)
time_dir0_lab2_t1, crestlen_dir0_lab2_t1, fbr_abs_dir0_lab2_t1, avgx_dir0_lab2_t1 = \
    subsample_by_time(tstart, tend, time_dir0_lab2_subarea, crestlen_dir0_lab2_subarea, fbr_abs_dir0_lab2_subarea, avgx_dir0_lab2_subarea)

crestends_dir0_all_t1 = subset_time(crestends_dir0_all, time_dir0_all_subarea, tstart, tend)
crestends_dir0_lab1_t1 = subset_time(crestends_dir0_lab1, time_dir0_lab1_subarea, tstart, tend)
crestends_dir0_lab2_t1 = subset_time(crestends_dir0_lab2, time_dir0_lab2_subarea, tstart, tend)

crestends_pertime_dir0_all_t1 = var_pertime(crestends_dir0_all_t1, time_dir0_all_t1)
crestends_pertime_dir0_lab1_t1 = var_pertime(crestends_dir0_lab1_t1, time_dir0_lab1_t1)
crestends_pertime_dir0_lab2_t1 = var_pertime(crestends_dir0_lab2_t1, time_dir0_lab2_t1)

fbr_abs_pertime_dir0_all_t1 = var_pertime(fbr_abs_dir0_all_t1, time_dir0_all_t1)
fbr_abs_pertime_dir0_lab1_t1 = var_pertime(fbr_abs_dir0_lab1_t1, time_dir0_lab1_t1)
fbr_abs_pertime_dir0_lab2_t1 = var_pertime(fbr_abs_dir0_lab2_t1, time_dir0_lab2_t1)

# calculate stats for second 10 min window
tstart = 600; tend = 1200
time_dir0_all_t2, crestlen_dir0_all_t2, fbr_abs_dir0_all_t2, avgx_dir0_all_t2 = \
    subsample_by_time(tstart, tend, time_dir0_all_subarea, crestlen_dir0_all_subarea, fbr_abs_dir0_all_subarea, avgx_dir0_all_subarea)
time_dir0_lab1_t2, crestlen_dir0_lab1_t2, fbr_abs_dir0_lab1_t2, avgx_dir0_lab1_t2 = \
    subsample_by_time(tstart, tend, time_dir0_lab1_subarea, crestlen_dir0_lab1_subarea, fbr_abs_dir0_lab1_subarea, avgx_dir0_lab1_subarea)
time_dir0_lab2_t2, crestlen_dir0_lab2_t2, fbr_abs_dir0_lab2_t2, avgx_dir0_lab2_t2 = \
    subsample_by_time(tstart, tend, time_dir0_lab2_subarea, crestlen_dir0_lab2_subarea, fbr_abs_dir0_lab2_subarea, avgx_dir0_lab2_subarea)

crestends_dir0_all_t2 = subset_time(crestends_dir0_all, time_dir0_all_subarea, tstart, tend)
crestends_dir0_lab1_t2 = subset_time(crestends_dir0_lab1, time_dir0_lab1_subarea, tstart, tend)
crestends_dir0_lab2_t2 = subset_time(crestends_dir0_lab2, time_dir0_lab2_subarea, tstart, tend)

crestends_pertime_dir0_all_t2 = var_pertime(crestends_dir0_all_t2, time_dir0_all_t2)
crestends_pertime_dir0_lab1_t2 = var_pertime(crestends_dir0_lab1_t2, time_dir0_lab1_t2)
crestends_pertime_dir0_lab2_t2 = var_pertime(crestends_dir0_lab2_t2, time_dir0_lab2_t2)

fbr_abs_pertime_dir0_all_t2 = var_pertime(fbr_abs_dir0_all_t2, time_dir0_all_t2)
fbr_abs_pertime_dir0_lab1_t2 = var_pertime(fbr_abs_dir0_lab1_t2, time_dir0_lab1_t2)
fbr_abs_pertime_dir0_lab2_t2 = var_pertime(fbr_abs_dir0_lab2_t2, time_dir0_lab2_t2)

##################################################################
rundir = 'hmo25_dir5_tp2_ntheta15'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

time_dir5_all, crestlen_dir5_all, avgx_dir5_all, minx_dir5_all, maxx_dir5_all, miny_dir5_all, maxy_dir5_all, fbr_abs_dir5_all, areacrest_dir5_all = np.loadtxt(os.path.join(fdir, 'crest_info_all.txt'), delimiter=',').T
time_dir5_lab1, crestlen_dir5_lab1, avgx_dir5_lab1, minx_dir5_lab1, maxx_dir5_lab1, miny_dir5_lab1, maxy_dir5_lab1, fbr_abs_dir5_lab1, areacrest_dir5_lab1 = np.loadtxt(os.path.join(fdir, 'crest_info_lab1.txt'), delimiter=',').T
time_dir5_lab2, crestlen_dir5_lab2, avgx_dir5_lab2, minx_dir5_lab2, maxx_dir5_lab2, miny_dir5_lab2, maxy_dir5_lab2, fbr_abs_dir5_lab2, areacrest_dir5_lab2 = np.loadtxt(os.path.join(fdir, 'crest_info_lab2.txt'), delimiter=',').T

######## process crest stats ##########
# restrict cross-shore region
time_dir5_all_cross = restrict_cross_crests(time_dir5_all, avgx_dir5_all, shore=shore[1])
areacrest_dir5_all_cross = restrict_cross_crests(areacrest_dir5_all, avgx_dir5_all, shore=shore[1])
crestlen_dir5_all_cross = restrict_cross_crests(crestlen_dir5_all, avgx_dir5_all, shore=shore[1])
fbr_abs_dir5_all_cross = restrict_cross_crests(fbr_abs_dir5_all, avgx_dir5_all, shore=shore[1])
avgx_dir5_all_cross = restrict_cross_crests(avgx_dir5_all, avgx_dir5_all, shore=shore[1])
miny_dir5_all_cross = restrict_cross_crests(miny_dir5_all, avgx_dir5_all, shore=shore[1])
maxy_dir5_all_cross = restrict_cross_crests(maxy_dir5_all, avgx_dir5_all, shore=shore[1])

time_dir5_lab1_cross = restrict_cross_crests(time_dir5_lab1, avgx_dir5_lab1, shore=shore[1])
areacrest_dir5_lab1_cross = restrict_cross_crests(areacrest_dir5_lab1, avgx_dir5_lab1, shore=shore[1])
crestlen_dir5_lab1_cross = restrict_cross_crests(crestlen_dir5_lab1, avgx_dir5_lab1, shore=shore[1])
fbr_abs_dir5_lab1_cross = restrict_cross_crests(fbr_abs_dir5_lab1, avgx_dir5_lab1, shore=shore[1])
avgx_dir5_lab1_cross = restrict_cross_crests(avgx_dir5_lab1, avgx_dir5_lab1, shore=shore[1])
miny_dir5_lab1_cross = restrict_cross_crests(miny_dir5_lab1, avgx_dir5_lab1, shore=shore[1])
maxy_dir5_lab1_cross = restrict_cross_crests(maxy_dir5_lab1, avgx_dir5_lab1, shore=shore[1])

time_dir5_lab2_cross = restrict_cross_crests(time_dir5_lab2, avgx_dir5_lab2, shore=shore[1])
areacrest_dir5_lab2_cross = restrict_cross_crests(areacrest_dir5_lab2, avgx_dir5_lab2, shore=shore[1])
crestlen_dir5_lab2_cross = restrict_cross_crests(crestlen_dir5_lab2, avgx_dir5_lab2, shore=shore[1])
fbr_abs_dir5_lab2_cross = restrict_cross_crests(fbr_abs_dir5_lab2, avgx_dir5_lab2, shore=shore[1])
avgx_dir5_lab2_cross = restrict_cross_crests(avgx_dir5_lab2, avgx_dir5_lab2, shore=shore[1])
miny_dir5_lab2_cross = restrict_cross_crests(miny_dir5_lab2, avgx_dir5_lab2, shore=shore[1])
maxy_dir5_lab2_cross = restrict_cross_crests(maxy_dir5_lab2, avgx_dir5_lab2, shore=shore[1])

# remove crests below area threshold 
time_dir5_all_subarea, crestlen_dir5_all_subarea, fbr_abs_dir5_all_subarea, avgx_dir5_all_subarea, miny_dir5_all_subarea, maxy_dir5_all_subarea = \
    subsample_by_area(time_dir5_all_cross, areacrest_dir5_all_cross, crestlen_dir5_all_cross, fbr_abs_dir5_all_cross, avgx_dir5_all_cross, miny_dir5_all_cross, maxy_dir5_all_cross)
time_dir5_lab1_subarea, crestlen_dir5_lab1_subarea, fbr_abs_dir5_lab1_subarea, avgx_dir5_lab1_subarea, miny_dir5_lab1_subarea, maxy_dir5_lab1_subarea = \
    subsample_by_area(time_dir5_lab1_cross, areacrest_dir5_lab1_cross, crestlen_dir5_lab1_cross, fbr_abs_dir5_lab1_cross, avgx_dir5_lab1_cross, miny_dir5_lab1_cross, maxy_dir5_lab1_cross)
time_dir5_lab2_subarea, crestlen_dir5_lab2_subarea, fbr_abs_dir5_lab2_subarea, avgx_dir5_lab2_subarea, miny_dir5_lab2_subarea, maxy_dir5_lab2_subarea = \
    subsample_by_area(time_dir5_lab2_cross, areacrest_dir5_lab2_cross, crestlen_dir5_lab2_cross, fbr_abs_dir5_lab2_cross, avgx_dir5_lab2_cross, miny_dir5_lab2_cross, maxy_dir5_lab2_cross)

crestends_dir5_all = crest_ends(time_dir5_all_subarea, miny_dir5_all_subarea, maxy_dir5_all_subarea, ymin=0, ymax=55)
crestends_dir5_lab1 = crest_ends(time_dir5_lab1_subarea, miny_dir5_lab1_subarea, maxy_dir5_lab1_subarea, ymin=0.6, ymax=26.1)
crestends_dir5_lab2 = crest_ends(time_dir5_lab2_subarea, miny_dir5_lab2_subarea, maxy_dir5_lab2_subarea, ymin=0.6, ymax=26.1)

crestends_pertime_dir5_all = var_pertime(crestends_dir5_all, time_dir5_all_subarea)
crestends_pertime_dir5_lab1 = var_pertime(crestends_dir5_lab1, time_dir5_lab1_subarea)
crestends_pertime_dir5_lab2 = var_pertime(crestends_dir5_lab2, time_dir5_lab2_subarea)

fbr_abs_pertime_dir5_all = var_pertime(fbr_abs_dir5_all_subarea, time_dir5_all_subarea)
fbr_abs_pertime_dir5_lab1 = var_pertime(fbr_abs_dir5_lab1_subarea, time_dir5_lab1_subarea)
fbr_abs_pertime_dir5_lab2 = var_pertime(fbr_abs_dir5_lab2_subarea, time_dir5_lab2_subarea)

# calculate stats for first 10 min window
tstart = 0; tend = tstart + 10*60/0.2
time_dir5_all_t1, crestlen_dir5_all_t1, fbr_abs_dir5_all_t1, avgx_dir5_all_t1 = \
    subsample_by_time(tstart, tend, time_dir5_all_subarea, crestlen_dir5_all_subarea, fbr_abs_dir5_all_subarea, avgx_dir5_all_subarea)
time_dir5_lab1_t1, crestlen_dir5_lab1_t1, fbr_abs_dir5_lab1_t1, avgx_dir5_lab1_t1 = \
    subsample_by_time(tstart, tend, time_dir5_lab1_subarea, crestlen_dir5_lab1_subarea, fbr_abs_dir5_lab1_subarea, avgx_dir5_lab1_subarea)
time_dir5_lab2_t1, crestlen_dir5_lab2_t1, fbr_abs_dir5_lab2_t1, avgx_dir5_lab2_t1 = \
    subsample_by_time(tstart, tend, time_dir5_lab2_subarea, crestlen_dir5_lab2_subarea, fbr_abs_dir5_lab2_subarea, avgx_dir5_lab2_subarea)

crestends_dir5_all_t1 = subset_time(crestends_dir5_all, time_dir5_all_subarea, tstart, tend)
crestends_dir5_lab1_t1 = subset_time(crestends_dir5_lab1, time_dir5_lab1_subarea, tstart, tend)
crestends_dir5_lab2_t1 = subset_time(crestends_dir5_lab2, time_dir5_lab2_subarea, tstart, tend)

crestends_pertime_dir5_all_t1 = var_pertime(crestends_dir5_all_t1, time_dir5_all_t1)
crestends_pertime_dir5_lab1_t1 = var_pertime(crestends_dir5_lab1_t1, time_dir5_lab1_t1)
crestends_pertime_dir5_lab2_t1 = var_pertime(crestends_dir5_lab2_t1, time_dir5_lab2_t1)

fbr_abs_pertime_dir5_all_t1 = var_pertime(fbr_abs_dir5_all_t1, time_dir5_all_t1)
fbr_abs_pertime_dir5_lab1_t1 = var_pertime(fbr_abs_dir5_lab1_t1, time_dir5_lab1_t1)
fbr_abs_pertime_dir5_lab2_t1 = var_pertime(fbr_abs_dir5_lab2_t1, time_dir5_lab2_t1)

# calculate stats for second 10 min window
tstart = 10*60/0.2; tend = tstart + 10*60/0.2
time_dir5_all_t2, crestlen_dir5_all_t2, fbr_abs_dir5_all_t2, avgx_dir5_all_t2 = \
    subsample_by_time(tstart, tend, time_dir5_all_subarea, crestlen_dir5_all_subarea, fbr_abs_dir5_all_subarea, avgx_dir5_all_subarea)
time_dir5_lab1_t2, crestlen_dir5_lab1_t2, fbr_abs_dir5_lab1_t2, avgx_dir5_lab1_t2 = \
    subsample_by_time(tstart, tend, time_dir5_lab1_subarea, crestlen_dir5_lab1_subarea, fbr_abs_dir5_lab1_subarea, avgx_dir5_lab1_subarea)
time_dir5_lab2_t2, crestlen_dir5_lab2_t2, fbr_abs_dir5_lab2_t2, avgx_dir5_lab2_t2 = \
    subsample_by_time(tstart, tend, time_dir5_lab2_subarea, crestlen_dir5_lab2_subarea, fbr_abs_dir5_lab2_subarea, avgx_dir5_lab2_subarea)

crestends_dir5_all_t2 = subset_time(crestends_dir5_all, time_dir5_all_subarea, tstart, tend)
crestends_dir5_lab1_t2 = subset_time(crestends_dir5_lab1, time_dir5_lab1_subarea, tstart, tend)
crestends_dir5_lab2_t2 = subset_time(crestends_dir5_lab2, time_dir5_lab2_subarea, tstart, tend)

crestends_pertime_dir5_all_t2 = var_pertime(crestends_dir5_all_t2, time_dir5_all_t2)
crestends_pertime_dir5_lab1_t2 = var_pertime(crestends_dir5_lab1_t2, time_dir5_lab1_t2)
crestends_pertime_dir5_lab2_t2 = var_pertime(crestends_dir5_lab2_t2, time_dir5_lab2_t2)

fbr_abs_pertime_dir5_all_t2 = var_pertime(fbr_abs_dir5_all_t2, time_dir5_all_t2)
fbr_abs_pertime_dir5_lab1_t2 = var_pertime(fbr_abs_dir5_lab1_t2, time_dir5_lab1_t2)
fbr_abs_pertime_dir5_lab2_t2 = var_pertime(fbr_abs_dir5_lab2_t2, time_dir5_lab2_t2)

##################################################################
rundir = 'hmo25_dir10_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

time_dir10_all, crestlen_dir10_all, avgx_dir10_all, minx_dir10_all, maxx_dir10_all, miny_dir10_all, maxy_dir10_all, fbr_abs_dir10_all, areacrest_dir10_all = np.loadtxt(os.path.join(fdir, 'crest_info_all.txt'), delimiter=',').T
time_dir10_lab1, crestlen_dir10_lab1, avgx_dir10_lab1, minx_dir10_lab1, maxx_dir10_lab1, miny_dir10_lab1, maxy_dir10_lab1, fbr_abs_dir10_lab1, areacrest_dir10_lab1 = np.loadtxt(os.path.join(fdir, 'crest_info_lab1.txt'), delimiter=',').T
time_dir10_lab2, crestlen_dir10_lab2, avgx_dir10_lab2, minx_dir10_lab2, maxx_dir10_lab2, miny_dir10_lab2, maxy_dir10_lab2, fbr_abs_dir10_lab2, areacrest_dir10_lab2 = np.loadtxt(os.path.join(fdir, 'crest_info_lab2.txt'), delimiter=',').T

######## process crest stats ##########
# restrict cross-shore region
time_dir10_all_cross = restrict_cross_crests(time_dir10_all, avgx_dir10_all, shore=shore[2])
areacrest_dir10_all_cross = restrict_cross_crests(areacrest_dir10_all, avgx_dir10_all, shore=shore[2])
crestlen_dir10_all_cross = restrict_cross_crests(crestlen_dir10_all, avgx_dir10_all, shore=shore[2])
fbr_abs_dir10_all_cross = restrict_cross_crests(fbr_abs_dir10_all, avgx_dir10_all, shore=shore[2])
avgx_dir10_all_cross = restrict_cross_crests(avgx_dir10_all, avgx_dir10_all, shore=shore[2])
miny_dir10_all_cross = restrict_cross_crests(miny_dir10_all, avgx_dir10_all, shore=shore[2])
maxy_dir10_all_cross = restrict_cross_crests(maxy_dir10_all, avgx_dir10_all, shore=shore[2])

time_dir10_lab1_cross = restrict_cross_crests(time_dir10_lab1, avgx_dir10_lab1, shore=shore[2])
areacrest_dir10_lab1_cross = restrict_cross_crests(areacrest_dir10_lab1, avgx_dir10_lab1, shore=shore[2])
crestlen_dir10_lab1_cross = restrict_cross_crests(crestlen_dir10_lab1, avgx_dir10_lab1, shore=shore[2])
fbr_abs_dir10_lab1_cross = restrict_cross_crests(fbr_abs_dir10_lab1, avgx_dir10_lab1, shore=shore[2])
avgx_dir10_lab1_cross = restrict_cross_crests(avgx_dir10_lab1, avgx_dir10_lab1, shore=shore[2])
miny_dir10_lab1_cross = restrict_cross_crests(miny_dir10_lab1, avgx_dir10_lab1, shore=shore[2])
maxy_dir10_lab1_cross = restrict_cross_crests(maxy_dir10_lab1, avgx_dir10_lab1, shore=shore[2])

time_dir10_lab2_cross = restrict_cross_crests(time_dir10_lab2, avgx_dir10_lab2, shore=shore[2])
areacrest_dir10_lab2_cross = restrict_cross_crests(areacrest_dir10_lab2, avgx_dir10_lab2, shore=shore[2])
crestlen_dir10_lab2_cross = restrict_cross_crests(crestlen_dir10_lab2, avgx_dir10_lab2, shore=shore[2])
fbr_abs_dir10_lab2_cross = restrict_cross_crests(fbr_abs_dir10_lab2, avgx_dir10_lab2, shore=shore[2])
avgx_dir10_lab2_cross = restrict_cross_crests(avgx_dir10_lab2, avgx_dir10_lab2, shore=shore[2])
miny_dir10_lab2_cross = restrict_cross_crests(miny_dir10_lab2, avgx_dir10_lab2, shore=shore[2])
maxy_dir10_lab2_cross = restrict_cross_crests(maxy_dir10_lab2, avgx_dir10_lab2, shore=shore[2])

# remove crests below area threshold 
time_dir10_all_subarea, crestlen_dir10_all_subarea, fbr_abs_dir10_all_subarea, avgx_dir10_all_subarea, miny_dir10_all_subarea, maxy_dir10_all_subarea = \
    subsample_by_area(time_dir10_all_cross, areacrest_dir10_all_cross, crestlen_dir10_all_cross, fbr_abs_dir10_all_cross, avgx_dir10_all_cross, miny_dir10_all_cross, maxy_dir10_all_cross)
time_dir10_lab1_subarea, crestlen_dir10_lab1_subarea, fbr_abs_dir10_lab1_subarea, avgx_dir10_lab1_subarea, miny_dir10_lab1_subarea, maxy_dir10_lab1_subarea = \
    subsample_by_area(time_dir10_lab1_cross, areacrest_dir10_lab1_cross, crestlen_dir10_lab1_cross, fbr_abs_dir10_lab1_cross, avgx_dir10_lab1_cross, miny_dir10_lab1_cross, maxy_dir10_lab1_cross)
time_dir10_lab2_subarea, crestlen_dir10_lab2_subarea, fbr_abs_dir10_lab2_subarea, avgx_dir10_lab2_subarea, miny_dir10_lab2_subarea, maxy_dir10_lab2_subarea = \
    subsample_by_area(time_dir10_lab2_cross, areacrest_dir10_lab2_cross, crestlen_dir10_lab2_cross, fbr_abs_dir10_lab2_cross, avgx_dir10_lab2_cross, miny_dir10_lab2_cross, maxy_dir10_lab2_cross)

crestends_dir10_all = crest_ends(time_dir10_all_subarea, miny_dir10_all_subarea, maxy_dir10_all_subarea, ymin=0, ymax=55)
crestends_dir10_lab1 = crest_ends(time_dir10_lab1_subarea, miny_dir10_lab1_subarea, maxy_dir10_lab1_subarea, ymin=0.6, ymax=26.1)
crestends_dir10_lab2 = crest_ends(time_dir10_lab2_subarea, miny_dir10_lab2_subarea, maxy_dir10_lab2_subarea, ymin=0.6, ymax=26.1)

crestends_pertime_dir10_all = var_pertime(crestends_dir10_all, time_dir10_all_subarea)
crestends_pertime_dir10_lab1 = var_pertime(crestends_dir10_lab1, time_dir10_lab1_subarea)
crestends_pertime_dir10_lab2 = var_pertime(crestends_dir10_lab2, time_dir10_lab2_subarea)

fbr_abs_pertime_dir10_all = var_pertime(fbr_abs_dir10_all_subarea, time_dir10_all_subarea)
fbr_abs_pertime_dir10_lab1 = var_pertime(fbr_abs_dir10_lab1_subarea, time_dir10_lab1_subarea)
fbr_abs_pertime_dir10_lab2 = var_pertime(fbr_abs_dir10_lab2_subarea, time_dir10_lab2_subarea)

# calculate stats for first 10 min window
tstart = 0; tend = tstart + 10*60/0.2
time_dir10_all_t1, crestlen_dir10_all_t1, fbr_abs_dir10_all_t1, avgx_dir10_all_t1 = \
    subsample_by_time(tstart, tend, time_dir10_all_subarea, crestlen_dir10_all_subarea, fbr_abs_dir10_all_subarea, avgx_dir10_all_subarea)
time_dir10_lab1_t1, crestlen_dir10_lab1_t1, fbr_abs_dir10_lab1_t1, avgx_dir10_lab1_t1 = \
    subsample_by_time(tstart, tend, time_dir10_lab1_subarea, crestlen_dir10_lab1_subarea, fbr_abs_dir10_lab1_subarea, avgx_dir10_lab1_subarea)
time_dir10_lab2_t1, crestlen_dir10_lab2_t1, fbr_abs_dir10_lab2_t1, avgx_dir10_lab2_t1 = \
    subsample_by_time(tstart, tend, time_dir10_lab2_subarea, crestlen_dir10_lab2_subarea, fbr_abs_dir10_lab2_subarea, avgx_dir10_lab2_subarea)

crestends_dir10_all_t1 = subset_time(crestends_dir10_all, time_dir10_all_subarea, tstart, tend)
crestends_dir10_lab1_t1 = subset_time(crestends_dir10_lab1, time_dir10_lab1_subarea, tstart, tend)
crestends_dir10_lab2_t1 = subset_time(crestends_dir10_lab2, time_dir10_lab2_subarea, tstart, tend)

crestends_pertime_dir10_all_t1 = var_pertime(crestends_dir10_all_t1, time_dir10_all_t1)
crestends_pertime_dir10_lab1_t1 = var_pertime(crestends_dir10_lab1_t1, time_dir10_lab1_t1)
crestends_pertime_dir10_lab2_t1 = var_pertime(crestends_dir10_lab2_t1, time_dir10_lab2_t1)

fbr_abs_pertime_dir10_all_t1 = var_pertime(fbr_abs_dir10_all_t1, time_dir10_all_t1)
fbr_abs_pertime_dir10_lab1_t1 = var_pertime(fbr_abs_dir10_lab1_t1, time_dir10_lab1_t1)
fbr_abs_pertime_dir10_lab2_t1 = var_pertime(fbr_abs_dir10_lab2_t1, time_dir10_lab2_t1)

# calculate stats for second 10 min window
tstart = 10*60/0.2; tend = tstart + 10*60/0.2
time_dir10_all_t2, crestlen_dir10_all_t2, fbr_abs_dir10_all_t2, avgx_dir10_all_t2 = \
    subsample_by_time(tstart, tend, time_dir10_all_subarea, crestlen_dir10_all_subarea, fbr_abs_dir10_all_subarea, avgx_dir10_all_subarea)
time_dir10_lab1_t2, crestlen_dir10_lab1_t2, fbr_abs_dir10_lab1_t2, avgx_dir10_lab1_t2 = \
    subsample_by_time(tstart, tend, time_dir10_lab1_subarea, crestlen_dir10_lab1_subarea, fbr_abs_dir10_lab1_subarea, avgx_dir10_lab1_subarea)
time_dir10_lab2_t2, crestlen_dir10_lab2_t2, fbr_abs_dir10_lab2_t2, avgx_dir10_lab2_t2 = \
    subsample_by_time(tstart, tend, time_dir10_lab2_subarea, crestlen_dir10_lab2_subarea, fbr_abs_dir10_lab2_subarea, avgx_dir10_lab2_subarea)

crestends_dir10_all_t2 = subset_time(crestends_dir10_all, time_dir10_all_subarea, tstart, tend)
crestends_dir10_lab1_t2 = subset_time(crestends_dir10_lab1, time_dir10_lab1_subarea, tstart, tend)
crestends_dir10_lab2_t2 = subset_time(crestends_dir10_lab2, time_dir10_lab2_subarea, tstart, tend)

crestends_pertime_dir10_all_t2 = var_pertime(crestends_dir10_all_t2, time_dir10_all_t2)
crestends_pertime_dir10_lab1_t2 = var_pertime(crestends_dir10_lab1_t2, time_dir10_lab1_t2)
crestends_pertime_dir10_lab2_t2 = var_pertime(crestends_dir10_lab2_t2, time_dir10_lab2_t2)

fbr_abs_pertime_dir10_all_t2 = var_pertime(fbr_abs_dir10_all_t2, time_dir10_all_t2)
fbr_abs_pertime_dir10_lab1_t2 = var_pertime(fbr_abs_dir10_lab1_t2, time_dir10_lab1_t2)
fbr_abs_pertime_dir10_lab2_t2 = var_pertime(fbr_abs_dir10_lab2_t2, time_dir10_lab2_t2)


##################################################################
rundir = 'hmo25_dir20_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

time_dir20_all, crestlen_dir20_all, avgx_dir20_all, minx_dir20_all, maxx_dir20_all, miny_dir20_all, maxy_dir20_all, fbr_abs_dir20_all, areacrest_dir20_all = np.loadtxt(os.path.join(fdir, 'crest_info_all.txt'), delimiter=',').T
time_dir20_lab1, crestlen_dir20_lab1, avgx_dir20_lab1, minx_dir20_lab1, maxx_dir20_lab1, miny_dir20_lab1, maxy_dir20_lab1, fbr_abs_dir20_lab1, areacrest_dir20_lab1 = np.loadtxt(os.path.join(fdir, 'crest_info_lab1.txt'), delimiter=',').T
time_dir20_lab2, crestlen_dir20_lab2, avgx_dir20_lab2, minx_dir20_lab2, maxx_dir20_lab2, miny_dir20_lab2, maxy_dir20_lab2, fbr_abs_dir20_lab2, areacrest_dir20_lab2 = np.loadtxt(os.path.join(fdir, 'crest_info_lab2.txt'), delimiter=',').T

######## process crest stats ##########
# restrict cross-shore region
time_dir20_all_cross = restrict_cross_crests(time_dir20_all, avgx_dir20_all, shore=shore[3])
areacrest_dir20_all_cross = restrict_cross_crests(areacrest_dir20_all, avgx_dir20_all, shore=shore[3])
crestlen_dir20_all_cross = restrict_cross_crests(crestlen_dir20_all, avgx_dir20_all, shore=shore[3])
fbr_abs_dir20_all_cross = restrict_cross_crests(fbr_abs_dir20_all, avgx_dir20_all, shore=shore[3])
avgx_dir20_all_cross = restrict_cross_crests(avgx_dir20_all, avgx_dir20_all, shore=shore[3])
miny_dir20_all_cross = restrict_cross_crests(miny_dir20_all, avgx_dir20_all, shore=shore[3])
maxy_dir20_all_cross = restrict_cross_crests(maxy_dir20_all, avgx_dir20_all, shore=shore[3])

time_dir20_lab1_cross = restrict_cross_crests(time_dir20_lab1, avgx_dir20_lab1, shore=shore[3])
areacrest_dir20_lab1_cross = restrict_cross_crests(areacrest_dir20_lab1, avgx_dir20_lab1, shore=shore[3])
crestlen_dir20_lab1_cross = restrict_cross_crests(crestlen_dir20_lab1, avgx_dir20_lab1, shore=shore[3])
fbr_abs_dir20_lab1_cross = restrict_cross_crests(fbr_abs_dir20_lab1, avgx_dir20_lab1, shore=shore[3])
avgx_dir20_lab1_cross = restrict_cross_crests(avgx_dir20_lab1, avgx_dir20_lab1, shore=shore[3])
miny_dir20_lab1_cross = restrict_cross_crests(miny_dir20_lab1, avgx_dir20_lab1, shore=shore[3])
maxy_dir20_lab1_cross = restrict_cross_crests(maxy_dir20_lab1, avgx_dir20_lab1, shore=shore[3])

time_dir20_lab2_cross = restrict_cross_crests(time_dir20_lab2, avgx_dir20_lab2, shore=shore[3])
areacrest_dir20_lab2_cross = restrict_cross_crests(areacrest_dir20_lab2, avgx_dir20_lab2, shore=shore[3])
crestlen_dir20_lab2_cross = restrict_cross_crests(crestlen_dir20_lab2, avgx_dir20_lab2, shore=shore[3])
fbr_abs_dir20_lab2_cross = restrict_cross_crests(fbr_abs_dir20_lab2, avgx_dir20_lab2, shore=shore[3])
avgx_dir20_lab2_cross = restrict_cross_crests(avgx_dir20_lab2, avgx_dir20_lab2, shore=shore[3])
miny_dir20_lab2_cross = restrict_cross_crests(miny_dir20_lab2, avgx_dir20_lab2, shore=shore[3])
maxy_dir20_lab2_cross = restrict_cross_crests(maxy_dir20_lab2, avgx_dir20_lab2, shore=shore[3])

# remove crests below area threshold 
time_dir20_all_subarea, crestlen_dir20_all_subarea, fbr_abs_dir20_all_subarea, avgx_dir20_all_subarea, miny_dir20_all_subarea, maxy_dir20_all_subarea = \
    subsample_by_area(time_dir20_all_cross, areacrest_dir20_all_cross, crestlen_dir20_all_cross, fbr_abs_dir20_all_cross, avgx_dir20_all_cross, miny_dir20_all_cross, maxy_dir20_all_cross)
time_dir20_lab1_subarea, crestlen_dir20_lab1_subarea, fbr_abs_dir20_lab1_subarea, avgx_dir20_lab1_subarea, miny_dir20_lab1_subarea, maxy_dir20_lab1_subarea = \
    subsample_by_area(time_dir20_lab1_cross, areacrest_dir20_lab1_cross, crestlen_dir20_lab1_cross, fbr_abs_dir20_lab1_cross, avgx_dir20_lab1_cross, miny_dir20_lab1_cross, maxy_dir20_lab1_cross)
time_dir20_lab2_subarea, crestlen_dir20_lab2_subarea, fbr_abs_dir20_lab2_subarea, avgx_dir20_lab2_subarea, miny_dir20_lab2_subarea, maxy_dir20_lab2_subarea = \
    subsample_by_area(time_dir20_lab2_cross, areacrest_dir20_lab2_cross, crestlen_dir20_lab2_cross, fbr_abs_dir20_lab2_cross, avgx_dir20_lab2_cross, miny_dir20_lab2_cross, maxy_dir20_lab2_cross)

crestends_dir20_all = crest_ends(time_dir20_all_subarea, miny_dir20_all_subarea, maxy_dir20_all_subarea, ymin=0, ymax=55)
crestends_dir20_lab1 = crest_ends(time_dir20_lab1_subarea, miny_dir20_lab1_subarea, maxy_dir20_lab1_subarea, ymin=0.6, ymax=26.1)
crestends_dir20_lab2 = crest_ends(time_dir20_lab2_subarea, miny_dir20_lab2_subarea, maxy_dir20_lab2_subarea, ymin=0.6, ymax=26.1)

crestends_pertime_dir20_all = var_pertime(crestends_dir20_all, time_dir20_all_subarea)
crestends_pertime_dir20_lab1 = var_pertime(crestends_dir20_lab1, time_dir20_lab1_subarea)
crestends_pertime_dir20_lab2 = var_pertime(crestends_dir20_lab2, time_dir20_lab2_subarea)

fbr_abs_pertime_dir20_all = var_pertime(fbr_abs_dir20_all_subarea, time_dir20_all_subarea)
fbr_abs_pertime_dir20_lab1 = var_pertime(fbr_abs_dir20_lab1_subarea, time_dir20_lab1_subarea)
fbr_abs_pertime_dir20_lab2 = var_pertime(fbr_abs_dir20_lab2_subarea, time_dir20_lab2_subarea)

# calculate stats for first 10 min window
tstart = 0; tend = tstart + 10*60/0.2
time_dir20_all_t1, crestlen_dir20_all_t1, fbr_abs_dir20_all_t1, avgx_dir20_all_t1 = \
    subsample_by_time(tstart, tend, time_dir20_all_subarea, crestlen_dir20_all_subarea, fbr_abs_dir20_all_subarea, avgx_dir20_all_subarea)
time_dir20_lab1_t1, crestlen_dir20_lab1_t1, fbr_abs_dir20_lab1_t1, avgx_dir20_lab1_t1 = \
    subsample_by_time(tstart, tend, time_dir20_lab1_subarea, crestlen_dir20_lab1_subarea, fbr_abs_dir20_lab1_subarea, avgx_dir20_lab1_subarea)
time_dir20_lab2_t1, crestlen_dir20_lab2_t1, fbr_abs_dir20_lab2_t1, avgx_dir20_lab2_t1 = \
    subsample_by_time(tstart, tend, time_dir20_lab2_subarea, crestlen_dir20_lab2_subarea, fbr_abs_dir20_lab2_subarea, avgx_dir20_lab2_subarea)

crestends_dir20_all_t1 = subset_time(crestends_dir20_all, time_dir20_all_subarea, tstart, tend)
crestends_dir20_lab1_t1 = subset_time(crestends_dir20_lab1, time_dir20_lab1_subarea, tstart, tend)
crestends_dir20_lab2_t1 = subset_time(crestends_dir20_lab2, time_dir20_lab2_subarea, tstart, tend)

crestends_pertime_dir20_all_t1 = var_pertime(crestends_dir20_all_t1, time_dir20_all_t1)
crestends_pertime_dir20_lab1_t1 = var_pertime(crestends_dir20_lab1_t1, time_dir20_lab1_t1)
crestends_pertime_dir20_lab2_t1 = var_pertime(crestends_dir20_lab2_t1, time_dir20_lab2_t1)

fbr_abs_pertime_dir20_all_t1 = var_pertime(fbr_abs_dir20_all_t1, time_dir20_all_t1)
fbr_abs_pertime_dir20_lab1_t1 = var_pertime(fbr_abs_dir20_lab1_t1, time_dir20_lab1_t1)
fbr_abs_pertime_dir20_lab2_t1 = var_pertime(fbr_abs_dir20_lab2_t1, time_dir20_lab2_t1)

# calculate stats for second 10 min window
tstart = 10*60/0.2; tend = tstart + 10*60/0.2
time_dir20_all_t2, crestlen_dir20_all_t2, fbr_abs_dir20_all_t2, avgx_dir20_all_t2 = \
    subsample_by_time(tstart, tend, time_dir20_all_subarea, crestlen_dir20_all_subarea, fbr_abs_dir20_all_subarea, avgx_dir20_all_subarea)
time_dir20_lab1_t2, crestlen_dir20_lab1_t2, fbr_abs_dir20_lab1_t2, avgx_dir20_lab1_t2 = \
    subsample_by_time(tstart, tend, time_dir20_lab1_subarea, crestlen_dir20_lab1_subarea, fbr_abs_dir20_lab1_subarea, avgx_dir20_lab1_subarea)
time_dir20_lab2_t2, crestlen_dir20_lab2_t2, fbr_abs_dir20_lab2_t2, avgx_dir20_lab2_t2 = \
    subsample_by_time(tstart, tend, time_dir20_lab2_subarea, crestlen_dir20_lab2_subarea, fbr_abs_dir20_lab2_subarea, avgx_dir20_lab2_subarea)

crestends_dir20_all_t2 = subset_time(crestends_dir20_all, time_dir20_all_subarea, tstart, tend)
crestends_dir20_lab1_t2 = subset_time(crestends_dir20_lab1, time_dir20_lab1_subarea, tstart, tend)
crestends_dir20_lab2_t2 = subset_time(crestends_dir20_lab2, time_dir20_lab2_subarea, tstart, tend)

crestends_pertime_dir20_all_t2 = var_pertime(crestends_dir20_all_t2, time_dir20_all_t2)
crestends_pertime_dir20_lab1_t2 = var_pertime(crestends_dir20_lab1_t2, time_dir20_lab1_t2)
crestends_pertime_dir20_lab2_t2 = var_pertime(crestends_dir20_lab2_t2, time_dir20_lab2_t2)

fbr_abs_pertime_dir20_all_t2 = var_pertime(fbr_abs_dir20_all_t2, time_dir20_all_t2)
fbr_abs_pertime_dir20_lab1_t2 = var_pertime(fbr_abs_dir20_lab1_t2, time_dir20_lab1_t2)
fbr_abs_pertime_dir20_lab2_t2 = var_pertime(fbr_abs_dir20_lab2_t2, time_dir20_lab2_t2)

##################################################################
rundir = 'hmo25_dir30_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

time_dir30_all, crestlen_dir30_all, avgx_dir30_all, minx_dir30_all, maxx_dir30_all, miny_dir30_all, maxy_dir30_all, fbr_abs_dir30_all, areacrest_dir30_all = np.loadtxt(os.path.join(fdir, 'crest_info_all.txt'), delimiter=',').T
time_dir30_lab1, crestlen_dir30_lab1, avgx_dir30_lab1, minx_dir30_lab1, maxx_dir30_lab1, miny_dir30_lab1, maxy_dir30_lab1, fbr_abs_dir30_lab1, areacrest_dir30_lab1 = np.loadtxt(os.path.join(fdir, 'crest_info_lab1.txt'), delimiter=',').T
time_dir30_lab2, crestlen_dir30_lab2, avgx_dir30_lab2, minx_dir30_lab2, maxx_dir30_lab2, miny_dir30_lab2, maxy_dir30_lab2, fbr_abs_dir30_lab2, areacrest_dir30_lab2 = np.loadtxt(os.path.join(fdir, 'crest_info_lab2.txt'), delimiter=',').T

######## process crest stats ##########
# restrict cross-shore region
time_dir30_all_cross = restrict_cross_crests(time_dir30_all, avgx_dir30_all, shore=shore[4])
areacrest_dir30_all_cross = restrict_cross_crests(areacrest_dir30_all, avgx_dir30_all, shore=shore[4])
crestlen_dir30_all_cross = restrict_cross_crests(crestlen_dir30_all, avgx_dir30_all, shore=shore[4])
fbr_abs_dir30_all_cross = restrict_cross_crests(fbr_abs_dir30_all, avgx_dir30_all, shore=shore[4])
avgx_dir30_all_cross = restrict_cross_crests(avgx_dir30_all, avgx_dir30_all, shore=shore[4])
miny_dir30_all_cross = restrict_cross_crests(miny_dir30_all, avgx_dir30_all, shore=shore[4])
maxy_dir30_all_cross = restrict_cross_crests(maxy_dir30_all, avgx_dir30_all, shore=shore[4])

time_dir30_lab1_cross = restrict_cross_crests(time_dir30_lab1, avgx_dir30_lab1, shore=shore[4])
areacrest_dir30_lab1_cross = restrict_cross_crests(areacrest_dir30_lab1, avgx_dir30_lab1, shore=shore[4])
crestlen_dir30_lab1_cross = restrict_cross_crests(crestlen_dir30_lab1, avgx_dir30_lab1, shore=shore[4])
fbr_abs_dir30_lab1_cross = restrict_cross_crests(fbr_abs_dir30_lab1, avgx_dir30_lab1, shore=shore[4])
avgx_dir30_lab1_cross = restrict_cross_crests(avgx_dir30_lab1, avgx_dir30_lab1, shore=shore[4])
miny_dir30_lab1_cross = restrict_cross_crests(miny_dir30_lab1, avgx_dir30_lab1, shore=shore[4])
maxy_dir30_lab1_cross = restrict_cross_crests(maxy_dir30_lab1, avgx_dir30_lab1, shore=shore[4])

time_dir30_lab2_cross = restrict_cross_crests(time_dir30_lab2, avgx_dir30_lab2, shore=shore[4])
areacrest_dir30_lab2_cross = restrict_cross_crests(areacrest_dir30_lab2, avgx_dir30_lab2, shore=shore[4])
crestlen_dir30_lab2_cross = restrict_cross_crests(crestlen_dir30_lab2, avgx_dir30_lab2, shore=shore[4])
fbr_abs_dir30_lab2_cross = restrict_cross_crests(fbr_abs_dir30_lab2, avgx_dir30_lab2, shore=shore[4])
avgx_dir30_lab2_cross = restrict_cross_crests(avgx_dir30_lab2, avgx_dir30_lab2, shore=shore[4])
miny_dir30_lab2_cross = restrict_cross_crests(miny_dir30_lab2, avgx_dir30_lab2, shore=shore[4])
maxy_dir30_lab2_cross = restrict_cross_crests(maxy_dir30_lab2, avgx_dir30_lab2, shore=shore[4])

# remove crests below area threshold 
time_dir30_all_subarea, crestlen_dir30_all_subarea, fbr_abs_dir30_all_subarea, avgx_dir30_all_subarea, miny_dir30_all_subarea, maxy_dir30_all_subarea = \
    subsample_by_area(time_dir30_all_cross, areacrest_dir30_all_cross, crestlen_dir30_all_cross, fbr_abs_dir30_all_cross, avgx_dir30_all_cross, miny_dir30_all_cross, maxy_dir30_all_cross)
time_dir30_lab1_subarea, crestlen_dir30_lab1_subarea, fbr_abs_dir30_lab1_subarea, avgx_dir30_lab1_subarea, miny_dir30_lab1_subarea, maxy_dir30_lab1_subarea = \
    subsample_by_area(time_dir30_lab1_cross, areacrest_dir30_lab1_cross, crestlen_dir30_lab1_cross, fbr_abs_dir30_lab1_cross, avgx_dir30_lab1_cross, miny_dir30_lab1_cross, maxy_dir30_lab1_cross)
time_dir30_lab2_subarea, crestlen_dir30_lab2_subarea, fbr_abs_dir30_lab2_subarea, avgx_dir30_lab2_subarea, miny_dir30_lab2_subarea, maxy_dir30_lab2_subarea = \
    subsample_by_area(time_dir30_lab2_cross, areacrest_dir30_lab2_cross, crestlen_dir30_lab2_cross, fbr_abs_dir30_lab2_cross, avgx_dir30_lab2_cross, miny_dir30_lab2_cross, maxy_dir30_lab2_cross)

crestends_dir30_all = crest_ends(time_dir30_all_subarea, miny_dir30_all_subarea, maxy_dir30_all_subarea, ymin=0, ymax=55)
crestends_dir30_lab1 = crest_ends(time_dir30_lab1_subarea, miny_dir30_lab1_subarea, maxy_dir30_lab1_subarea, ymin=0.6, ymax=26.1)
crestends_dir30_lab2 = crest_ends(time_dir30_lab2_subarea, miny_dir30_lab2_subarea, maxy_dir30_lab2_subarea, ymin=0.6, ymax=26.1)

crestends_pertime_dir30_all = var_pertime(crestends_dir30_all, time_dir30_all_subarea)
crestends_pertime_dir30_lab1 = var_pertime(crestends_dir30_lab1, time_dir30_lab1_subarea)
crestends_pertime_dir30_lab2 = var_pertime(crestends_dir30_lab2, time_dir30_lab2_subarea)

fbr_abs_pertime_dir30_all = var_pertime(fbr_abs_dir30_all_subarea, time_dir30_all_subarea)
fbr_abs_pertime_dir30_lab1 = var_pertime(fbr_abs_dir30_lab1_subarea, time_dir30_lab1_subarea)
fbr_abs_pertime_dir30_lab2 = var_pertime(fbr_abs_dir30_lab2_subarea, time_dir30_lab2_subarea)

# calculate stats for first 10 min window
tstart = 0; tend = tstart + 10*60/0.2
time_dir30_all_t1, crestlen_dir30_all_t1, fbr_abs_dir30_all_t1, avgx_dir30_all_t1 = \
    subsample_by_time(tstart, tend, time_dir30_all_subarea, crestlen_dir30_all_subarea, fbr_abs_dir30_all_subarea, avgx_dir30_all_subarea)
time_dir30_lab1_t1, crestlen_dir30_lab1_t1, fbr_abs_dir30_lab1_t1, avgx_dir30_lab1_t1 = \
    subsample_by_time(tstart, tend, time_dir30_lab1_subarea, crestlen_dir30_lab1_subarea, fbr_abs_dir30_lab1_subarea, avgx_dir30_lab1_subarea)
time_dir30_lab2_t1, crestlen_dir30_lab2_t1, fbr_abs_dir30_lab2_t1, avgx_dir30_lab2_t1 = \
    subsample_by_time(tstart, tend, time_dir30_lab2_subarea, crestlen_dir30_lab2_subarea, fbr_abs_dir30_lab2_subarea, avgx_dir30_lab2_subarea)

crestends_dir30_all_t1 = subset_time(crestends_dir30_all, time_dir30_all_subarea, tstart, tend)
crestends_dir30_lab1_t1 = subset_time(crestends_dir30_lab1, time_dir30_lab1_subarea, tstart, tend)
crestends_dir30_lab2_t1 = subset_time(crestends_dir30_lab2, time_dir30_lab2_subarea, tstart, tend)

crestends_pertime_dir30_all_t1 = var_pertime(crestends_dir30_all_t1, time_dir30_all_t1)
crestends_pertime_dir30_lab1_t1 = var_pertime(crestends_dir30_lab1_t1, time_dir30_lab1_t1)
crestends_pertime_dir30_lab2_t1 = var_pertime(crestends_dir30_lab2_t1, time_dir30_lab2_t1)

fbr_abs_pertime_dir30_all_t1 = var_pertime(fbr_abs_dir30_all_t1, time_dir30_all_t1)
fbr_abs_pertime_dir30_lab1_t1 = var_pertime(fbr_abs_dir30_lab1_t1, time_dir30_lab1_t1)
fbr_abs_pertime_dir30_lab2_t1 = var_pertime(fbr_abs_dir30_lab2_t1, time_dir30_lab2_t1)

# calculate stats for second 10 min window
tstart = 10*60/0.2; tend = tstart + 10*60/0.2
time_dir30_all_t2, crestlen_dir30_all_t2, fbr_abs_dir30_all_t2, avgx_dir30_all_t2 = \
    subsample_by_time(tstart, tend, time_dir30_all_subarea, crestlen_dir30_all_subarea, fbr_abs_dir30_all_subarea, avgx_dir30_all_subarea)
time_dir30_lab1_t2, crestlen_dir30_lab1_t2, fbr_abs_dir30_lab1_t2, avgx_dir30_lab1_t2 = \
    subsample_by_time(tstart, tend, time_dir30_lab1_subarea, crestlen_dir30_lab1_subarea, fbr_abs_dir30_lab1_subarea, avgx_dir30_lab1_subarea)
time_dir30_lab2_t2, crestlen_dir30_lab2_t2, fbr_abs_dir30_lab2_t2, avgx_dir30_lab2_t2 = \
    subsample_by_time(tstart, tend, time_dir30_lab2_subarea, crestlen_dir30_lab2_subarea, fbr_abs_dir30_lab2_subarea, avgx_dir30_lab2_subarea)

crestends_dir30_all_t2 = subset_time(crestends_dir30_all, time_dir30_all_subarea, tstart, tend)
crestends_dir30_lab1_t2 = subset_time(crestends_dir30_lab1, time_dir30_lab1_subarea, tstart, tend)
crestends_dir30_lab2_t2 = subset_time(crestends_dir30_lab2, time_dir30_lab2_subarea, tstart, tend)

crestends_pertime_dir30_all_t2 = var_pertime(crestends_dir30_all_t2, time_dir30_all_t2)
crestends_pertime_dir30_lab1_t2 = var_pertime(crestends_dir30_lab1_t2, time_dir30_lab1_t2)
crestends_pertime_dir30_lab2_t2 = var_pertime(crestends_dir30_lab2_t2, time_dir30_lab2_t2)

fbr_abs_pertime_dir30_all_t2 = var_pertime(fbr_abs_dir30_all_t2, time_dir30_all_t2)
fbr_abs_pertime_dir30_lab1_t2 = var_pertime(fbr_abs_dir30_lab1_t2, time_dir30_lab1_t2)
fbr_abs_pertime_dir30_lab2_t2 = var_pertime(fbr_abs_dir30_lab2_t2, time_dir30_lab2_t2)

##################################################################
rundir = 'hmo25_dir40_tp2'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

time_dir40_all, crestlen_dir40_all, avgx_dir40_all, minx_dir40_all, maxx_dir40_all, miny_dir40_all, maxy_dir40_all, fbr_abs_dir40_all, areacrest_dir40_all = np.loadtxt(os.path.join(fdir, 'crest_info_all.txt'), delimiter=',').T
time_dir40_lab1, crestlen_dir40_lab1, avgx_dir40_lab1, minx_dir40_lab1, maxx_dir40_lab1, miny_dir40_lab1, maxy_dir40_lab1, fbr_abs_dir40_lab1, areacrest_dir40_lab1 = np.loadtxt(os.path.join(fdir, 'crest_info_lab1.txt'), delimiter=',').T
time_dir40_lab2, crestlen_dir40_lab2, avgx_dir40_lab2, minx_dir40_lab2, maxx_dir40_lab2, miny_dir40_lab2, maxy_dir40_lab2, fbr_abs_dir40_lab2, areacrest_dir40_lab2 = np.loadtxt(os.path.join(fdir, 'crest_info_lab2.txt'), delimiter=',').T

######## process crest stats ##########
# restrict cross-shore region
time_dir40_all_cross = restrict_cross_crests(time_dir40_all, avgx_dir40_all, shore=shore[5])
areacrest_dir40_all_cross = restrict_cross_crests(areacrest_dir40_all, avgx_dir40_all, shore=shore[5])
crestlen_dir40_all_cross = restrict_cross_crests(crestlen_dir40_all, avgx_dir40_all, shore=shore[5])
fbr_abs_dir40_all_cross = restrict_cross_crests(fbr_abs_dir40_all, avgx_dir40_all, shore=shore[5])
avgx_dir40_all_cross = restrict_cross_crests(avgx_dir40_all, avgx_dir40_all, shore=shore[5])
miny_dir40_all_cross = restrict_cross_crests(miny_dir40_all, avgx_dir40_all, shore=shore[5])
maxy_dir40_all_cross = restrict_cross_crests(maxy_dir40_all, avgx_dir40_all, shore=shore[5])

time_dir40_lab1_cross = restrict_cross_crests(time_dir40_lab1, avgx_dir40_lab1, shore=shore[5])
areacrest_dir40_lab1_cross = restrict_cross_crests(areacrest_dir40_lab1, avgx_dir40_lab1, shore=shore[5])
crestlen_dir40_lab1_cross = restrict_cross_crests(crestlen_dir40_lab1, avgx_dir40_lab1, shore=shore[5])
fbr_abs_dir40_lab1_cross = restrict_cross_crests(fbr_abs_dir40_lab1, avgx_dir40_lab1, shore=shore[5])
avgx_dir40_lab1_cross = restrict_cross_crests(avgx_dir40_lab1, avgx_dir40_lab1, shore=shore[5])
miny_dir40_lab1_cross = restrict_cross_crests(miny_dir40_lab1, avgx_dir40_lab1, shore=shore[5])
maxy_dir40_lab1_cross = restrict_cross_crests(maxy_dir40_lab1, avgx_dir40_lab1, shore=shore[5])

time_dir40_lab2_cross = restrict_cross_crests(time_dir40_lab2, avgx_dir40_lab2, shore=shore[5])
areacrest_dir40_lab2_cross = restrict_cross_crests(areacrest_dir40_lab2, avgx_dir40_lab2, shore=shore[5])
crestlen_dir40_lab2_cross = restrict_cross_crests(crestlen_dir40_lab2, avgx_dir40_lab2, shore=shore[5])
fbr_abs_dir40_lab2_cross = restrict_cross_crests(fbr_abs_dir40_lab2, avgx_dir40_lab2, shore=shore[5])
avgx_dir40_lab2_cross = restrict_cross_crests(avgx_dir40_lab2, avgx_dir40_lab2, shore=shore[5])
miny_dir40_lab2_cross = restrict_cross_crests(miny_dir40_lab2, avgx_dir40_lab2, shore=shore[5])
maxy_dir40_lab2_cross = restrict_cross_crests(maxy_dir40_lab2, avgx_dir40_lab2, shore=shore[5])

# remove crests below area threshold 
time_dir40_all_subarea, crestlen_dir40_all_subarea, fbr_abs_dir40_all_subarea, avgx_dir40_all_subarea, miny_dir40_all_subarea, maxy_dir40_all_subarea = \
    subsample_by_area(time_dir40_all_cross, areacrest_dir40_all_cross, crestlen_dir40_all_cross, fbr_abs_dir40_all_cross, avgx_dir40_all_cross, miny_dir40_all_cross, maxy_dir40_all_cross)
time_dir40_lab1_subarea, crestlen_dir40_lab1_subarea, fbr_abs_dir40_lab1_subarea, avgx_dir40_lab1_subarea, miny_dir40_lab1_subarea, maxy_dir40_lab1_subarea = \
    subsample_by_area(time_dir40_lab1_cross, areacrest_dir40_lab1_cross, crestlen_dir40_lab1_cross, fbr_abs_dir40_lab1_cross, avgx_dir40_lab1_cross, miny_dir40_lab1_cross, maxy_dir40_lab1_cross)
time_dir40_lab2_subarea, crestlen_dir40_lab2_subarea, fbr_abs_dir40_lab2_subarea, avgx_dir40_lab2_subarea, miny_dir40_lab2_subarea, maxy_dir40_lab2_subarea = \
    subsample_by_area(time_dir40_lab2_cross, areacrest_dir40_lab2_cross, crestlen_dir40_lab2_cross, fbr_abs_dir40_lab2_cross, avgx_dir40_lab2_cross, miny_dir40_lab2_cross, maxy_dir40_lab2_cross)

crestends_dir40_all = crest_ends(time_dir40_all_subarea, miny_dir40_all_subarea, maxy_dir40_all_subarea, ymin=0, ymax=55)
crestends_dir40_lab1 = crest_ends(time_dir40_lab1_subarea, miny_dir40_lab1_subarea, maxy_dir40_lab1_subarea, ymin=0.6, ymax=26.1)
crestends_dir40_lab2 = crest_ends(time_dir40_lab2_subarea, miny_dir40_lab2_subarea, maxy_dir40_lab2_subarea, ymin=0.6, ymax=26.1)

crestends_pertime_dir40_all = var_pertime(crestends_dir40_all, time_dir40_all_subarea)
crestends_pertime_dir40_lab1 = var_pertime(crestends_dir40_lab1, time_dir40_lab1_subarea)
crestends_pertime_dir40_lab2 = var_pertime(crestends_dir40_lab2, time_dir40_lab2_subarea)

fbr_abs_pertime_dir40_all = var_pertime(fbr_abs_dir40_all_subarea, time_dir40_all_subarea)
fbr_abs_pertime_dir40_lab1 = var_pertime(fbr_abs_dir40_lab1_subarea, time_dir40_lab1_subarea)
fbr_abs_pertime_dir40_lab2 = var_pertime(fbr_abs_dir40_lab2_subarea, time_dir40_lab2_subarea)

# calculate stats for first 10 min window
tstart = 0; tend = tstart + 10*60/0.2
time_dir40_all_t1, crestlen_dir40_all_t1, fbr_abs_dir40_all_t1, avgx_dir40_all_t1 = \
    subsample_by_time(tstart, tend, time_dir40_all_subarea, crestlen_dir40_all_subarea, fbr_abs_dir40_all_subarea, avgx_dir40_all_subarea)
time_dir40_lab1_t1, crestlen_dir40_lab1_t1, fbr_abs_dir40_lab1_t1, avgx_dir40_lab1_t1 = \
    subsample_by_time(tstart, tend, time_dir40_lab1_subarea, crestlen_dir40_lab1_subarea, fbr_abs_dir40_lab1_subarea, avgx_dir40_lab1_subarea)
time_dir40_lab2_t1, crestlen_dir40_lab2_t1, fbr_abs_dir40_lab2_t1, avgx_dir40_lab2_t1 = \
    subsample_by_time(tstart, tend, time_dir40_lab2_subarea, crestlen_dir40_lab2_subarea, fbr_abs_dir40_lab2_subarea, avgx_dir40_lab2_subarea)

crestends_dir40_all_t1 = subset_time(crestends_dir40_all, time_dir40_all_subarea, tstart, tend)
crestends_dir40_lab1_t1 = subset_time(crestends_dir40_lab1, time_dir40_lab1_subarea, tstart, tend)
crestends_dir40_lab2_t1 = subset_time(crestends_dir40_lab2, time_dir40_lab2_subarea, tstart, tend)

crestends_pertime_dir40_all_t1 = var_pertime(crestends_dir40_all_t1, time_dir40_all_t1)
crestends_pertime_dir40_lab1_t1 = var_pertime(crestends_dir40_lab1_t1, time_dir40_lab1_t1)
crestends_pertime_dir40_lab2_t1 = var_pertime(crestends_dir40_lab2_t1, time_dir40_lab2_t1)

fbr_abs_pertime_dir40_all_t1 = var_pertime(fbr_abs_dir40_all_t1, time_dir40_all_t1)
fbr_abs_pertime_dir40_lab1_t1 = var_pertime(fbr_abs_dir40_lab1_t1, time_dir40_lab1_t1)
fbr_abs_pertime_dir40_lab2_t1 = var_pertime(fbr_abs_dir40_lab2_t1, time_dir40_lab2_t1)

# calculate stats for second 10 min window
tstart = 10*60/0.2; tend = tstart + 10*60/0.2
time_dir40_all_t2, crestlen_dir40_all_t2, fbr_abs_dir40_all_t2, avgx_dir40_all_t2 = \
    subsample_by_time(tstart, tend, time_dir40_all_subarea, crestlen_dir40_all_subarea, fbr_abs_dir40_all_subarea, avgx_dir40_all_subarea)
time_dir40_lab1_t2, crestlen_dir40_lab1_t2, fbr_abs_dir40_lab1_t2, avgx_dir40_lab1_t2 = \
    subsample_by_time(tstart, tend, time_dir40_lab1_subarea, crestlen_dir40_lab1_subarea, fbr_abs_dir40_lab1_subarea, avgx_dir40_lab1_subarea)
time_dir40_lab2_t2, crestlen_dir40_lab2_t2, fbr_abs_dir40_lab2_t2, avgx_dir40_lab2_t2 = \
    subsample_by_time(tstart, tend, time_dir40_lab2_subarea, crestlen_dir40_lab2_subarea, fbr_abs_dir40_lab2_subarea, avgx_dir40_lab2_subarea)

crestends_dir40_all_t2 = subset_time(crestends_dir40_all, time_dir40_all_subarea, tstart, tend)
crestends_dir40_lab1_t2 = subset_time(crestends_dir40_lab1, time_dir40_lab1_subarea, tstart, tend)
crestends_dir40_lab2_t2 = subset_time(crestends_dir40_lab2, time_dir40_lab2_subarea, tstart, tend)

crestends_pertime_dir40_all_t2 = var_pertime(crestends_dir40_all_t2, time_dir40_all_t2)
crestends_pertime_dir40_lab1_t2 = var_pertime(crestends_dir40_lab1_t2, time_dir40_lab1_t2)
crestends_pertime_dir40_lab2_t2 = var_pertime(crestends_dir40_lab2_t2, time_dir40_lab2_t2)

fbr_abs_pertime_dir40_all_t2 = var_pertime(fbr_abs_dir40_all_t2, time_dir40_all_t2)
fbr_abs_pertime_dir40_lab1_t2 = var_pertime(fbr_abs_dir40_lab1_t2, time_dir40_lab1_t2)
fbr_abs_pertime_dir40_lab2_t2 = var_pertime(fbr_abs_dir40_lab2_t2, time_dir40_lab2_t2)

##################################################################
rundir = 'hmo25_dir20_tp15'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

time_dir20_tp15_all, crestlen_dir20_tp15_all, avgx_dir20_tp15_all, minx_dir20_tp15_all, maxx_dir20_tp15_all, miny_dir20_tp15_all, maxy_dir20_tp15_all, fbr_abs_dir20_tp15_all, areacrest_dir20_tp15_all = np.loadtxt(os.path.join(fdir, 'crest_info_all.txt'), delimiter=',').T
time_dir20_tp15_lab1, crestlen_dir20_tp15_lab1, avgx_dir20_tp15_lab1, minx_dir20_tp15_lab1, maxx_dir20_tp15_lab1, miny_dir20_tp15_lab1, maxy_dir20_tp15_lab1, fbr_abs_dir20_tp15_lab1, areacrest_dir20_tp15_lab1 = np.loadtxt(os.path.join(fdir, 'crest_info_lab1.txt'), delimiter=',').T
time_dir20_tp15_lab2, crestlen_dir20_tp15_lab2, avgx_dir20_tp15_lab2, minx_dir20_tp15_lab2, maxx_dir20_tp15_lab2, miny_dir20_tp15_lab2, maxy_dir20_tp15_lab2, fbr_abs_dir20_tp15_lab2, areacrest_dir20_tp15_lab2 = np.loadtxt(os.path.join(fdir, 'crest_info_lab2.txt'), delimiter=',').T

######## process crest stats ##########
# restrict cross-shore region
time_dir20_tp15_all_cross = restrict_cross_crests(time_dir20_tp15_all, avgx_dir20_tp15_all, shore=shore[6])
areacrest_dir20_tp15_all_cross = restrict_cross_crests(areacrest_dir20_tp15_all, avgx_dir20_tp15_all, shore=shore[6])
crestlen_dir20_tp15_all_cross = restrict_cross_crests(crestlen_dir20_tp15_all, avgx_dir20_tp15_all, shore=shore[6])
fbr_abs_dir20_tp15_all_cross = restrict_cross_crests(fbr_abs_dir20_tp15_all, avgx_dir20_tp15_all, shore=shore[6])
avgx_dir20_tp15_all_cross = restrict_cross_crests(avgx_dir20_tp15_all, avgx_dir20_tp15_all, shore=shore[6])
miny_dir20_tp15_all_cross = restrict_cross_crests(miny_dir20_tp15_all, avgx_dir20_tp15_all, shore=shore[6])
maxy_dir20_tp15_all_cross = restrict_cross_crests(maxy_dir20_tp15_all, avgx_dir20_tp15_all, shore=shore[6])

time_dir20_tp15_lab1_cross = restrict_cross_crests(time_dir20_tp15_lab1, avgx_dir20_tp15_lab1, shore=shore[6])
areacrest_dir20_tp15_lab1_cross = restrict_cross_crests(areacrest_dir20_tp15_lab1, avgx_dir20_tp15_lab1, shore=shore[6])
crestlen_dir20_tp15_lab1_cross = restrict_cross_crests(crestlen_dir20_tp15_lab1, avgx_dir20_tp15_lab1, shore=shore[6])
fbr_abs_dir20_tp15_lab1_cross = restrict_cross_crests(fbr_abs_dir20_tp15_lab1, avgx_dir20_tp15_lab1, shore=shore[6])
avgx_dir20_tp15_lab1_cross = restrict_cross_crests(avgx_dir20_tp15_lab1, avgx_dir20_tp15_lab1, shore=shore[6])
miny_dir20_tp15_lab1_cross = restrict_cross_crests(miny_dir20_tp15_lab1, avgx_dir20_tp15_lab1, shore=shore[6])
maxy_dir20_tp15_lab1_cross = restrict_cross_crests(maxy_dir20_tp15_lab1, avgx_dir20_tp15_lab1, shore=shore[6])

time_dir20_tp15_lab2_cross = restrict_cross_crests(time_dir20_tp15_lab2, avgx_dir20_tp15_lab2, shore=shore[6])
areacrest_dir20_tp15_lab2_cross = restrict_cross_crests(areacrest_dir20_tp15_lab2, avgx_dir20_tp15_lab2, shore=shore[6])
crestlen_dir20_tp15_lab2_cross = restrict_cross_crests(crestlen_dir20_tp15_lab2, avgx_dir20_tp15_lab2, shore=shore[6])
fbr_abs_dir20_tp15_lab2_cross = restrict_cross_crests(fbr_abs_dir20_tp15_lab2, avgx_dir20_tp15_lab2, shore=shore[6])
avgx_dir20_tp15_lab2_cross = restrict_cross_crests(avgx_dir20_tp15_lab2, avgx_dir20_tp15_lab2, shore=shore[6])
miny_dir20_tp15_lab2_cross = restrict_cross_crests(miny_dir20_tp15_lab2, avgx_dir20_tp15_lab2, shore=shore[6])
maxy_dir20_tp15_lab2_cross = restrict_cross_crests(maxy_dir20_tp15_lab2, avgx_dir20_tp15_lab2, shore=shore[6])

# remove crests below area threshold 
time_dir20_tp15_all_subarea, crestlen_dir20_tp15_all_subarea, fbr_abs_dir20_tp15_all_subarea, avgx_dir20_tp15_all_subarea, miny_dir20_tp15_all_subarea, maxy_dir20_tp15_all_subarea = \
    subsample_by_area(time_dir20_tp15_all_cross, areacrest_dir20_tp15_all_cross, crestlen_dir20_tp15_all_cross, fbr_abs_dir20_tp15_all_cross, avgx_dir20_tp15_all_cross, miny_dir20_tp15_all_cross, maxy_dir20_tp15_all_cross)
time_dir20_tp15_lab1_subarea, crestlen_dir20_tp15_lab1_subarea, fbr_abs_dir20_tp15_lab1_subarea, avgx_dir20_tp15_lab1_subarea, miny_dir20_tp15_lab1_subarea, maxy_dir20_tp15_lab1_subarea = \
    subsample_by_area(time_dir20_tp15_lab1_cross, areacrest_dir20_tp15_lab1_cross, crestlen_dir20_tp15_lab1_cross, fbr_abs_dir20_tp15_lab1_cross, avgx_dir20_tp15_lab1_cross, miny_dir20_tp15_lab1_cross, maxy_dir20_tp15_lab1_cross)
time_dir20_tp15_lab2_subarea, crestlen_dir20_tp15_lab2_subarea, fbr_abs_dir20_tp15_lab2_subarea, avgx_dir20_tp15_lab2_subarea, miny_dir20_tp15_lab2_subarea, maxy_dir20_tp15_lab2_subarea = \
    subsample_by_area(time_dir20_tp15_lab2_cross, areacrest_dir20_tp15_lab2_cross, crestlen_dir20_tp15_lab2_cross, fbr_abs_dir20_tp15_lab2_cross, avgx_dir20_tp15_lab2_cross, miny_dir20_tp15_lab2_cross, maxy_dir20_tp15_lab2_cross)

crestends_dir20_tp15_all = crest_ends(time_dir20_tp15_all_subarea, miny_dir20_tp15_all_subarea, maxy_dir20_tp15_all_subarea, ymin=0, ymax=55)
crestends_dir20_tp15_lab1 = crest_ends(time_dir20_tp15_lab1_subarea, miny_dir20_tp15_lab1_subarea, maxy_dir20_tp15_lab1_subarea, ymin=0.6, ymax=26.1)
crestends_dir20_tp15_lab2 = crest_ends(time_dir20_tp15_lab2_subarea, miny_dir20_tp15_lab2_subarea, maxy_dir20_tp15_lab2_subarea, ymin=0.6, ymax=26.1)

crestends_pertime_dir20_tp15_all = var_pertime(crestends_dir20_tp15_all, time_dir20_tp15_all_subarea)
crestends_pertime_dir20_tp15_lab1 = var_pertime(crestends_dir20_tp15_lab1, time_dir20_tp15_lab1_subarea)
crestends_pertime_dir20_tp15_lab2 = var_pertime(crestends_dir20_tp15_lab2, time_dir20_tp15_lab2_subarea)

fbr_abs_pertime_dir20_tp15_all = var_pertime(fbr_abs_dir20_tp15_all_subarea, time_dir20_tp15_all_subarea)
fbr_abs_pertime_dir20_tp15_lab1 = var_pertime(fbr_abs_dir20_tp15_lab1_subarea, time_dir20_tp15_lab1_subarea)
fbr_abs_pertime_dir20_tp15_lab2 = var_pertime(fbr_abs_dir20_tp15_lab2_subarea, time_dir20_tp15_lab2_subarea)

# calculate stats for first 10 min window
tstart = 0; tend = tstart + 10*60/0.2
time_dir20_tp15_all_t1, crestlen_dir20_tp15_all_t1, fbr_abs_dir20_tp15_all_t1, avgx_dir20_tp15_all_t1 = \
    subsample_by_time(tstart, tend, time_dir20_tp15_all_subarea, crestlen_dir20_tp15_all_subarea, fbr_abs_dir20_tp15_all_subarea, avgx_dir20_tp15_all_subarea)
time_dir20_tp15_lab1_t1, crestlen_dir20_tp15_lab1_t1, fbr_abs_dir20_tp15_lab1_t1, avgx_dir20_tp15_lab1_t1 = \
    subsample_by_time(tstart, tend, time_dir20_tp15_lab1_subarea, crestlen_dir20_tp15_lab1_subarea, fbr_abs_dir20_tp15_lab1_subarea, avgx_dir20_tp15_lab1_subarea)
time_dir20_tp15_lab2_t1, crestlen_dir20_tp15_lab2_t1, fbr_abs_dir20_tp15_lab2_t1, avgx_dir20_tp15_lab2_t1 = \
    subsample_by_time(tstart, tend, time_dir20_tp15_lab2_subarea, crestlen_dir20_tp15_lab2_subarea, fbr_abs_dir20_tp15_lab2_subarea, avgx_dir20_tp15_lab2_subarea)

crestends_dir20_tp15_all_t1 = subset_time(crestends_dir20_tp15_all, time_dir20_tp15_all_subarea, tstart, tend)
crestends_dir20_tp15_lab1_t1 = subset_time(crestends_dir20_tp15_lab1, time_dir20_tp15_lab1_subarea, tstart, tend)
crestends_dir20_tp15_lab2_t1 = subset_time(crestends_dir20_tp15_lab2, time_dir20_tp15_lab2_subarea, tstart, tend)

crestends_pertime_dir20_tp15_all_t1 = var_pertime(crestends_dir20_tp15_all_t1, time_dir20_tp15_all_t1)
crestends_pertime_dir20_tp15_lab1_t1 = var_pertime(crestends_dir20_tp15_lab1_t1, time_dir20_tp15_lab1_t1)
crestends_pertime_dir20_tp15_lab2_t1 = var_pertime(crestends_dir20_tp15_lab2_t1, time_dir20_tp15_lab2_t1)

fbr_abs_pertime_dir20_tp15_all_t1 = var_pertime(fbr_abs_dir20_tp15_all_t1, time_dir20_tp15_all_t1)
fbr_abs_pertime_dir20_tp15_lab1_t1 = var_pertime(fbr_abs_dir20_tp15_lab1_t1, time_dir20_tp15_lab1_t1)
fbr_abs_pertime_dir20_tp15_lab2_t1 = var_pertime(fbr_abs_dir20_tp15_lab2_t1, time_dir20_tp15_lab2_t1)

# calculate stats for second 10 min window
tstart = 10*60/0.2; tend = tstart + 10*60/0.2
time_dir20_tp15_all_t2, crestlen_dir20_tp15_all_t2, fbr_abs_dir20_tp15_all_t2, avgx_dir20_tp15_all_t2 = \
    subsample_by_time(tstart, tend, time_dir20_tp15_all_subarea, crestlen_dir20_tp15_all_subarea, fbr_abs_dir20_tp15_all_subarea, avgx_dir20_tp15_all_subarea)
time_dir20_tp15_lab1_t2, crestlen_dir20_tp15_lab1_t2, fbr_abs_dir20_tp15_lab1_t2, avgx_dir20_tp15_lab1_t2 = \
    subsample_by_time(tstart, tend, time_dir20_tp15_lab1_subarea, crestlen_dir20_tp15_lab1_subarea, fbr_abs_dir20_tp15_lab1_subarea, avgx_dir20_tp15_lab1_subarea)
time_dir20_tp15_lab2_t2, crestlen_dir20_tp15_lab2_t2, fbr_abs_dir20_tp15_lab2_t2, avgx_dir20_tp15_lab2_t2 = \
    subsample_by_time(tstart, tend, time_dir20_tp15_lab2_subarea, crestlen_dir20_tp15_lab2_subarea, fbr_abs_dir20_tp15_lab2_subarea, avgx_dir20_tp15_lab2_subarea)

crestends_dir20_tp15_all_t2 = subset_time(crestends_dir20_tp15_all, time_dir20_tp15_all_subarea, tstart, tend)
crestends_dir20_tp15_lab1_t2 = subset_time(crestends_dir20_tp15_lab1, time_dir20_tp15_lab1_subarea, tstart, tend)
crestends_dir20_tp15_lab2_t2 = subset_time(crestends_dir20_tp15_lab2, time_dir20_tp15_lab2_subarea, tstart, tend)

crestends_pertime_dir20_tp15_all_t2 = var_pertime(crestends_dir20_tp15_all_t2, time_dir20_tp15_all_t2)
crestends_pertime_dir20_tp15_lab1_t2 = var_pertime(crestends_dir20_tp15_lab1_t2, time_dir20_tp15_lab1_t2)
crestends_pertime_dir20_tp15_lab2_t2 = var_pertime(crestends_dir20_tp15_lab2_t2, time_dir20_tp15_lab2_t2)

fbr_abs_pertime_dir20_tp15_all_t2 = var_pertime(fbr_abs_dir20_tp15_all_t2, time_dir20_tp15_all_t2)
fbr_abs_pertime_dir20_tp15_lab1_t2 = var_pertime(fbr_abs_dir20_tp15_lab1_t2, time_dir20_tp15_lab1_t2)
fbr_abs_pertime_dir20_tp15_lab2_t2 = var_pertime(fbr_abs_dir20_tp15_lab2_t2, time_dir20_tp15_lab2_t2)

##################################################################
rundir = 'hmo25_dir20_tp25'
rootdir = os.path.join('/gscratch', 'nearshore', 'enuss','lab_runs_y550','postprocessing')
savedir = os.path.join(rootdir, 'compiled_output_'+rundir, 'plots')
fdir = os.path.join(rootdir, 'compiled_output_'+rundir, 'lab_netcdfs')

time_dir20_tp25_all, crestlen_dir20_tp25_all, avgx_dir20_tp25_all, minx_dir20_tp25_all, maxx_dir20_tp25_all, miny_dir20_tp25_all, maxy_dir20_tp25_all, fbr_abs_dir20_tp25_all, areacrest_dir20_tp25_all = np.loadtxt(os.path.join(fdir, 'crest_info_all.txt'), delimiter=',').T
time_dir20_tp25_lab1, crestlen_dir20_tp25_lab1, avgx_dir20_tp25_lab1, minx_dir20_tp25_lab1, maxx_dir20_tp25_lab1, miny_dir20_tp25_lab1, maxy_dir20_tp25_lab1, fbr_abs_dir20_tp25_lab1, areacrest_dir20_tp25_lab1 = np.loadtxt(os.path.join(fdir, 'crest_info_lab1.txt'), delimiter=',').T
time_dir20_tp25_lab2, crestlen_dir20_tp25_lab2, avgx_dir20_tp25_lab2, minx_dir20_tp25_lab2, maxx_dir20_tp25_lab2, miny_dir20_tp25_lab2, maxy_dir20_tp25_lab2, fbr_abs_dir20_tp25_lab2, areacrest_dir20_tp25_lab2 = np.loadtxt(os.path.join(fdir, 'crest_info_lab2.txt'), delimiter=',').T

######## process crest stats ##########
# restrict cross-shore region
time_dir20_tp25_all_cross = restrict_cross_crests(time_dir20_tp25_all, avgx_dir20_tp25_all, shore=shore[7])
areacrest_dir20_tp25_all_cross = restrict_cross_crests(areacrest_dir20_tp25_all, avgx_dir20_tp25_all, shore=shore[7])
crestlen_dir20_tp25_all_cross = restrict_cross_crests(crestlen_dir20_tp25_all, avgx_dir20_tp25_all, shore=shore[7])
fbr_abs_dir20_tp25_all_cross = restrict_cross_crests(fbr_abs_dir20_tp25_all, avgx_dir20_tp25_all, shore=shore[7])
avgx_dir20_tp25_all_cross = restrict_cross_crests(avgx_dir20_tp25_all, avgx_dir20_tp25_all, shore=shore[7])
miny_dir20_tp25_all_cross = restrict_cross_crests(miny_dir20_tp25_all, avgx_dir20_tp25_all, shore=shore[7])
maxy_dir20_tp25_all_cross = restrict_cross_crests(maxy_dir20_tp25_all, avgx_dir20_tp25_all, shore=shore[7])

time_dir20_tp25_lab1_cross = restrict_cross_crests(time_dir20_tp25_lab1, avgx_dir20_tp25_lab1, shore=shore[7])
areacrest_dir20_tp25_lab1_cross = restrict_cross_crests(areacrest_dir20_tp25_lab1, avgx_dir20_tp25_lab1, shore=shore[7])
crestlen_dir20_tp25_lab1_cross = restrict_cross_crests(crestlen_dir20_tp25_lab1, avgx_dir20_tp25_lab1, shore=shore[7])
fbr_abs_dir20_tp25_lab1_cross = restrict_cross_crests(fbr_abs_dir20_tp25_lab1, avgx_dir20_tp25_lab1, shore=shore[7])
avgx_dir20_tp25_lab1_cross = restrict_cross_crests(avgx_dir20_tp25_lab1, avgx_dir20_tp25_lab1, shore=shore[7])
miny_dir20_tp25_lab1_cross = restrict_cross_crests(miny_dir20_tp25_lab1, avgx_dir20_tp25_lab1, shore=shore[7])
maxy_dir20_tp25_lab1_cross = restrict_cross_crests(maxy_dir20_tp25_lab1, avgx_dir20_tp25_lab1, shore=shore[7])

time_dir20_tp25_lab2_cross = restrict_cross_crests(time_dir20_tp25_lab2, avgx_dir20_tp25_lab2, shore=shore[7])
areacrest_dir20_tp25_lab2_cross = restrict_cross_crests(areacrest_dir20_tp25_lab2, avgx_dir20_tp25_lab2, shore=shore[7])
crestlen_dir20_tp25_lab2_cross = restrict_cross_crests(crestlen_dir20_tp25_lab2, avgx_dir20_tp25_lab2, shore=shore[7])
fbr_abs_dir20_tp25_lab2_cross = restrict_cross_crests(fbr_abs_dir20_tp25_lab2, avgx_dir20_tp25_lab2, shore=shore[7])
avgx_dir20_tp25_lab2_cross = restrict_cross_crests(avgx_dir20_tp25_lab2, avgx_dir20_tp25_lab2, shore=shore[7])
miny_dir20_tp25_lab2_cross = restrict_cross_crests(miny_dir20_tp25_lab2, avgx_dir20_tp25_lab2, shore=shore[7])
maxy_dir20_tp25_lab2_cross = restrict_cross_crests(maxy_dir20_tp25_lab2, avgx_dir20_tp25_lab2, shore=shore[7])

# remove crests below area threshold 
time_dir20_tp25_all_subarea, crestlen_dir20_tp25_all_subarea, fbr_abs_dir20_tp25_all_subarea, avgx_dir20_tp25_all_subarea, miny_dir20_tp25_all_subarea, maxy_dir20_tp25_all_subarea = \
    subsample_by_area(time_dir20_tp25_all_cross, areacrest_dir20_tp25_all_cross, crestlen_dir20_tp25_all_cross, fbr_abs_dir20_tp25_all_cross, avgx_dir20_tp25_all_cross, miny_dir20_tp25_all_cross, maxy_dir20_tp25_all_cross)
time_dir20_tp25_lab1_subarea, crestlen_dir20_tp25_lab1_subarea, fbr_abs_dir20_tp25_lab1_subarea, avgx_dir20_tp25_lab1_subarea, miny_dir20_tp25_lab1_subarea, maxy_dir20_tp25_lab1_subarea = \
    subsample_by_area(time_dir20_tp25_lab1_cross, areacrest_dir20_tp25_lab1_cross, crestlen_dir20_tp25_lab1_cross, fbr_abs_dir20_tp25_lab1_cross, avgx_dir20_tp25_lab1_cross, miny_dir20_tp25_lab1_cross, maxy_dir20_tp25_lab1_cross)
time_dir20_tp25_lab2_subarea, crestlen_dir20_tp25_lab2_subarea, fbr_abs_dir20_tp25_lab2_subarea, avgx_dir20_tp25_lab2_subarea, miny_dir20_tp25_lab2_subarea, maxy_dir20_tp25_lab2_subarea = \
    subsample_by_area(time_dir20_tp25_lab2_cross, areacrest_dir20_tp25_lab2_cross, crestlen_dir20_tp25_lab2_cross, fbr_abs_dir20_tp25_lab2_cross, avgx_dir20_tp25_lab2_cross, miny_dir20_tp25_lab2_cross, maxy_dir20_tp25_lab2_cross)

crestends_dir20_tp25_all = crest_ends(time_dir20_tp25_all_subarea, miny_dir20_tp25_all_subarea, maxy_dir20_tp25_all_subarea, ymin=0, ymax=55)
crestends_dir20_tp25_lab1 = crest_ends(time_dir20_tp25_lab1_subarea, miny_dir20_tp25_lab1_subarea, maxy_dir20_tp25_lab1_subarea, ymin=0.6, ymax=26.1)
crestends_dir20_tp25_lab2 = crest_ends(time_dir20_tp25_lab2_subarea, miny_dir20_tp25_lab2_subarea, maxy_dir20_tp25_lab2_subarea, ymin=0.6, ymax=26.1)

crestends_pertime_dir20_tp25_all = var_pertime(crestends_dir20_tp25_all, time_dir20_tp25_all_subarea)
crestends_pertime_dir20_tp25_lab1 = var_pertime(crestends_dir20_tp25_lab1, time_dir20_tp25_lab1_subarea)
crestends_pertime_dir20_tp25_lab2 = var_pertime(crestends_dir20_tp25_lab2, time_dir20_tp25_lab2_subarea)

fbr_abs_pertime_dir20_tp25_all = var_pertime(fbr_abs_dir20_tp25_all_subarea, time_dir20_tp25_all_subarea)
fbr_abs_pertime_dir20_tp25_lab1 = var_pertime(fbr_abs_dir20_tp25_lab1_subarea, time_dir20_tp25_lab1_subarea)
fbr_abs_pertime_dir20_tp25_lab2 = var_pertime(fbr_abs_dir20_tp25_lab2_subarea, time_dir20_tp25_lab2_subarea)

# calculate stats for first 10 min window
tstart = 0; tend = tstart + 10*60/0.2
time_dir20_tp25_all_t1, crestlen_dir20_tp25_all_t1, fbr_abs_dir20_tp25_all_t1, avgx_dir20_tp25_all_t1 = \
    subsample_by_time(tstart, tend, time_dir20_tp25_all_subarea, crestlen_dir20_tp25_all_subarea, fbr_abs_dir20_tp25_all_subarea, avgx_dir20_tp25_all_subarea)
time_dir20_tp25_lab1_t1, crestlen_dir20_tp25_lab1_t1, fbr_abs_dir20_tp25_lab1_t1, avgx_dir20_tp25_lab1_t1 = \
    subsample_by_time(tstart, tend, time_dir20_tp25_lab1_subarea, crestlen_dir20_tp25_lab1_subarea, fbr_abs_dir20_tp25_lab1_subarea, avgx_dir20_tp25_lab1_subarea)
time_dir20_tp25_lab2_t1, crestlen_dir20_tp25_lab2_t1, fbr_abs_dir20_tp25_lab2_t1, avgx_dir20_tp25_lab2_t1 = \
    subsample_by_time(tstart, tend, time_dir20_tp25_lab2_subarea, crestlen_dir20_tp25_lab2_subarea, fbr_abs_dir20_tp25_lab2_subarea, avgx_dir20_tp25_lab2_subarea)

crestends_dir20_tp25_all_t1 = subset_time(crestends_dir20_tp25_all, time_dir20_tp25_all_subarea, tstart, tend)
crestends_dir20_tp25_lab1_t1 = subset_time(crestends_dir20_tp25_lab1, time_dir20_tp25_lab1_subarea, tstart, tend)
crestends_dir20_tp25_lab2_t1 = subset_time(crestends_dir20_tp25_lab2, time_dir20_tp25_lab2_subarea, tstart, tend)

crestends_pertime_dir20_tp25_all_t1 = var_pertime(crestends_dir20_tp25_all_t1, time_dir20_tp25_all_t1)
crestends_pertime_dir20_tp25_lab1_t1 = var_pertime(crestends_dir20_tp25_lab1_t1, time_dir20_tp25_lab1_t1)
crestends_pertime_dir20_tp25_lab2_t1 = var_pertime(crestends_dir20_tp25_lab2_t1, time_dir20_tp25_lab2_t1)

fbr_abs_pertime_dir20_tp25_all_t1 = var_pertime(fbr_abs_dir20_tp25_all_t1, time_dir20_tp25_all_t1)
fbr_abs_pertime_dir20_tp25_lab1_t1 = var_pertime(fbr_abs_dir20_tp25_lab1_t1, time_dir20_tp25_lab1_t1)
fbr_abs_pertime_dir20_tp25_lab2_t1 = var_pertime(fbr_abs_dir20_tp25_lab2_t1, time_dir20_tp25_lab2_t1)

# calculate stats for second 10 min window
tstart = 10*60/0.2; tend = tstart + 10*60/0.2
time_dir20_tp25_all_t2, crestlen_dir20_tp25_all_t2, fbr_abs_dir20_tp25_all_t2, avgx_dir20_tp25_all_t2 = \
    subsample_by_time(tstart, tend, time_dir20_tp25_all_subarea, crestlen_dir20_tp25_all_subarea, fbr_abs_dir20_tp25_all_subarea, avgx_dir20_tp25_all_subarea)
time_dir20_tp25_lab1_t2, crestlen_dir20_tp25_lab1_t2, fbr_abs_dir20_tp25_lab1_t2, avgx_dir20_tp25_lab1_t2 = \
    subsample_by_time(tstart, tend, time_dir20_tp25_lab1_subarea, crestlen_dir20_tp25_lab1_subarea, fbr_abs_dir20_tp25_lab1_subarea, avgx_dir20_tp25_lab1_subarea)
time_dir20_tp25_lab2_t2, crestlen_dir20_tp25_lab2_t2, fbr_abs_dir20_tp25_lab2_t2, avgx_dir20_tp25_lab2_t2 = \
    subsample_by_time(tstart, tend, time_dir20_tp25_lab2_subarea, crestlen_dir20_tp25_lab2_subarea, fbr_abs_dir20_tp25_lab2_subarea, avgx_dir20_tp25_lab2_subarea)

crestends_dir20_tp25_all_t2 = subset_time(crestends_dir20_tp25_all, time_dir20_tp25_all_subarea, tstart, tend)
crestends_dir20_tp25_lab1_t2 = subset_time(crestends_dir20_tp25_lab1, time_dir20_tp25_lab1_subarea, tstart, tend)
crestends_dir20_tp25_lab2_t2 = subset_time(crestends_dir20_tp25_lab2, time_dir20_tp25_lab2_subarea, tstart, tend)

crestends_pertime_dir20_tp25_all_t2 = var_pertime(crestends_dir20_tp25_all_t2, time_dir20_tp25_all_t2)
crestends_pertime_dir20_tp25_lab1_t2 = var_pertime(crestends_dir20_tp25_lab1_t2, time_dir20_tp25_lab1_t2)
crestends_pertime_dir20_tp25_lab2_t2 = var_pertime(crestends_dir20_tp25_lab2_t2, time_dir20_tp25_lab2_t2)

fbr_abs_pertime_dir20_tp25_all_t2 = var_pertime(fbr_abs_dir20_tp25_all_t2, time_dir20_tp25_all_t2)
fbr_abs_pertime_dir20_tp25_lab1_t2 = var_pertime(fbr_abs_dir20_tp25_lab1_t2, time_dir20_tp25_lab1_t2)
fbr_abs_pertime_dir20_tp25_lab2_t2 = var_pertime(fbr_abs_dir20_tp25_lab2_t2, time_dir20_tp25_lab2_t2)

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
dirspread = np.array([0.24,3.40,9.65,16.39,22.06,25.67])

### lab data ###
labcrestends = np.array([1.3679, 3.4496, 5.0379, 5.7969, 6.1887])
labmeancrest = np.array([16.5014, 7.6756, 5.0073, 4.5050, 4.1587])
labdirspread = np.array([2.3285, 6.2768, 11.6920, 16.3781, 22.4739])

####### CREST LENGTH ######
meancrest_all = np.array([np.mean(crestlen_dir0_all_subarea), 
                      np.mean(crestlen_dir5_all_subarea), 
                      np.mean(crestlen_dir10_all_subarea), 
                      np.mean(crestlen_dir20_all_subarea), 
                      np.mean(crestlen_dir30_all_subarea), 
                      np.mean(crestlen_dir40_all_subarea)])

meancrest_lab1 = np.array([np.mean(crestlen_dir0_lab1_subarea), 
                      np.mean(crestlen_dir5_lab1_subarea), 
                      np.mean(crestlen_dir10_lab1_subarea), 
                      np.mean(crestlen_dir20_lab1_subarea), 
                      np.mean(crestlen_dir30_lab1_subarea), 
                      np.mean(crestlen_dir40_lab1_subarea)])

meancrest_lab2 = np.array([np.mean(crestlen_dir0_lab2_subarea), 
                      np.mean(crestlen_dir5_lab2_subarea), 
                      np.mean(crestlen_dir10_lab2_subarea), 
                      np.mean(crestlen_dir20_lab2_subarea), 
                      np.mean(crestlen_dir30_lab2_subarea), 
                      np.mean(crestlen_dir40_lab2_subarea)])

meancrest_all_t1 = np.array([np.mean(crestlen_dir0_all_t1), 
                      np.mean(crestlen_dir5_all_t1), 
                      np.mean(crestlen_dir10_all_t1), 
                      np.mean(crestlen_dir20_all_t1), 
                      np.mean(crestlen_dir30_all_t1), 
                      np.mean(crestlen_dir40_all_t1)])

meancrest_all_t2 = np.array([np.mean(crestlen_dir0_all_t2), 
                      np.mean(crestlen_dir5_all_t2), 
                      np.mean(crestlen_dir10_all_t2), 
                      np.mean(crestlen_dir20_all_t2), 
                      np.mean(crestlen_dir30_all_t2), 
                      np.mean(crestlen_dir40_all_t2)])

tmp = (np.mean((meancrest_all_t1 - meancrest_all_t2)/meancrest_all_t1))*100
print('Percentage difference in crest length (time): %0.2f' % tmp)

meancrest_lab1_t1 = np.array([np.mean(crestlen_dir0_lab1_t1), 
                      np.mean(crestlen_dir5_lab1_t1), 
                      np.mean(crestlen_dir10_lab1_t1), 
                      np.mean(crestlen_dir20_lab1_t1), 
                      np.mean(crestlen_dir30_lab1_t1), 
                      np.mean(crestlen_dir40_lab1_t1)])

meancrest_lab1_t2 = np.array([np.mean(crestlen_dir0_lab1_t2), 
                      np.mean(crestlen_dir5_lab1_t2), 
                      np.mean(crestlen_dir10_lab1_t2), 
                      np.mean(crestlen_dir20_lab1_t2), 
                      np.mean(crestlen_dir30_lab1_t2), 
                      np.mean(crestlen_dir40_lab1_t2)])

meancrest_lab2_t1 = np.array([np.mean(crestlen_dir0_lab2_t1), 
                      np.mean(crestlen_dir5_lab2_t1), 
                      np.mean(crestlen_dir10_lab2_t1), 
                      np.mean(crestlen_dir20_lab2_t1), 
                      np.mean(crestlen_dir30_lab2_t1), 
                      np.mean(crestlen_dir40_lab2_t1)])

meancrest_lab2_t2 = np.array([np.mean(crestlen_dir0_lab2_t2), 
                      np.mean(crestlen_dir5_lab2_t2), 
                      np.mean(crestlen_dir10_lab2_t2), 
                      np.mean(crestlen_dir20_lab2_t2), 
                      np.mean(crestlen_dir30_lab2_t2), 
                      np.mean(crestlen_dir40_lab2_t2)])

#### CREST ENDS ####
lab_sz = np.array([4.2900, 4.1300, 3.9500, 3.9000, 4.0100])
Wtank = 26.5
Wmod = 55.0
SZtank = np.mean(lab_sz)
#SZmod = 31.5+22 - 49

Atank = Wtank*SZtank
#Amod = Wmod*SZmod
#Amodlab = Wtank*SZmod
#sz = np.array([27.05, 27.55, 27.40, 27.40, 27.40, 27.40, 25.35, 27.95])+22
sz = 23.5+22
mean_sz = np.mean(shore[:-2]-sz)

meancrestends_all = np.array([np.mean(crestends_pertime_dir0_all), 
                          np.mean(crestends_pertime_dir5_all), 
                          np.mean(crestends_pertime_dir10_all), 
                          np.mean(crestends_pertime_dir20_all), 
                          np.mean(crestends_pertime_dir30_all), 
                          np.mean(crestends_pertime_dir40_all)])

crestend_density_all = meancrestends_all/(mean_sz*Wmod)
meancrestends_all = crestend_density_all*Atank


meancrestends_all_t1 = np.array([np.mean(crestends_pertime_dir0_all_t1), 
                          np.mean(crestends_pertime_dir5_all_t1), 
                          np.mean(crestends_pertime_dir10_all_t1), 
                          np.mean(crestends_pertime_dir20_all_t1), 
                          np.mean(crestends_pertime_dir30_all_t1), 
                          np.mean(crestends_pertime_dir40_all_t1)])

crestend_density_all_t1 = meancrestends_all_t1/(mean_sz*Wmod)
meancrestends_all_t1 = crestend_density_all_t1*Atank

meancrestends_all_t2 = np.array([np.mean(crestends_pertime_dir0_all_t2), 
                        np.mean(crestends_pertime_dir5_all_t2), 
                        np.mean(crestends_pertime_dir10_all_t2), 
                        np.mean(crestends_pertime_dir20_all_t2), 
                        np.mean(crestends_pertime_dir30_all_t2), 
                        np.mean(crestends_pertime_dir40_all_t2)])

crestend_density_all_t2 = meancrestends_all_t2/(mean_sz*Wmod)
meancrestends_all_t2 = crestend_density_all_t2*Atank

#### VORTICITY INJECTION ####
####################################################
medianfbr_abs_pertime_all = np.array([np.median(fbr_abs_pertime_dir0_all[np.where(np.isfinite(fbr_abs_pertime_dir0_all)==True)[0]]), 
                      np.median(fbr_abs_pertime_dir5_all[np.where(np.isfinite(fbr_abs_pertime_dir5_all)==True)[0]]), 
                      np.median(fbr_abs_pertime_dir10_all[np.where(np.isfinite(fbr_abs_pertime_dir10_all)==True)[0]]), 
                      np.median(fbr_abs_pertime_dir20_all[np.where(np.isfinite(fbr_abs_pertime_dir20_all)==True)[0]]), 
                      np.median(fbr_abs_pertime_dir30_all[np.where(np.isfinite(fbr_abs_pertime_dir30_all)==True)[0]]), 
                      np.median(fbr_abs_pertime_dir40_all[np.where(np.isfinite(fbr_abs_pertime_dir40_all)==True)[0]])])

medianfbr_abs_pertime_all_t1 = np.array([np.median(fbr_abs_pertime_dir0_all_t1[np.where(np.isfinite(fbr_abs_pertime_dir0_all_t1)==True)[0]]), 
                      np.median(fbr_abs_pertime_dir5_all_t1[np.where(np.isfinite(fbr_abs_pertime_dir5_all_t1)==True)[0]]), 
                      np.median(fbr_abs_pertime_dir10_all_t1[np.where(np.isfinite(fbr_abs_pertime_dir10_all_t1)==True)[0]]), 
                      np.median(fbr_abs_pertime_dir20_all_t1[np.where(np.isfinite(fbr_abs_pertime_dir20_all_t1)==True)[0]]), 
                      np.median(fbr_abs_pertime_dir30_all_t1[np.where(np.isfinite(fbr_abs_pertime_dir30_all_t1)==True)[0]]), 
                      np.median(fbr_abs_pertime_dir40_all_t1[np.where(np.isfinite(fbr_abs_pertime_dir40_all_t1)==True)[0]])])

medianfbr_abs_pertime_all_t2 = np.array([np.median(fbr_abs_pertime_dir0_all_t2[np.where(np.isfinite(fbr_abs_pertime_dir0_all_t2)==True)[0]]), 
                      np.median(fbr_abs_pertime_dir5_all_t2[np.where(np.isfinite(fbr_abs_pertime_dir5_all_t2)==True)[0]]), 
                      np.median(fbr_abs_pertime_dir10_all_t2[np.where(np.isfinite(fbr_abs_pertime_dir10_all_t2)==True)[0]]), 
                      np.median(fbr_abs_pertime_dir20_all_t2[np.where(np.isfinite(fbr_abs_pertime_dir20_all_t2)==True)[0]]), 
                      np.median(fbr_abs_pertime_dir30_all_t2[np.where(np.isfinite(fbr_abs_pertime_dir30_all_t2)==True)[0]]), 
                      np.median(fbr_abs_pertime_dir40_all_t2[np.where(np.isfinite(fbr_abs_pertime_dir40_all_t2)==True)[0]])])

#############################################################
color1='#003f5c'
color2='#444e86'
color3='#955196'
color4='#dd5182'
color5='#ff6e54'
color6='#ffa600'
color = 'tab:grey'

fig, ax = plt.subplots(ncols=6, figsize=(8,5.5))
lwidth = 1
msize = 5
fsize = 10

xmin = 0.08
ymin = 0.08
width = 0.19
height = 0.37
xoffset = 0.12
yoffset = 0.1
formatter = ticker.ScalarFormatter(useMathText=True)

ax[0].plot(dirspread, (meancrest_all_t1+meancrest_all_t2)/2, '-^', linewidth=lwidth, markersize=msize, color=color, label=r'$\mathrm{Model\ (full\ grid)}$')
ax[0].plot(dirspread, ((meancrest_lab1_t1+meancrest_lab2_t1)/2 + (meancrest_lab1_t2+meancrest_lab2_t2)/2)/2, 'o', alpha=0.7, linewidth=lwidth-1, markersize=msize, color=color, label=r'$\mathrm{Model\ (lab\ grid)}$')
ax[0].plot(labdirspread, labmeancrest, '-o', linewidth=lwidth, markersize=msize, color=color3, alpha=0.8, label=r'$\mathrm{Observations}$')
ax[0].legend(loc='upper right', fontsize=fsize-2)
ax[0].set_ylim(0,35)
ax[0].set_xlim(0,30)
ax[0].set_ylabel(r'$\overline{\lambda_{c}} (m)}$', fontsize=fsize)
ax[0].set_xlabel(r'$\sigma_\theta$ ($\degree$)', fontsize=fsize)
ax[0].set_position([xmin, ymin+yoffset+height, width, height])
if 1:
    L = 5.2
    dirs = np.linspace(0, 40)
    lcrest = (1/4)*L/np.sin(dirs*np.pi/180)
    ax[0].plot(dirs, lcrest, '--', color=color, alpha=0.5, linewidth=lwidth)
ax[0].grid(True)
ax[0].text(0.5, 1.5, r'$(a)$', fontsize=fsize+3)
ax[0].tick_params(axis='x', which='major', labelsize=fsize)
ax[0].tick_params(axis='y', which='major', labelsize=fsize) 
ax[0].set_xticks([0,5,10,15,20,25,30])
ax[0].set_yticks([5, 15, 25, 35])
ax[0].yaxis.set_major_formatter(formatter)

crestlen_15 = (np.mean(crestlen_dir20_tp15_all_t1)+np.mean(crestlen_dir20_tp15_all_t2))/2
crestlen_2 = (np.mean(crestlen_dir20_all_t1)+np.mean(crestlen_dir20_all_t2))/2
crestlen_25 = (np.mean(crestlen_dir20_tp25_all_t1)+np.mean(crestlen_dir20_tp25_all_t2))/2
ax[1].plot(np.array([1.5, 2, 2.5]), np.array([crestlen_15, crestlen_2, crestlen_25]), '-^', color=color6, linewidth=lwidth, markersize=msize)
ax[1].grid(True)
ax[1].tick_params(axis='x', which='major', labelsize=fsize)
ax[1].tick_params(axis='y', which='major', labelsize=fsize) 
ax[1].set_xticks([1, 1.5, 2, 2.5, 3])
ax[1].set_yticks([3, 4, 5, 6])
ax[1].set_xlabel(r'$T_p$ $(\mathrm{s})$', fontsize=fsize)
ax[1].set_ylabel(r'$\overline{\lambda_c}$ ($\mathrm{m}^{-2}$)', fontsize=fsize)
ax[1].set_position([xmin, ymin, width, height])
ax[1].text(1.1, 3.06, r'$(d)$', fontsize=fsize+3)
ax[1].yaxis.set_major_formatter(formatter)

ax[2].plot(dirspread, (crestend_density_all_t1+crestend_density_all_t2)/2, '-^', linewidth=lwidth, markersize=msize, color=color)
ax[2].plot(labdirspread, labcrestends/Atank, '-o', linewidth=lwidth, markersize=msize, color=color3, alpha=0.8)
ax[2].set_ylim(0,0.06)
ax[2].set_ylabel(r'$d_{ce}$ ($\mathrm{m}^{-2}$)', fontsize=fsize)
ax[2].set_xlabel(r'$\sigma_\theta$ ($\degree$)', fontsize=fsize)
ax[2].grid(True)
ax[2].text(0.5, 0.003, r'$(b)$', fontsize=fsize+3)
ax[2].tick_params(axis='x', which='major', labelsize=fsize)
ax[2].tick_params(axis='y', which='major', labelsize=fsize)
ax[2].set_xticks([0, 5, 10, 15, 20, 25, 30])
ax[2].set_yticks([0, 0.02, 0.04, 0.06])
ax[2].yaxis.set_major_formatter(formatter)

ax01 = ax[2].twinx()
ax01.set_ylim(0,0.06*Atank)
ax01.plot(dirspread, (meancrestends_all_t1+meancrestends_all_t2)/2, '-^', linewidth=lwidth, markersize=msize, color=color)
ax01.plot(labdirspread, labcrestends, '-o', linewidth=lwidth, markersize=msize, color=color3, alpha=0.8)
ax01.set_ylabel(r'$d_{ce}A_{lab}$ $(\#)$', fontsize=fsize)
ax01.tick_params(axis='y', which='major', labelsize=fsize) 
ax01.set_yticks([0, 1.5, 3, 4.5, 6])
ax01.set_xticks([0,5,10,15,20,25,30])
ax01.yaxis.set_major_formatter(formatter)
ax[2].set_position([xmin+xoffset+width, ymin+yoffset+height, width, height])

crestends15 = ((np.mean(crestends_pertime_dir20_tp15_all_t1)+np.mean(crestends_pertime_dir20_tp15_all_t2))/2)
crestend_density_15 = crestends15/(mean_sz*Wmod)
crestends_15 = crestend_density_15*Atank

crestends2 = ((np.mean(crestends_pertime_dir20_all_t1)+np.mean(crestends_pertime_dir20_all_t2))/2)
crestend_density_2 = crestends2/(mean_sz*Wmod)
crestends_2 = crestend_density_2*Atank

crestends25 = ((np.mean(crestends_pertime_dir20_tp25_all_t1)+np.mean(crestends_pertime_dir20_tp25_all_t2))/2)
crestend_density_25 = crestends25/(mean_sz*Wmod)
crestends_25 = crestend_density_25*Atank
ax[3].plot(np.array([1.5, 2, 2.5]), np.array([crestend_density_15, crestend_density_2, crestend_density_25]), '-^', color=color6, linewidth=lwidth, markersize=msize)
ax[3].grid(True)
ax[3].tick_params(axis='x', which='major', labelsize=fsize)
ax[3].tick_params(axis='y', which='major', labelsize=fsize) 
ax[3].set_yticks([0.04, 0.05, 0.06, 0.07])
ax[3].set_xticks([1, 1.5, 2, 2.5, 3])
ax[3].set_xlabel(r'$T_p$ $(\mathrm{s})$', fontsize=fsize)
ax[3].set_ylabel(r'$d_{ce}$ ($\mathrm{m}^{-2}$)', fontsize=fsize)
ax[3].text(1.1, 0.041, r'$(e)$', fontsize=fsize+3)
ax[3].set_ylim(0.04, 0.07)

ax3 = ax[3].twinx()
ax3.plot(np.array([1.5, 2, 2.5]), np.array([crestend_density_15, crestend_density_2, crestend_density_25])*Atank, '-^', color=color6, linewidth=lwidth, markersize=msize)
ax3.set_ylabel(r'$d_{ce}A_{lab}$ $(\#)$', fontsize=fsize)
ax3.tick_params(axis='y', which='major', labelsize=fsize) 
ax3.set_xticks([1, 1.5, 2, 2.5, 3])
ax3.set_yticks([4, 5, 6, 7])
ax3.set_ylim(0.04*Atank, 0.07*Atank)
ax[3].set_position([xmin+xoffset+width, ymin, width, height])

scalar = dx*dy
medianfbr_abs_pertime_den_all_t1 = medianfbr_abs_pertime_all_t1/(mean_sz*Wmod)*scalar
medianfbr_abs_pertime_den_all_t2 = medianfbr_abs_pertime_all_t2/(mean_sz*Wmod)*scalar

labfbrdir = np.array([2, 10, 18, 23])
labfbr = np.array([0.4, 0.8, 0.85, 0.88])
labscale = 1

ax[4].plot(dirspread, (medianfbr_abs_pertime_den_all_t1 + medianfbr_abs_pertime_den_all_t2)/2, '-^', linewidth=lwidth, markersize=msize, color=color)
ax[4].plot(labfbrdir, labfbr/Atank*labscale, '-o', color=color3, markersize=msize, linewidth=lwidth)
ax[4].set_ylabel(r'$d_\Omega$ ($\mathrm{m s}^{-2}$)', fontsize=fsize)
ax[4].set_xlabel(r'$\sigma_\theta$ ($\degree$)', fontsize=fsize)
ax[4].grid(True)
ax[4].text(0.5, 0.01, r'$(c)$', fontsize=fsize+3)
ax[4].set_ylim(0, 0.4)
ax[4].tick_params(axis='x', which='major', labelsize=fsize)
ax[4].tick_params(axis='y', which='major', labelsize=fsize) 
ax[4].set_xticks([0,5,10,15,20,25,30])
ax[4].set_yticks([0, 0.1, 0.2, 0.3, 0.4])

ax02 = ax[4].twinx()
ax02.plot(dirspread, (medianfbr_abs_pertime_den_all_t1 + medianfbr_abs_pertime_den_all_t2)/2*Atank, '-^', linewidth=lwidth, markersize=msize, color=color)
ax02.plot(labfbrdir, labfbr*labscale, '-o', color=color3, markersize=msize, linewidth=lwidth)
ax02.set_ylim(0, 0.4*Atank)
ax02.set_xticks([0,5,10,15,20,25,30])
ax02.set_yticks([0, 10, 20, 30, 40])
ax02.tick_params(axis='y', which='major', labelsize=fsize)
ax02.set_ylabel(r'$\Omega_{sz}$ $(\mathrm{s^{-2}})$') 
ax[4].set_position([xmin+2*(xoffset+width)+0.25*xoffset, ymin+yoffset+height, width, height])

medianfbr_den_15 = ((np.median(fbr_abs_pertime_dir20_tp15_all_t1)+np.median(fbr_abs_pertime_dir20_tp15_all_t2))/2)/((shore[-2]-22.75-22)*Wmod)*scalar
medianfbr_den_2 = ((np.median(fbr_abs_pertime_dir20_all_t1)+np.median(fbr_abs_pertime_dir20_all_t2))/2)/(mean_sz*Wmod)*scalar
medianfbr_den_25 = ((np.median(fbr_abs_pertime_dir20_tp25_all_t1)+np.median(fbr_abs_pertime_dir20_tp25_all_t2))/2)/((shore[-1]-23.60-22)*Wmod)*scalar

ax[5].plot(np.array([1.5, 2, 2.5]), np.array([medianfbr_den_15, medianfbr_den_2, medianfbr_den_25]), '-^', color=color6, linewidth=lwidth, markersize=msize)
ax[5].grid(True)
ax[5].tick_params(axis='x', which='major', labelsize=fsize)
ax[5].tick_params(axis='y', which='major', labelsize=fsize) 
ax[5].set_yticks([0.22, 0.24, 0.26, 0.28, 0.3])
ax[5].set_xticks([1, 1.5, 2, 2.5, 3])
ax[5].set_xlabel(r'$T_p$ $(\mathrm{s})$', fontsize=fsize)
ax[5].set_ylabel(r'$d_\Omega$ ($\mathrm{m s}^{-2}$)', fontsize=fsize)
ax[5].text(1.1, 0.225, r'$(f)$', fontsize=fsize+3)
ax[5].set_ylim(0.22, 0.3)

ax5 = ax[5].twinx()
ax5.plot(np.array([1.5, 2, 2.5]), np.array([medianfbr_den_15, medianfbr_den_2, medianfbr_den_25])*Atank, '-^', color=color6, linewidth=lwidth, markersize=msize)
ax5.tick_params(axis='y', which='major', labelsize=fsize)
ax5.set_ylabel(r'$\Omega_{sz}$ $(\mathrm{s^{-2}})$') 
ax5.set_xticks([1, 1.5, 2, 2.5, 3])
ax5.set_yticks([24, 26, 28, 30, 32])
ax5.set_ylim(0.22*Atank, 0.3*Atank)
ax[5].set_position([xmin+2*(xoffset+width)+0.25*xoffset, ymin, width, height])

fig.savefig(os.path.join(plotsavedir, 'crestlen_ends_medianfbr_wnewall.png'))

###############################################
lwidth = 1.5

fig, ax = plt.subplots(figsize=(8,6), ncols=2, nrows=2)
nbins = 50
var = fbr_abs_pertime_dir0_all[np.isfinite(fbr_abs_pertime_dir0_all)]
counts1, bins1 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[0,0].plot(bins1[:-1], counts1, color=color1, linewidth=lwidth, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[0], alpha=1)

var = fbr_abs_pertime_dir5_all[np.isfinite(fbr_abs_pertime_dir5_all)]
counts5, bins5 = np.histogram(np.log10(var[var>0]), bins=nbins+nbins+20)
ax[0,0].plot(bins5[:-1], counts5, color=color2, linewidth=lwidth, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[1])

var = fbr_abs_pertime_dir10_all[np.isfinite(fbr_abs_pertime_dir10_all)]
counts10, bins10 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[0,0].plot(bins10[:-1], counts10, color=color3, linewidth=lwidth, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[2])

var = fbr_abs_pertime_dir20_all[np.isfinite(fbr_abs_pertime_dir20_all)]
counts20, bins20 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[0,0].plot(bins20[:-1], counts20, color=color4, linewidth=lwidth, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[3])

var = fbr_abs_pertime_dir30_all[np.isfinite(fbr_abs_pertime_dir30_all)]
counts30, bins30 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[0,0].plot(bins30[:-1], counts30, color=color5, linewidth=lwidth, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[4], alpha=0.5)

var = fbr_abs_pertime_dir40_all[np.isfinite(fbr_abs_pertime_dir40_all)]
counts40, bins40 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[0,0].plot(bins40[:-1], counts40, color=color6, linewidth=lwidth, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[5], alpha=0.5)
ax[0,0].legend(loc='best', fontsize=fsize-3)
ax[0,0].set_xlim(2,5)
ax[0,0].grid(True)
ax[0,0].text(4.6, 550, r'$(a)$', fontsize=16)
ax[0,0].set_ylabel(r'$\mathrm{Count}$')

ax[1,0].set_ylabel(r'$\mathrm{Count}$')
ax[1,0].set_xlabel(r'$\log (V_{sz})$ ($\mathrm{m s}^{-2}$)')
###

var = fbr_abs_dir0_all_subarea[np.isfinite(fbr_abs_dir0_all_subarea)]
counts1, bins1 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[0,1].plot(bins1[:-1], counts1, color=color1, linewidth=lwidth, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[0], alpha=1)

var = fbr_abs_dir5_all_subarea[np.isfinite(fbr_abs_dir5_all_subarea)]
counts5, bins5 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[0,1].plot(bins5[:-1], counts5, color=color2, linewidth=lwidth, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[1])

var = fbr_abs_dir10_all_subarea[np.isfinite(fbr_abs_dir10_all_subarea)]
counts10, bins10 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[0,1].plot(bins10[:-1], counts10, color=color3, linewidth=lwidth, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[2])

var = fbr_abs_dir20_all_subarea[np.isfinite(fbr_abs_dir20_all_subarea)]
counts20, bins20 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[0,1].plot(bins20[:-1], counts20, color=color4, linewidth=lwidth, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[3])

var = fbr_abs_dir30_all_subarea[np.isfinite(fbr_abs_dir30_all_subarea)]
counts30, bins30 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[0,1].plot(bins30[:-1], counts30, color=color5, linewidth=lwidth, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[4], alpha=0.5)

var = fbr_abs_dir40_all_subarea[np.isfinite(fbr_abs_dir40_all_subarea)]
counts40, bins40 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[0,1].plot(bins40[:-1], counts40, color=color6, linewidth=lwidth, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[5], alpha=0.5)
ax[0,1].set_xlim(1.5,5)

ax[0,1].set_ylabel(r'$\mathrm{Count}$')
ax[0,1].grid(True)
ax[0,1].text(4.6, 5600, r'$(b)$', fontsize=16)
ax[0,1].set_yticks([0, 1000, 2000, 3000, 4000, 5000, 6000])

ax[1,1].set_xlabel(r'$\log(V_c)$ ($\mathrm{m s}^{-2}$)')

var = fbr_abs_pertime_dir20_tp25_all[np.isfinite(fbr_abs_pertime_dir20_tp25_all)]
counts20_tp25, bins20_tp25 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[1,0].plot(bins20_tp25[:-1], counts20_tp25, color=color1, linewidth=lwidth, label=r'$T_p$ = 2.5 $\mathrm{s}$' % dirspread[3])

var = fbr_abs_pertime_dir20_all[np.isfinite(fbr_abs_pertime_dir20_all)]
counts20, bins20 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[1,0].plot(bins20[:-1], counts20, color=color4, linewidth=lwidth, label=r'$T_p$ = 2.0 $\mathrm{s}$' % dirspread[3])

var = fbr_abs_pertime_dir20_tp15_all[np.isfinite(fbr_abs_pertime_dir20_tp15_all)]
counts20_tp15, bins20_tp15 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[1,0].plot(bins20_tp15[:-1], counts20_tp15, color=color6, linewidth=lwidth, label=r'$T_p$ = 1.5 $\mathrm{s}$' % dirspread[3])
ax[1,0].legend(loc='best', fontsize=fsize-3)
ax[1,0].grid(True)
ax[1,0].set_xlim(2, 5)
ax[1,0].set_xticks([2.0, 2.5, 3, 3.5, 4, 4.5, 5.0])
ax[1,0].set_yticks([0, 100, 200, 300, 400, 500, 600])
ax[1,0].text(4.6, 550, r'$(c)$', fontsize=16)

var = fbr_abs_dir20_tp25_all_subarea[np.isfinite(fbr_abs_dir20_tp25_all_subarea)]
counts20_tp25, bins20_tp25 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[1,1].plot(bins20_tp25[:-1], counts20_tp25, color=color1, linewidth=lwidth, label=r'$T_p$ = 2.5 $\mathrm{s}$' % dirspread[3])

var = fbr_abs_dir20_all_subarea[np.isfinite(fbr_abs_dir20_all_subarea)]
counts20, bins20 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[1,1].plot(bins20[:-1], counts20, color=color4, linewidth=lwidth, label=r'$T_p$ = 2.0 $\mathrm{s}$' % dirspread[3])

var = fbr_abs_dir20_tp15_all_subarea[np.isfinite(fbr_abs_dir20_tp15_all_subarea)]
counts20_tp15, bins20_tp15 = np.histogram(np.log10(var[var>0]), bins=nbins)
ax[1,1].plot(bins20_tp15[:-1], counts20_tp15, color=color6, linewidth=lwidth, label=r'$T_p$ = 1.5 $\mathrm{s}$' % dirspread[3])
ax[1,1].grid(True)
ax[1,1].set_yticks([0, 1000, 2000, 3000, 4000, 5000, 6000])
ax[1,1].text(4.6, 5600, r'$(d)$', fontsize=16)
ax[1,1].set_ylabel(r'$\mathrm{Count}$')

fig.tight_layout()
fig.savefig(os.path.join(plotsavedir, 'fbr_abs_hist_pertime_percrest.png'))



##########################################################################
import matplotlib.colors as colors

ylim = 4*10**4
fig, ax = plt.subplots(figsize=(8,7.5), nrows=3, ncols=3, sharex=True, sharey=True)
alpha = 0.5
nbins = 50
vmax = 10**4
fsize=16
p00 = ax[0,0].hist2d(crestlen_dir0_all_subarea[np.isfinite(fbr_abs_dir0_all_subarea)], fbr_abs_dir0_all_subarea[np.isfinite(fbr_abs_dir0_all_subarea)], bins=nbins, norm=colors.LogNorm(vmax=vmax), cmap=cmo.matter)
cbar00 = fig.colorbar(p00[3], ax=ax[0,0], label=r'$\mathrm{Count}$')
ax[0,0].set_ylabel(r'$\Omega_c$ ($\mathrm{m s}^{-2}$)')
ax[0,0].set_title(r'$\sigma_\theta = %.1f \degree$' % dirspread[0])
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax[0,0].yaxis.set_major_formatter(formatter)
ax[0,0].text(2, ylim, r'$\mathrm{(a)}$', fontsize=fsize)
ax[0,0].grid(True)

p01 = ax[0,1].hist2d(crestlen_dir5_all_subarea[np.isfinite(fbr_abs_dir5_all_subarea)], fbr_abs_dir5_all_subarea[np.isfinite(fbr_abs_dir5_all_subarea)], bins=nbins, norm=colors.LogNorm(vmax=vmax), cmap=cmo.matter)
cbar01 = fig.colorbar(p01[3], ax=ax[0,1], label=r'$\mathrm{Count}$')
ax[0,1].set_title(r'$\sigma_\theta = %.1f \degree$' % dirspread[1])
ax[0,1].yaxis.set_major_formatter(formatter)
ax[0,1].text(2, ylim, r'$\mathrm{(b)}$', fontsize=fsize)
ax[0,1].grid(True)

p02 = ax[0,2].hist2d(crestlen_dir10_all_subarea[np.isfinite(fbr_abs_dir10_all_subarea)], fbr_abs_dir10_all_subarea[np.isfinite(fbr_abs_dir10_all_subarea)], bins=nbins, norm=colors.LogNorm(vmax=vmax), cmap=cmo.matter)
cbar02 = fig.colorbar(p02[3], ax=ax[0,2], label=r'$\mathrm{Count}$')
ax[0,2].set_title(r'$\sigma_\theta = %.1f \degree$' % dirspread[2])
ax[0,2].yaxis.set_major_formatter(formatter)
ax[0,2].text(2, ylim, r'$\mathrm{(c)}$', fontsize=fsize)
ax[0,2].grid(True)

p10 = ax[1,0].hist2d(crestlen_dir20_all_subarea[np.isfinite(fbr_abs_dir20_all_subarea)], fbr_abs_dir20_all_subarea[np.isfinite(fbr_abs_dir20_all_subarea)], bins=nbins, norm=colors.LogNorm(vmax=vmax), cmap=cmo.matter)
cbar10 = fig.colorbar(p10[3], ax=ax[1,0], label=r'$\mathrm{Count}$')
ax[1,0].set_ylabel(r'$\Omega_c$ $(\mathrm{m s}^{-2})$')
ax[1,0].set_title(r'$\sigma_\theta = %.1f \degree$' % dirspread[3])
ax[1,0].yaxis.set_major_formatter(formatter)
ax[1,0].text(2, ylim, r'$\mathrm{(d)}$', fontsize=fsize)
ax[1,0].grid(True)

p11 = ax[1,1].hist2d(crestlen_dir30_all_subarea[np.isfinite(fbr_abs_dir30_all_subarea)], fbr_abs_dir30_all_subarea[np.isfinite(fbr_abs_dir30_all_subarea)], bins=nbins, norm=colors.LogNorm(vmax=vmax), cmap=cmo.matter)
cbar11 = fig.colorbar(p11[3], ax=ax[1,1], label=r'$\mathrm{Count}$')
ax[1,1].set_title(r'$\sigma_\theta = %.1f \degree$' % dirspread[4])
ax[1,1].yaxis.set_major_formatter(formatter)
ax[1,1].text(2, ylim, r'$\mathrm{(e)}$', fontsize=fsize)
ax[1,1].grid(True)

p12 = ax[1,2].hist2d(crestlen_dir40_all_subarea[np.isfinite(fbr_abs_dir40_all_subarea)], fbr_abs_dir40_all_subarea[np.isfinite(fbr_abs_dir40_all_subarea)], bins=nbins, norm=colors.LogNorm(vmax=vmax), cmap=cmo.matter)
cbar12 = fig.colorbar(p12[3], ax=ax[1,2], label=r'$\mathrm{Count}$')
ax[1,2].set_title(r'$\sigma_\theta = %.1f \degree$' % dirspread[5])
ax[1,2].yaxis.set_major_formatter(formatter)
ax[1,2].text(2, ylim, r'$\mathrm{(f)}$', fontsize=fsize)
ax[1,2].set_ylim(0,4.5*10**4)
ax[1,2].grid(True)


p20 = ax[2,0].hist2d(crestlen_dir20_tp15_all_subarea[np.isfinite(fbr_abs_dir20_tp15_all_subarea)], fbr_abs_dir20_tp15_all_subarea[np.isfinite(fbr_abs_dir20_tp15_all_subarea)], bins=nbins, norm=colors.LogNorm(vmax=vmax), cmap=cmo.matter)
cbar20 = fig.colorbar(p20[3], ax=ax[2,0], label=r'$\mathrm{Count}$')
ax[2,0].set_ylabel(r'$\Omega_c$ ($\mathrm{m s}^{-2}$)')
ax[2,0].set_title(r'$T_p = 1.5\ \mathrm{s}$')
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
ax[2,0].yaxis.set_major_formatter(formatter)
ax[2,0].text(2, ylim, r'$\mathrm{(g)}$', fontsize=fsize)
ax[2,0].grid(True)
ax[2,0].set_xlabel(r'$\lambda_{c}$ $\mathrm{(m)}$')

p21 = ax[2,1].hist2d(crestlen_dir20_all_subarea[np.isfinite(fbr_abs_dir20_all_subarea)], fbr_abs_dir20_all_subarea[np.isfinite(fbr_abs_dir20_all_subarea)], bins=nbins, norm=colors.LogNorm(vmax=vmax), cmap=cmo.matter)
cbar21 = fig.colorbar(p21[3], ax=ax[2,1], label=r'$\mathrm{Count}$')
ax[2,1].set_title(r'$T_p = 2.0\ \mathrm{s}$')
ax[2,1].yaxis.set_major_formatter(formatter)
ax[2,1].text(2, ylim, r'$\mathrm{(h)}$', fontsize=fsize)
ax[2,1].grid(True)
ax[2,1].set_xlabel(r'$\lambda_{c}$ $\mathrm{(m)}$')

p22 = ax[2,2].hist2d(crestlen_dir20_tp25_all_subarea[np.isfinite(fbr_abs_dir20_tp25_all_subarea)], fbr_abs_dir20_tp25_all_subarea[np.isfinite(fbr_abs_dir20_tp25_all_subarea)], bins=nbins, norm=colors.LogNorm(vmax=vmax), cmap=cmo.matter)
cbar22 = fig.colorbar(p22[3], ax=ax[2,2], label=r'$\mathrm{Count}$')
ax[2,2].set_title(r'$T_p = 2.5\ \mathrm{s}$')
ax[2,2].yaxis.set_major_formatter(formatter)
ax[2,2].text(2, ylim, r'$\mathrm{(i)}$', fontsize=fsize)
ax[2,2].grid(True)
ax[2,2].set_ylim(0,4.5*10**4)
ax[2,2].set_xlim(0,55.5)
ax[2,2].set_xlabel(r'$\lambda_{c}$ $\mathrm{(m)}$')

fig.tight_layout()
fig.savefig(os.path.join(plotsavedir, 'crestlen_vs_fbr_2dhist.png'))


################################################################
###############################################################

nbins = 25
alpha = 0.5
fig, ax = plt.subplots(figsize=(8,3), sharex=True, sharey=True, ncols=2)
ax[0].hist(avgx_dir40_all_subarea-22, bins=nbins, color=color6, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[5])
ax[0].hist(avgx_dir30_all_subarea-22, bins=nbins, color=color5, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[4])
ax[0].hist(avgx_dir20_all_subarea-22, bins=nbins, color=color4, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[3])
ax[0].hist(avgx_dir10_all_subarea-22, bins=nbins, color=color3, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[2])
ax[0].hist(avgx_dir5_all_subarea-22, bins=nbins, color=color2, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[1])
ax[0].hist(avgx_dir0_all_subarea-22, bins=nbins, color=color1, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[0])
ax[0].axvline(23.5, linestyle='--', color=color, linewidth=lwidth)
ax[0].grid(True)
ax[0].set_xlabel(r'$x\ (\mathrm{m})$')
ax[0].set_ylabel(r'$\mathrm{Count}$')
ax[0].legend(loc='best', fontsize=fsize-3)
ax[0].text(29.95,22000, r'$\mathrm{(a)}$', fontsize=fsize)

ax[1].hist(avgx_dir20_tp15_all_subarea-22, bins=nbins, color=color1, alpha=alpha, label=r'$T_p = 1.5\ s$')
ax[1].hist(avgx_dir20_all_subarea-22, bins=nbins, color=color4, alpha=alpha, label=r'$T_p = 2.0\ s$')
ax[1].hist(avgx_dir20_tp25_all_subarea-22, bins=nbins, color=color6, alpha=alpha, label=r'$T_p = 2.5\ s$')
ax[1].grid(True)
ax[1].set_xlabel(r'$x\ (\mathrm{m})$')
ax[1].legend(loc='best', fontsize=fsize-3)
ax[1].set_xlim(15,31.5)
ax[1].axvline(23.5, linestyle='--', color=color, linewidth=lwidth)
ax[1].text(29.95,22000, r'$\mathrm{(b)}$', fontsize=fsize)
fig.tight_layout()
fig.savefig(os.path.join(plotsavedir, 'cross_position_hist_dirs_tp.png'))


################################################################
###############################################################

nbins = 18
fig, ax = plt.subplots(figsize=(8,6), ncols=2, nrows=2, sharex=True)
ax[0,0].hist(crestlen_dir40_all_subarea, bins=nbins, color=color6, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[5])
ax[0,0].hist(crestlen_dir30_all_subarea, bins=nbins, color=color5, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[4])
ax[0,0].hist(crestlen_dir20_all_subarea, bins=nbins, color=color4, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[3])
ax[0,0].hist(crestlen_dir10_all_subarea, bins=nbins, color=color3, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[2])
ax[0,0].hist(crestlen_dir5_all_subarea, bins=nbins, color=color2, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[1])
ax[0,0].hist(crestlen_dir0_all_subarea, bins=nbins, color=color1, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[0])
ax[0,0].grid(True)
ax[0,0].set_ylabel(r'$\mathrm{Count}$')
ax[0,0].set_xlim(0,55)
ax[0,0].legend(loc='upper right')
ax[0,0].set_yticks([0, 10000, 20000, 30000, 40000, 50000, 60000])
ax[0,0].text(2.65, 53000, r'$\mathrm{(a)}$', fontsize=18)

ax[0,1].hist(crestends_pertime_dir40_all, bins=nbins, color=color6, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[5])
ax[0,1].hist(crestends_pertime_dir30_all, bins=nbins, color=color5, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[4])
ax[0,1].hist(crestends_pertime_dir20_all, bins=nbins, color=color4, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[3])
ax[0,1].hist(crestends_pertime_dir10_all, bins=nbins, color=color3, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[2])
ax[0,1].hist(crestends_pertime_dir5_all, bins=nbins, color=color2, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[1])
ax[0,1].hist(crestends_pertime_dir0_all, bins=nbins, color=color1, alpha=alpha, label=r'$\sigma_\theta = %.1f \degree$' % dirspread[0])
ax[0,1].grid(True)
ax[0,1].set_xlim(0,55)
ax[0,1].set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
ax[0,1].text(2.65, 3100, r'$\mathrm{(b)}$', fontsize=18)

ax[1,0].hist(crestlen_dir20_tp15_all_subarea, bins=nbins, color=color1, label=r'$T_p = 1.5\ s$', alpha=alpha)
ax[1,0].grid(True)
ax[1,0].set_ylabel(r'$\mathrm{Count}$')
ax[1,0].set_xlim(0,55)
ax[1,0].set_xlabel(r'$\lambda_{c}\ \mathrm{(m)}$')
ax[1,0].legend(loc='upper right')

ax[1,0].hist(crestlen_dir20_all_subarea, bins=nbins, color=color4, label=r'$T_p = 2.0\ s$', alpha=alpha)
ax[1,0].grid(True)
ax[1,0].set_ylabel(r'$\mathrm{Count}$')
ax[1,0].set_xlim(0,55)
ax[1,0].legend(loc='upper right')

ax[1,0].hist(crestlen_dir20_tp25_all_subarea, bins=nbins, color=color6, label=r'$T_p = 2.5\ s$', alpha=alpha)
ax[1,0].grid(True)
ax[1,0].set_ylabel(r'$\mathrm{Count}$')
ax[1,0].set_xlim(0,55)
ax[1,0].legend(loc='upper right')
ax[1,0].text(2.65, 53000, r'$\mathrm{(c)}$', fontsize=18)
ax[1,0].set_yticks([0, 10000, 20000, 30000, 40000, 50000, 60000])

ax[1,1].hist(crestends_pertime_dir20_tp15_all, bins=nbins, color=color1, alpha=alpha)
ax[1,1].grid(True)
ax[1,1].set_xlim(0,55)
ax[1,1].set_xlabel(r'$\mathrm{Crest\ ends\ (\#)}$')

ax[1,1].hist(crestends_pertime_dir20_all, bins=nbins, color=color4, alpha=alpha)
ax[1,1].grid(True)
ax[1,1].set_xlim(0,55)

ax[1,1].hist(crestends_pertime_dir20_tp25_all, bins=nbins, color=color6, alpha=alpha)
ax[1,1].grid(True)
ax[1,1].set_xlim(0,55)
ax[1,1].set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
ax[1,1].text(2.65, 3100, r'$\mathrm{(d)}$', fontsize=18)

fig.tight_layout()
fig.savefig(os.path.join(plotsavedir, 'crest_stats_hist_periods_1row.png'))