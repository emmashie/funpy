import os
import numpy as np
import re
import pandas as pd
from funpy.model_utils import compute_spec, compute_Hsig_spectrally 
import glob
import funpy.wave_functions as wf

dat = np.loadtxt(os.path.join('/data2', 'enuss', 'TRC_cross-shore_profile.txt'), delimiter=',')
labx = dat[:,1]
labz = dat[:,0]-1.07

def find_pos(filepath, string):
	file = open(filepath, 'r')
	flag = 0
	index = -1 

	content = file.readlines()
	file = open(filepath, 'r')
	for line in file:
		index += 1
		if string in line:
			flag = 1
			break
	pos = float(re.findall(r"[-+]?\d*\.\d+|\d+", content[index])[0])
	file.close()
	return pos

def find_depth(xpos, x, dep):
	ind = np.argmin((xpos-x)**2)
	return dep[ind]

def hyd(p, dm, h0, rho=1000, g=9.81, patm=1):
	return (p-patm)/(rho*g) + dm - h0

def sl(phyd, dm, h0, dt, g=9.81, n=20):
	dsldt = pd.Series(np.gradient(phyd)/dt).rolling(n,center=True).mean().to_numpy()
	dsl2dt2 = pd.Series(np.gradient(dsldt)/dt).rolling(n,center=True).mean().to_numpy()
	return phyd - h0/(2*g)*(1 - (dm/h0)**2)*dsl2dt2

def snl(psl, dm, h0, dt, g=9.81, n=20):
	dsldt = pd.Series(np.gradient(psl)/dt).rolling(n,center=True).mean().to_numpy()
	dsl2dt2 = pd.Series(np.gradient(dsldt)/dt).rolling(n,center=True).mean().to_numpy()
	return psl - 1/g*(dsldt*dsldt + psl*dsl2dt2 - (dm/h0)**2*dsldt**2)

def load_press_insitu(filepath, duration=45, rho=1000, g=9.81):
	dat = np.loadtxt(filepath, comments='%')
	# find sensor positions
	xpos = find_pos(filepath, 'X:')
	ypos = find_pos(filepath, 'Y:')
	zpos = find_pos(filepath, 'Z:')
	#xpos, zpos = lab2model(xpos, zpos)
	dt = duration/len(dat)
	time = np.arange(0, 45, dt)*60
	return time, dat, xpos, ypos, zpos

def load_uv_insitu(filepath, duration=45*60):
	dat = np.loadtxt(filepath, comments='%')
	xpos = find_pos(filepath, 'X:')
	ypos = find_pos(filepath, 'Y:')
	zpos = find_pos(filepath, 'Z:')
	#xpos, zpos = lab2model(xpos, zpos)
	dt = duration/len(dat)
	time = np.arange(0, 45*60, dt)
	return time, dat, xpos, ypos, zpos	

def load_array(filepath, random, trial):
	obsdir = os.path.join(filepath, 'Random%d' % random, 'Trial%02d' % trial)

	press_flist = [file for file in glob.glob(os.path.join(obsdir,'press*.txt'))]
	u_flist = [file for file in glob.glob(os.path.join(obsdir,'u*.txt'))]
	v_flist = [file for file in glob.glob(os.path.join(obsdir,'v*.txt'))]

	### load data
	xpos = np.zeros(len(press_flist))
	ypos = np.zeros(len(press_flist))
	Hs = np.zeros(len(press_flist))
	eta = []
	u = []
	v = []

	I = np.min((len(press_flist), len(u_flist), len(v_flist)))
	for i in range(I):
		time, press_, xpos_, ypos_, zpos_ = load_press_insitu(press_flist[i])
		h0 = find_depth(xpos_, labx, -labz)
		dt = time[1]-time[0]
		dm = h0 - np.mean(press_)/(1000*9.81)
		press_hyd = hyd(press_, dm, h0)
		press_hyd = press_hyd - np.mean(press_hyd)
		freq, spec = compute_spec(press_hyd, dt=time[1]-time[0], n=20) ## check WL and OL
		press_sl = sl(press_hyd, dm, h0, dt)
		press_snl = snl(press_sl, dm, h0, dt)  
		xpos[i] = xpos_
		ypos[i] = ypos_
		eta.append(press_snl)
		valid = np.where(np.isfinite(press_snl)==True)[0]
		#freq, spec = compute_spec(press_snl[valid], dt=time[1]-time[0], n = 20)
		##  spec correction 
		k = np.zeros(len(freq))
		for j in range(len(freq)):
			if j>115 & j<230:
				k[j] = wf.wavenumber(freq[j], h0, wtype='deep')
			else:
				k[j] = wf.wavenumber(freq[j], h0, shallow_depth=0.2)
		spec_true = spec*(np.cosh(k*h0)/np.cosh(k*(h0+dm)))**-2
		Hs[i] = compute_Hsig_spectrally(freq, spec_true, fmin=0.25, fmax=1.2)
		time, u_, xpos_, ypos_, zpos_ = load_uv_insitu(u_flist[i])
		time, v_, xpos_, ypos_, zpos_ = load_uv_insitu(v_flist[i])	
		u.append(u_)
		v.append(v_)		

	eta = np.asarray(eta)
	u = np.asarray(u)
	v = np.asarray(v)
	return eta, Hs, u, v, xpos, ypos, dt

def array_ind(x, xpos):
	ind1 = np.where(x<xpos)[0]
	ind2 = np.where(x>xpos)[0]
	ind1_loc = np.mean(x[ind1])
	ind2_loc = np.mean(x[ind2])
	return ind1, ind2, ind1_loc, ind2_loc
