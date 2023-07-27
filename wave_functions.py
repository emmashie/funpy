#!/usr/bin/env python
import numpy as np 
from scipy.signal import welch

def calculate_spectra(pressure, WL=128, OL=64, fsamp=1):
	freq, Pxxf = welch(pressure, fs = fsamp, window = 'hann', nperseg = WL, noverlap = OL)
	return freq, Pxxf

def wave_type(depth, wavelength):
	if depth/wavelength<1/20:
		wtype = 'shallow'
	elif depth/wavelength>1/2:
		wtype = 'deep'
	else:
		wtype = 'intermediate'
	return wtype

def wavenum(omega,depth,tolerance=1e-6,max_iterations=10):
    # Linear dispersion relation wavenumber solver.
    #   k = wavenum(omega,depth) 
    # Solves for the wave-numbers 'k' for a given depth that satisfy the 
    # dispersion relation: omega^2 = g*k*tanh(k*depth) using the 
    # Newton-Raphson method.
    # @author: Sam Brenner
    # 
    g = 9.81;
    # initial guess based on Fenton and McKee, 1990:
    k0 = omega**2/g
    kn = k0* ( np.tanh((k0*depth)**(3/4)) )**(-2/3) 
    converged = 0 #convergence flag
    # iterate with a  Newton-Raphson method:
    for n in range(max_iterations): 
        err = np.abs( omega**2 - g*kn*np.tanh(kn*depth) )
        if all(err<tolerance):
            converged = 1
            break
        fkn = g*kn*np.tanh(kn*depth) - omega**2
        dfkn = g*np.tanh(kn*depth) + g*depth*kn/np.cosh(kn*depth)**2
        kn = kn - fkn/dfkn   
    if converged == 0:
        print('did not converge')
    return kn

def wavenumber(f, depth, shallow_depth=20, g=9.81, wtype='intermediate'):
	""" input frequency (Hz) and depth (m)
		converted from Jim Thompsons wavelength.m
	"""
	omega = 2*np.pi*f 
	depthR = np.round(depth, decimals=1)
	if wtype=='intermediate':
		if depth<shallow_depth: #shallow
			guess_k = np.sqrt(omega**2/(g*depthR))
			eps = 0.01*guess_k 
			err = np.abs(omega**2 - g*guess_k*np.tanh(guess_k*depthR))
		else:
			guess_k = omega**2/g
			eps = 0.01*guess_k
			err = np.abs(omega**2 - g*guess_k*np.tanh(guess_k*depthR))
		k = guess_k
		while err>eps:
			k = guess_k - (omega**2 - g*guess_k*np.tanh(guess_k*depthR))/(-g*np.tanh(guess_k*depthR) - g*guess_k*depthR*np.cosh(guess_k)**2)
			err = np.abs(omega**2 - g*k*np.tanh(k*depthR))
			guess_k = k
	if wtype=='deep':
		k = omega**2/g 
	if wtype=='shallow':
		k = omega/np.sqrt(g*depth)
	return k

def wavelength(k):
	return 2*np.pi/k

def displacement(amplitude, k, omega, depth, g=9.81):
	d = amplitude*g*k/omega**2
	return d, d*np.tanh(k*depth)

def pressure(amplitude, k, depth, rho=1000, g=9.81):
	return rho*g*amplitude*(1/np.cosh(k*depth))

def energy_density(amplitude, rho=1000, g=9.81):
	return 1/16*rho*g*(2*amplitude)**2

def group_speed(L, period, depth):
	k = 2*np.pi/L
	return L/period*(0.5+k*depth/np.sinh(2*k*depth))

def energy_flux(energy_density, cg):
	return energy_density*cg

def ursel(L, amplitude, depth):
	return L**2*2*amplitude/depth**3

def phase_speed(omega, k, d, g=9.81):
	return g/omega*np.tanh(k*d)

def stokes_drift(amplitude, L, c, depth):
	k = 2*np.pi/L
	return (np.pi*2*amplitude/L)**2*(c/2)*(np.cosh(2*k*depth)/np.sinh(k*depth)**2)

def x_hat(X, U, g=9.81):
	return g*X/U**2

def h_hat(H, U, g=9.81):
	return g*H/U**2

def t_hat(period, U, g=9.81):
	return g*period/U

def duration_limited_time(X, U, g=9.81):
	return 77.23*X**0.67/(U**0.34*g**0.33)

def fetch_limited_h_hat(x_hat):
	return 0.002*x_hat**0.5

def fetch_limited_t_hat(x_hat):
	return 0.25*x_hat**(1/3)

def spectral_Hs(spec, freq):
	return 4*np.sqrt(np.sum(spec*(freq[1]-freq[0])))

def spectral_Tp(spec, freq):
	num = np.sum(spec*(freq[1]-freq[0]))
	dnom = np.sum(np.dot(spec*(freq[1]-freq[0]), freq))
	return num/dnom 

def surf_similarity(H0, L0, slope):
	return slope*(H0/L0)**-0.5

def breaker_type(zeta0):
	if zeta0<0.5:
		btype = 'spilling'
	elif zeta0>3.3:
		btype = 'collapsing & surging'
	else:
		btype = 'plunging'
	return btype

def Ks(cg0, cg):
	return np.sqrt(cg0/cg)

def Kr(alpha0, alpha):
	return np.sqrt(np.cos(alpha0)/np.cos(alpha))

def H_shoaling(Ks, Kr, H0):
	return H0*Ks*Kr 

def make_jonswap_spectrum(Hsig, Tp, freq=np.arange(0.01,1.01,0.01), gamma=3.3):
	"""
	Create Jonswap spectrum code modified from Melissa Moulton:
	 function [Sf]=create_jonswap_spectrum(Hsig,Tp,gamma);
	  Usage: This function creates a Jonswap Spectrum on the basis of given
	  significant wave height (Hsig), peak wave period (Tp) and peakedness parameter (gamma).
	  Another possibility is to use the WAFO Toolbox (http://www.maths.lth.se/matstat/wafo/) which 
	  provides a more accurate implementation of JONSWAP spectrum:
	
	 Nirnimesh Kumar and George Voulgaris
	 Coastal Processes and Sediment Dynamics Lab
	 Dept. of Earth and Ocean Sciences,
	 Univ. of South Carolina, Columbia, SC
	 03/06/2013
	"""
	g     = 9.81            # Acceleration due to gravity       
	omegap = 2*np.pi/Tp        # Peak angular frequency
	omega = 2*np.pi*freq       # Angular frequency
	domega = np.diff(omega) 
	domega = np.asarray([domega[0] for i in range(len(omega))])

	sigma = np.zeros(len(freq))
	sigma[np.where(omega>omegap)[0]] = 0.09
	sigma[np.where(omega<=omegap)[0]] = 0.07

	a = np.exp(-((omega-omegap)**2)/(2*(omegap**2)*(sigma**2)))
	beta = 5/4 
	Sw = (1/(omega**5))*(np.exp(-beta*(omegap**4)/(omega**4)))*(gamma**a)
	NormFac = ((Hsig/g)**2)/16/np.sum(Sw*domega)

	Sw = Sw*NormFac*g**2 
	Sf = 2*np.pi*Sw     # Units are m^2/Hz	
	return Sf
