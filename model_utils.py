import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import numpy.ma as ma
from scipy.signal import welch, csd
from funpy import filter_functions as ff 
import cv2

def load_masked_variable(var, varfile, mask, maskfile):
    mask_ds = xr.open_mfdataset(maskfile, combine='nested', concat_dim='time')
    var_unmasked = xr.open_mfdataset(varfile, combine='nested', concat_dim='time')[var]
    x = np.asarray(xr.open_mfdataset(varfile, combine='nested', concat_dim='time')['x'])
    y = np.asarray(xr.open_mfdataset(varfile, combine='nested', concat_dim='time')['y'])
    mask = mask_ds[mask]
    var_masked = ma.masked_where(mask==0, var_unmasked)
    return var_masked, x, y 

def model2lab(x):
    return x - 22

def compute_Hsig(eta, start, end):
    return 4*np.nanstd(eta[start:end, :, :], axis=0)

def compute_spec(var, dt, WL=128, OL=64, n=1, axis=0):
    freq, spec = welch(var, fs=1/dt, window='hann', nperseg=WL*n, noverlap=OL*n, axis=axis)
    return freq, spec

def compute_Hsig_spectrally(freq, spec, fmin, fmax):
    ind = np.where((freq>fmin)&(freq<fmax))[0]
    Hs = 4*np.sqrt(np.nansum(spec[ind], axis=0)*np.diff(freq)[0])
    return Hs

def load_var_lab(fdir, var, filepath, maskvar, maskfile):
    var, x, y = load_masked_variable(var, os.path.join(fdir,filepath), maskvar, os.path.join(fdir,maskfile))
    x = model2lab(x)
    x_ind = np.where(x>0)[0]
    return x[x_ind], y, var[:,:,x_ind]

def binize_var(var, threshold):
    var[var>threshold] = 1
    var[var<=threshold] = 0
    return var 

def spatially_avg(var, x, y, order=1, filtx=0.5, filty=0.5):
    window = ff.lanczos_2Dwindow(y, x, order, 0.5, 0.5)
    var_bar = ff.lanczos_2D(var.data, var.mask, window, len(y), len(x))
    return var_bar

def find_crests(var, x, y, threshold=0, connectivity=8, order=1, filtx=0.5, filty=0.5):
    var_bar = spatially_avg(var, x, y, order=order, filtx=filtx, filty=filty)
    var_bin = binize_var(var_bar, threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(var_bin, connectivity=connectivity)
    return var_bar, var_bin, num_labels, labels

def calc_crestlen_fbr(x, y, num_labels, labels, fbr):
    [xx, yy] = np.meshgrid(x, y)

    crestend_max_x = np.zeros(num_labels)
    crestend_min_x = np.zeros(num_labels)
    crestend_max_y = np.zeros(num_labels)
    crestend_min_y = np.zeros(num_labels)
    crestlen = np.zeros(num_labels)
    crest_fbr_std = np.zeros(num_labels)
    crest_fbr_abs = np.zeros(num_labels)
    crest_fbr_sq = np.zeros(num_labels)
    crest_fbr_mean = np.zeros(num_labels)
    
    for i in range(num_labels):
        ind_x = np.where(labels==i+1)[0]
        ind_y = np.where(labels==i+1)[1]
        if len(ind_x)>0:
            crest_x = xx[ind_x, ind_y]
            crest_y = yy[ind_x, ind_y]
            crestend_max_y[i] = np.max(crest_y)
            crestend_min_y[i] = np.min(crest_y)
            crestend_max_x[i] = crest_x[np.argmax(crest_y)]
            crestend_min_x[i] = crest_x[np.argmin(crest_y)]
            
            crest_fbr_std[i] = np.std(fbr[ind_x, ind_y])
            crest_fbr_abs[i] = np.sum(np.abs(fbr[ind_x, ind_y]))
            crest_fbr_sq[i] = np.sum(fbr[ind_x, ind_y]**2)
            crest_fbr_mean[i] = np.mean(np.abs(fbr[ind_x, ind_y]))
            crest_y_unique = np.unique(crest_y)
            crest_x_avg = np.zeros(len(crest_y_unique))
            crestlen_tmp = 0
            for j in range(len(crest_y_unique)):
                ind = np.where(crest_y == crest_y_unique[j])[0]
                crest_x_avg[j] = np.mean(crest_x[ind])
                if j>0:
                    crestlen_tmp += np.sqrt((crest_x_avg[j]-crest_x_avg[j-1])**2 + (crest_y_unique[j]-crest_y_unique[j-1])**2)

            crestlen[i] = crestlen_tmp

    alonglen = crestend_max_y - crestend_min_y  
    return crestend_min_x, crestend_max_x, crestend_min_y, crestend_max_y, alonglen, crestlen, crest_fbr_std, crest_fbr_abs, crest_fbr_sq, crest_fbr_mean

def tridiag(alpha, beta, gamma, b):
    # Solve the tridiagonal system Ax=b, alpha is below 
    # the diagonal, beta is the diagonal, and gamma is 
    # above the diagonal

    N = len(b[0,:])

    # perform forward elimination 
    for i in range(1,N):
        coeff = alpha[i-1]/beta[i-1]
        beta[i] = beta[i] - coeff*gamma[i-1]
        b[:,i] = b[:,i] - coeff*b[:,i-1]

    # perform back substitution
    x2 = np.zeros(b.shape, dtype=complex)
    x2[:,N-1] = b[:,N-1]/beta[N-1]
    for i in range(N-2,-1,-1):
        x2[:,i] = (b[:,i] - np.expand_dims(gamma[i], axis=0)*x2[:,i+1])/beta[i]

    return x2

def vel_decomposition(u, v, dx, dy):
    """ Velocity decomposition function that returns the 
        velocity stream function (psi) and velocity 
        potential (phi) given a velocity field. With zero
        velocity on the x boundary (assumption). The 
        equation being solved here is of the form:
        ui^ + vj^ = div(phi) + curl(psi)

        This function is rewritten from Dr. Matthew Spydell's 
        matlab function in the funwaveC toolbox 
    """
    ### set up dimenional values
    [ny, nx] = u[0,:,:].shape
    Ly = ny*dy
    Lx = nx*dx

    ### take spatial derivatives 
    ux = np.gradient(u, dx, axis=2)
    uy = np.gradient(u, dy, axis=1)
    vx = np.gradient(v, dx, axis=2)
    vy = np.gradient(v, dy, axis=1)
    vbar = np.mean(np.mean(v, axis=-1), axis=-1)
    ubar = np.mean(np.mean(u, axis=-1), axis=-1)
    psi_at_lx = vbar*Lx 
    phi_at_lx = ubar*Lx 
    del u, v 

    divu = ux + vy ## forcing for the velocity potential 
    divu[:,:,-1] = divu[:,:,-1] - np.expand_dims(phi_at_lx, axis=-1)/dx**2
    curlu = vx - uy ## forcing for the streamfunction 
    curlu[:,:,-1] = curlu[:,:,-1] - np.expand_dims(psi_at_lx, axis=-1)/dx**2
    del ux, uy, vx, vy 

    ### solve Laplaces's equation using fft in y
    un_diag = 1/dx**2 
    alpha = un_diag*np.ones(nx)
    on_diag = -2/dx**2 
    beta = on_diag*np.ones(nx)
    beta_noflux = beta.copy() 
    beta_noflux[-1] = 2/dx**2
    ov_diag = 1/dx**2
    gamma = alpha.copy() 
    gamma_noflux = gamma.copy()
    gamma_noflux[0] = 2/dx**2 

    Gpsi = np.fft.fftshift(np.fft.fftn(curlu, axes=[1]), axes=[1])
    Gphi = np.fft.fftshift(np.fft.fftn(divu, axes=[1]), axes=[1])

    del curlu, divu

    kpos = np.arange(0, ny/2)
    kneg = np.arange(-ny/2, 0)
    N = np.fft.fftshift(np.append(kpos, kneg))

    Xpsi = np.zeros(Gpsi.shape, dtype=complex)
    Xphi = np.zeros(Gphi.shape, dtype=complex)

    for a in range(ny):
        c = -4*np.pi**2*N[a]**2/Ly**2
        g = Gpsi[:,a,:]
        Xpsi[:,a,:] = tridiag(alpha, beta+c, gamma, g)
        g = Gphi[:,a,:]
        Xphi[:,a,:] = tridiag(alpha, beta+c, gamma_noflux, g)

    psi0 = np.fft.ifft(np.fft.ifftshift(Xpsi, axes=[1]), axis=1)
    u_psi = -np.gradient(psi0.real, dy, axis=1)
    u_psi = np.append(np.expand_dims(u_psi[:,-1,:], axis=1), u_psi[:,:-1,:], axis=1)
    psi = psi0.copy()
    psi[:,:,0] = np.zeros(psi[:,:,0].shape)
    psi[:,:,-1] = np.ones(psi[:,:,-1].shape)*np.expand_dims(psi_at_lx, axis=-1)
    v_psi = np.gradient(psi.real, dx, axis=2)
    del Xpsi

    phi = np.fft.ifft(np.fft.fftshift(Xphi, axes=[1]), axis=1)
    del Xphi
    phi[:,:,-1] = np.ones(phi[:,:,-1].shape)*np.expand_dims(phi_at_lx, axis=-1)
    u_phi = np.gradient(phi.real, dx, axis=2)
    v_phi = np.gradient(phi.real, dy, axis=1)
    return u_psi, v_psi, u_phi, v_phi 

def calculate_dirspread(eta, u, v, dt, lf, WL, OL):
    # Estimation of Power and Cross Spectrum using FFT Analysis
    f1, Cuu = csd(u, u, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
    f1, Cvv = csd(v, v, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
    f1, Cpp = csd(eta, eta, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
    f1, Cpu = csd(eta, u, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
    f1, Cpv = csd(eta, v, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)
    f1, Cuv = csd(u, v, fs=1/dt, window='hann', nperseg=WL, noverlap=OL, axis=0)

    # Conversion from Pressure/Velocty Spectrum to Wave Height Spectrum
    ind = np.where(f1>lf)[0] # low freqeuncy cutoff
    lF = len(ind) # number of frequencies we keep

    # Remove values corresponding to frequencies lower than Lf
    f1 = f1[ind]
    Cuu = Cuu[ind,:]
    Cvv = Cvv[ind,:]
    Cpp = Cpp[ind,:]
    Cpu = Cpu[ind,:]
    Cpv = Cpv[ind,:]
    Cuv = Cuv[ind,:]

    # Spectra calculation
    Su = Cuu+Cvv 
    Sp = Cpp 

    # Calculation of fourier coefficients from Herbers et al, 1999 JGR
    A1 = Cpu.real/((Cpp.real*(Cuu.real+Cvv.real))**0.5)
    B1 = Cpv.real/((Cpp.real*(Cuu.real+Cvv.real))**0.5)
    A2 = (Cuu.real-Cvv.real)/(Cuu.real+Cvv.real)
    B2 = 2*Cuv.real/(Cuu.real+Cvv.real)
    Theta = 0.5*np.arctan2(B2,A2)*180/np.pi 
    Dirspread = (0.5*(1-(A2*np.cos(2*Theta*np.pi/180)+B2*np.sin(2*Theta*np.pi/180))))**0.5*180/np.pi
    Dir = np.arctan2(B1,A1)*180/np.pi

    # Weighted averaging 
    Cpu_avg = weighted_average(Cpu.real, Sp, f1)
    Cpp_avg = weighted_average(Cpp.real, Sp, f1)
    Cuu_avg = weighted_average(Cuu.real, Sp, f1)
    Cvv_avg = weighted_average(Cvv.real, Sp, f1)
    Cuv_avg = weighted_average(Cuv.real, Sp, f1)
    Cpv_avg = weighted_average(Cpv.real, Sp, f1)

    # Bulk fourier coefficients
    A1_avg = Cpu_avg/((Cpp_avg*(Cuu_avg+Cvv_avg))**0.5)
    B1_avg = Cpv_avg/((Cpp_avg*(Cuu_avg+Cvv_avg))**0.5)
    A2_avg = (Cuu_avg-Cvv_avg)/(Cuu_avg+Cvv_avg)
    B2_avg = 2*Cuv_avg/(Cuu_avg+Cvv_avg)
    Theta_avg = 0.5*np.arctan2(B2_avg,A2_avg)*180/np.pi 

    # A2, B2 estimate
    arg1 = 2*Theta_avg*np.pi/180
    arg2 = 2*Theta_avg*np.pi/180    
    Dirspread_avg = (0.5*(1-(A2_avg*np.cos(arg1)+B2_avg*np.sin(arg2))))**0.5*180/np.pi

    # A1, B1 estimate
    #arg1 = Theta_avg*np.pi/180
    #arg2 = Theta_avg*np.pi/180 
    #Dirspread_avg = (2*(1-(A1_avg*np.cos(arg1)+B1_avg*np.sin(arg2))))**0.5*180/np.pi

    Dir_avg = np.arctan2(B1_avg,A1_avg)*180/np.pi

    return Dirspread_avg, Theta_avg

def weighted_average(var, S, f, fmin=0.33, fmax=2.5):
    ind = np.where((f>fmin)&(f<fmax))[0]
    df = f[1] - f[0]
    num = np.sum(var[ind,:]*S[ind,:], axis=0)*df
    den = np.sum(S[ind,:], axis=0)*df 
    return num/den

def curl(u, v, dx, dy, time_axis=True):
    if time_axis==True:
        dvdx_dudy = np.gradient(v, dx, axis=2) - np.gradient(u, dy, axis=1)
    if time_axis==False:
        dvdx_dudy = np.gradient(v, dx, axis=1) - np.gradient(u, dy, axis=0)
    return dvdx_dudy
