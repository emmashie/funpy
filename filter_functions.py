#!/usr/bin/env python
# coding: utf-8


import scipy.signal as signal
import numpy as np


### Filtering functions written by Seth Travis ###


def lanczos_1Dwindow(datatime,order,filt_time):
    
    import numpy as np
    
    # Define the Lanczos filter window for the longitudinal direction
    cutoff = filt_time
    step_size = abs(datatime[1]-datatime[0])
    windowwidth = int((cutoff/step_size)*order)
    if windowwidth < 3:
        print('   *** Note: window width found to be fewer than 3 points. Setting window width to default minimum of 3 points.')
        windowwidth = 3
    fc = 1/(cutoff/step_size)

    windowwidth = int(windowwidth)
    if windowwidth%2 == 0:
        windowwidth = windowwidth + 1
    halfwidth = int((windowwidth-1)/2)

    k = np.arange(1., halfwidth)
    window = np.zeros(windowwidth)
    window[halfwidth] = 2*fc
    sigma = np.sin(np.pi * k / halfwidth) * halfwidth / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * fc * k) / (np.pi * k)
    window[halfwidth-1:0:-1] = sigma * firstfactor
    window[halfwidth+1:-1] = sigma * firstfactor 
    windowwidth = windowwidth - 2
    halfwidth = halfwidth - 1
    window = window / np.sum(window)
    
    return window


# In[ ]:


def lanczos_2Dwindow(x,y,order,filt_x,filt_y):
    
    import numpy as np
    
    # Define the Lanczos filter window for the longitudinal direction
    xwindow = lanczos_1Dwindow(x,order,filt_x)
    
    # Define the Lanczos filter window for the latitudinal direction
    ywindow = lanczos_1Dwindow(y,order,filt_y)
    
    window = np.expand_dims(ywindow,axis=1) * np.expand_dims(xwindow,axis=0)    
    window = window / np.sum(window)
    
    return window


# In[ ]:


def lanczos_3Dwindow(x,y,t,order,filt_x,filt_y,filt_t):
    
    import numpy as np
    
    # Define the Lanczos filter window for the longitudinal direction
    xwindow = lanczos_1Dwindow(x,order,filt_x)
        
    # Define the Lanczos filter window for the latitudinal direction
    ywindow = lanczos_1Dwindow(y,order,filt_y)
    
    # Define the Lanczos filter window for the temporal direction
    twindow = lanczos_1Dwindow(t,order,filt_t)
    
    # Use the three directional windows to create a 3D window
    window = (np.expand_dims(np.expand_dims(twindow,axis=1),axis=2) *
              np.expand_dims(np.expand_dims(ywindow,axis=0),axis=2) *
              np.expand_dims(np.expand_dims(xwindow,axis=0),axis=1))    
    window = window / np.sum(window)
    
    return window


# In[ ]:


def lanczos_1D(rawdata,window,*args,**kwargs):
        
    import numpy as np
    
    if ('datawrap' in kwargs):
        if kwargs['datawrap']:
            datawrap = True
        else:
            datawrap = False
    else:
        datawrap = False
    
    ##
    if np.ma.is_masked(rawdata):
        curmask = rawdata.mask
        rawdata = rawdata.data
        rawdata[curmask] = np.nan
    smoothdata = np.copy(rawdata)
    
    weight_max = window.max()
    windowwidth = int(len(window))
    halfwidth = int(windowwidth/2)

    NT = len(rawdata)
    
    if datawrap:
        maxrange = np.concatenate((np.arange(NT-halfwidth,NT).astype(int),
                                   np.arange(0,NT).astype(int),
                                   np.arange(0,halfwidth).astype(int)),axis=0)
    
    if not(datawrap):
        for tind in range(0,halfwidth):
            databox = rawdata[0:tind+halfwidth+1]
            datawindow = np.copy(window[halfwidth-tind:])
            
            goodinds = np.where(np.invert(np.isnan(databox)))[0]            
            databox = databox[goodinds]
            datawindow = datawindow[goodinds]
            
            weight = np.sum(datawindow)
            weight_len = len(goodinds)
            
            if (((np.isfinite(rawdata[tind])) or 
                ((weight > 0.2) and (weight_len > 2)) or
                ((weight > 0.1) and (weight_len < 6))) and
                (weight >= weight_max)):
                smoothdata[tind] = (np.sum(databox*datawindow)/
                                 weight) 
            else:
                smoothdata[tind] = np.nan
                
        for tind in range(halfwidth,NT-halfwidth):
            databox = rawdata[tind-halfwidth:tind+halfwidth+1]
            datawindow = np.copy(window)
            
            goodinds = np.where(np.invert(np.isnan(databox)))[0]            
            databox = databox[goodinds]
            datawindow = datawindow[goodinds]
            
            weight = np.sum(datawindow)
            weight_len = len(goodinds)
            
            if (((np.isfinite(rawdata[tind])) or 
                ((weight > 0.2) and (weight_len > 2)) or
                ((weight > 0.1) and (weight_len < 6))) and
                (weight >= weight_max)):
                smoothdata[tind] = (np.sum(databox*datawindow)/
                                 weight) 
            else:
                smoothdata[tind] = np.nan
        for tind in range(NT-halfwidth,NT):
            databox = rawdata[tind-halfwidth:]
            datawindow = np.copy(window[0:halfwidth+(NT-tind)])
            
            goodinds = np.where(np.invert(np.isnan(databox)))[0]            
            databox = databox[goodinds]
            datawindow = datawindow[goodinds]
            
            weight = np.sum(datawindow)
            weight_len = len(goodinds)
            
            if (((np.isfinite(rawdata[tind])) or 
                ((weight > 0.2) and (weight_len > 2)) or
                ((weight > 0.1) and (weight_len < 6))) and
                (weight >= weight_max)):
                smoothdata[tind] = (np.sum(databox*datawindow)/
                                 weight) 
            else:
                smoothdata[tind] = np.nan
    else:
        for tind in range(0,halfwidth):
            inds = maxrange[tind:tind+windowwidth]
            databox = rawdata[inds]
            datawindow = np.copy(window)
            
            goodinds = np.where(np.invert(np.isnan(databox)))[0]            
            databox = databox[goodinds]
            datawindow = datawindow[goodinds]
            
            weight = np.sum(datawindow)
            weight_len = len(goodinds)
            
            if (((np.isfinite(rawdata[tind])) or 
                ((weight > 0.2) and (weight_len > 2)) or
                ((weight > 0.1) and (weight_len < 6))) and
                (weight >= weight_max)):
                smoothdata[tind] = (np.sum(databox*datawindow)/
                                 weight) 
            else:
                smoothdata[tind] = np.nan

    smoothdata = np.ma.masked_where(np.isnan(smoothdata),smoothdata)
        
    return smoothdata


# In[ ]:


def lanczos_2D(rawdata,land,window,NY,NX,*args,**kwargs):
        
    import numpy as np
    
    if ('lonwrap' in kwargs):
        if kwargs['lonwrap']:
            lonwrap = True
        else:
            lonwrap = False
    else:
        lonwrap = False
    
    ##
    if np.ma.is_masked(rawdata):
        curmask = rawdata.mask
        rawdata = rawdata.data
        rawdata[curmask] = np.nan
    smoothdata = np.copy(rawdata) 
    
    smoothdata = np.ma.masked_where(land,smoothdata)
    [ywindowwidth,xwindowwidth] = np.shape(window)
    
    weight_max = window.max()
    yhalfwidth = int(ywindowwidth/2)
    xhalfwidth = int(xwindowwidth/2) 
    
    if lonwrap:
        xmaxrange = np.concatenate((np.arange(NX-xhalfwidth,NX).astype(int),
                                    np.arange(0,NX).astype(int),
                                    np.arange(0,xhalfwidth).astype(int)),axis=0)
    
    if not(lonwrap):        
        for iind in range(0,xhalfwidth):
            for jind in range(0,yhalfwidth):
                databox = rawdata[0:jind+yhalfwidth+1,0:iind+xhalfwidth+1]
                datawindow = np.copy(window[yhalfwidth-jind:,xhalfwidth-iind:])
                datawindow[np.isnan(databox)] = np.nan
                smoothdata[jind,iind] = np.nansum(databox*datawindow)/np.nansum(datawindow)

            for jind in range(yhalfwidth,NY-yhalfwidth):
                databox = rawdata[jind-yhalfwidth:jind+yhalfwidth+1,0:iind+xhalfwidth+1]
                datawindow = np.copy(window[:,xhalfwidth-iind:])
                datawindow[np.isnan(databox)] = np.nan
                smoothdata[jind,iind] = np.nansum(databox*datawindow)/np.nansum(datawindow)

            for jind in range(NY-yhalfwidth,NY):
                databox = rawdata[jind-yhalfwidth:,0:iind+xhalfwidth+1]
                datawindow = np.copy(window[0:yhalfwidth+(NY-jind),xhalfwidth-iind:])
                datawindow[np.isnan(databox)] = np.nan
                smoothdata[jind,iind] = np.nansum(databox*datawindow)/np.nansum(datawindow)
        
        
        for iind in range(xhalfwidth,NX-xhalfwidth):
            for jind in range(0,yhalfwidth):
                databox = rawdata[0:jind+yhalfwidth+1,iind-xhalfwidth:iind+xhalfwidth+1]
                datawindow = np.copy(window[yhalfwidth-jind:,:])
                datawindow[np.isnan(databox)] = np.nan
                smoothdata[jind,iind] = np.nansum(databox*datawindow)/np.nansum(datawindow)

            for jind in range(yhalfwidth,NY-yhalfwidth):
                databox = rawdata[jind-yhalfwidth:jind+yhalfwidth+1,iind-xhalfwidth:iind+xhalfwidth+1]
                datawindow = np.copy(window)
                datawindow[np.isnan(databox)] = np.nan
                smoothdata[jind,iind] = np.nansum(databox*datawindow)/np.nansum(datawindow)

            for jind in range(NY-yhalfwidth,NY):
                databox = rawdata[jind-yhalfwidth:,iind-xhalfwidth:iind+xhalfwidth+1]
                datawindow = np.copy(window[0:yhalfwidth+(NY-jind),:])
                datawindow[np.isnan(databox)] = np.nan
                smoothdata[jind,iind] = np.nansum(databox*datawindow)/np.nansum(datawindow)
                
        
        for iind in range(NX-xhalfwidth,NX):
            for jind in range(0,yhalfwidth):
                databox = rawdata[0:jind+yhalfwidth+1,iind-xhalfwidth:]
                datawindow = np.copy(window[yhalfwidth-jind:,0:xhalfwidth+(NX-iind)])
                datawindow[np.isnan(databox)] = np.nan
                smoothdata[jind,iind] = np.nansum(databox*datawindow)/np.nansum(datawindow)

            for jind in range(yhalfwidth,NY-yhalfwidth):
                databox = rawdata[jind-yhalfwidth:jind+yhalfwidth+1,iind-xhalfwidth:]
                datawindow = np.copy(window[:,0:xhalfwidth+(NX-iind)])
                datawindow[np.isnan(databox)] = np.nan
                smoothdata[jind,iind] = np.nansum(databox*datawindow)/np.nansum(datawindow)

            for jind in range(NY-yhalfwidth,NY):
                databox = rawdata[jind-yhalfwidth:,iind-xhalfwidth:]
                datawindow = np.copy(window[0:yhalfwidth+(NY-jind),0:xhalfwidth+(NX-iind)])
                datawindow[np.isnan(databox)] = np.nan
                smoothdata[jind,iind] = np.nansum(databox*datawindow)/np.nansum(datawindow)
        
        
    else:    
        for iind in range(0,NX):
            xinds = xmaxrange[iind:iind+xwindowwidth]
            for jind in range(0,yhalfwidth):
                databox = rawdata[0:jind+yhalfwidth+1,xinds]
                datawindow = np.copy(window[yhalfwidth-jind:,:])
                datawindow[np.isnan(databox)] = np.nan
                smoothdata[jind,iind] = np.nansum(databox*datawindow)/np.nansum(datawindow)

            for jind in range(yhalfwidth,NY-yhalfwidth):
                databox = rawdata[jind-yhalfwidth:jind+yhalfwidth+1,xinds]
                datawindow = np.copy(window)
                datawindow[np.isnan(databox)] = np.nan
                smoothdata[jind,iind] = np.nansum(databox*datawindow)/np.nansum(datawindow)

            for jind in range(NY-yhalfwidth,NY):
                databox = rawdata[jind-yhalfwidth:,xinds]
                datawindow = np.copy(window[0:yhalfwidth+(NY-jind),:])
                datawindow[np.isnan(databox)] = np.nan
                smoothdata[jind,iind] = np.nansum(databox*datawindow)/np.nansum(datawindow)

    
    smoothdata[land] = np.nan
    smoothdata = np.ma.masked_where(land,smoothdata)
    
    return smoothdata


# In[ ]:


def lanczos_3D(rawdata,land,window,NT,NY,NX):
    
    import numpy as np
    
    ##
    if np.ma.is_masked(rawdata):
        curmask = rawdata.mask
        rawdata = rawdata.data
        rawdata[curmask] = np.nan
    smoothdata = np.copy(rawdata) 
    if land.any():
        for t in range(0,NT):
            smoothdata[t,:,:] = np.ma.masked_where(land,smoothdata[t,:,:])
    [twindowwidth,ywindowwidth,xwindowwidth] = np.shape(window)
    
    land = np.ma.masked_where(land == 1,land)
    if not(np.ma.is_masked(land)):
        landtemp = np.ma.masked_where(land == 0,land) 
        landmask = ~np.ma.getmask(landtemp)
        del landtemp
    else:
        landmask = np.ma.getmask(land) 
    
    for j in range(0,NY):
        jind = np.argmin(abs(lat-lat[j]))
        
        if jind < yhalfwidth:
            ydatainds = np.arange(0,jind+yhalfwidth+1,1)
            ywindowinds = np.arange(yhalfwidth-jind,ywindowwidth,1)
        elif jind > NY-yhalfwidth-1:
            ydatainds = np.arange(jind-yhalfwidth,NY,1)
            ywindowinds = np.arange(0,yhalfwidth+(NY-jind),1)
            jind = len(ydatainds) + jind - NY
        else:
            ydatainds = np.arange(jind-yhalfwidth,jind+yhalfwidth+1,1)
            ywindowinds = np.arange(0,ywindowwidth)
            jind = yhalfwidth 

        ## Access all the data
        if np.ma.is_masked(rawdata):
            rawdata = rawdata.data
        for t in range(0,NT):
            datatemp = rawdata[t,:,:]
            datatemp[landmask] = np.nan
            rawdata[t,:,:] = datatemp
        del datatemp



        xmaxrange = np.concatenate((np.arange(NX-xhalfwidth,NX),
                                    np.arange(0,NX),
                                    np.arange(0,xhalfwidth)),axis=0)
        xwindowinds = np.arange(0,xwindowwidth)
        NXdata = len(xwindowinds)
        NYdata = len(ywindowinds)

        ydatainds = np.arange(0,len(lat))  
        ydatarepeat = np.repeat(ydatainds,NXdata)
        ywindowrepeat = np.repeat(ywindowinds,NXdata)

        landrange = np.where(np.invert(landmask[jind,:]))[0]

        for i in landrange:
            iind = np.argmin(abs(lon[i]-lon))
            xdatainds = xmaxrange[iind:iind+xwindowwidth]

            for t in range(0,NT): 
                if t < thalfwidth:
                    tdatainds = np.arange(0,t+thalfwidth+1,1)
                    twindowinds = np.arange(thalfwidth-t,twindowwidth,1)
                elif t > NT-thalfwidth-1:
                    tdatainds = np.arange(t-thalfwidth,NT,1)
                    twindowinds = np.arange(0,thalfwidth+(NT-t),1)
                else:
                    tdatainds = np.arange(t-thalfwidth,t+thalfwidth+1,1)
                    twindowinds = np.arange(0,twindowwidth)

                windoweddata = rawdata[tdatainds[0]:tdatainds[-1]+1,
                                       :,xdatainds]
                gooddata = np.where(np.isfinite(windoweddata))[:]

                windowinds = np.zeros(np.shape(gooddata)).astype(int)
                windowinds[0,:] = twindowinds[gooddata[0][:]]
                windowinds[1,:] = ywindowinds[gooddata[1][:]]
                windowinds[2,:] = xwindowinds[gooddata[2][:]]

                
                datainds = np.zeros(np.shape(gooddata)).astype(int)
                datainds[0,:] = gooddata[0][:] + tdatainds[0]
                datainds[1,:] = gooddata[1][:] + ydatainds[0]
                datainds[2,:] = xdatainds[gooddata[2][:]]        

                weight = np.sum(xytwindow[windowinds[0,:],
                                          windowinds[1,:],
                                          windowinds[2,:]])
                weight_count = len(gooddata[0][:])
                if (np.isfinite(rawdata[t,jind,iind]) and (weight >= weight_max)):               
                    smoothdata[t,j,i] = (np.sum(rawdata[datainds[0,:],
                                                      datainds[1,:],
                                                      datainds[2,:]]*
                                              xytwindow[windowinds[0,:],
                                                        windowinds[1,:],
                                                        windowinds[2,:]])/weight)
                else:
                    weight = np.sum(xytwindow[windowinds[0,:],windowinds[1,:],windowinds[2,:]])
                    weight_count = len(gooddata[0][:])
                    if ((weight_count >= 2) and (weight >= weight_max)):
                        newdata = (np.sum(rawdata[datainds[0,:],datainds[1,:],datainds[2,:]]*
                                          xytwindow[windowinds[0,:],windowinds[1,:],windowinds[2,:]])/weight)
                        if (newdata < -4) or (newdata > -1):
                            smoothdata[t,j,i] = np.nan
                        else:
                            smoothdata[t,j,i] = newdata
                    else:
                        smoothdata[t,j,i] = np.nan

    return smoothdata


# In[ ]:


def spectra_filter(x,y,cutoff,order,filttype, axis=0):
    
    import scipy.signal as signal
    
    filtertypes = ['low','high']
    if filttype not in filtertypes:
        print('Filter type "',filttype,'"is not a valid option.')
        print('Please choose from the following and try again:',filtertypes)
        return
    
    dx = (x[1:]-x[:-1]).mean()
    fs = 1/dx
    Nyq = fs/2
    W = 1/cutoff
    
    ## Design the Butterworth filter
    N  = order # Filter order
    Wn = W/Nyq # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')

    ## Apply the filter, obtaining a low-pass filter    
    yfilt = signal.filtfilt(B, A, y, axis=axis)
    if filttype == 'high':
        yfilt = y - yfilt    
    
    return yfilt


# In[ ]:


def spectra_calc(x, y, m):

    import scipy.signal as signal
    import numpy as np
    
    dx = (x[1:] - x[:-1]).mean()
    X = x[-1] - x[0]
    NX = len(x)
    if NX%2 == 0:
        N2 = int(np.floor(NX/2))
    else:
        N2 = int(np.floor((NX-1)/2))    

    fs = 1 / X
    freqs = fs * np.arange(1,N2)
    freqs = np.fft.fftfreq(NX,d=dx)
    
    fftdata1 = np.fft.fftshift(np.fft.fft(y))
    spectra_all = fftdata1*np.conj(fftdata1)/(NX*dx)
    spectra_raw = 2*spectra_all[-N2+1:]
    freqs = freqs[:N2-1]
    

    spectra = np.zeros(len(spectra_raw))
    if m > 0:
        spectra[m:-m+1] = spectra_average(m, spectra_raw)
    else:
        spectra = spectra_raw

    spectra[spectra == 0] = np.nan
    
    return freqs, spectra


# In[ ]:


def spectra_average(m, spectra):

    import numpy as np
    
    if  m <= 1:
        spectra_ave = spectra
    else:
        w = np.ones(2*m)
        w = w / sum(w)

        spectra_ave = np.convolve(spectra, w, 'valid')
    
    return spectra_ave


# In[ ]:


def correlation_func(data1,data2):
    
    import numpy as np
    
    try:    
        data1_mean = np.nanmean(data1)
        data2_mean = np.nanmean(data2)

        data1_var = np.nanvar(data1)
        data2_var = np.nanvar(data2)

        cov = np.nanmean((data1-data1_mean) * (data2-data2_mean))
        corr = cov / np.sqrt(data1_var * data2_var)
    except:
        corr = np.nan
    
    return corr


# In[ ]:


def regression_func(data1,data2):
    
    import numpy as np
    
    try:    
        data1_mean = np.nanmean(data1)
        data2_mean = np.nanmean(data2)

        data1_var = np.nanvar(data1)

        cov = np.nanmean((data1-data1_mean) * (data2-data2_mean))
        regr = cov / np.sqrt(data1_var)
    except:
        regr = np.nan
    
    return regr


# In[ ]:


def correlation_map(landmask,data1,data2):
    
    import numpy as np
    
    [NY,NX] = landmask.shape
    corrmap = np.zeros((NY,NX))
    if landmask.any():
        corrmap = np.ma.masked_where(landmask,corrmap)
    
    [yinds,xinds] = np.where(np.invert(landmask))
    for i in range(0,len(yinds)):
        corrmap[yinds[i],xinds[i]] = correlation_func(data1[:,yinds[i],xinds[i]],
                                                      data2[:,yinds[i],xinds[i]])
    
    return corrmap     


# In[ ]:


def regression_map(landmask,data1,data2):
    
    import numpy as np
    
    [NY,NX] = landmask.shape
    regrmap = np.zeros((NY,NX))
    if landmask.any():
        regrmap = np.ma.masked_where(landmask,regrmap)
    
    [yinds,xinds] = np.where(np.invert(landmask))
    for i in range(0,len(yinds)):
        regrmap[yinds[i],xinds[i]] = regression_func(data1[:,yinds[i],xinds[i]],
                                                     data2[:,yinds[i],xinds[i]])
    
    return regrmap     


# In[ ]:


def data_2D_reconstruct(landmask, data):

    import numpy as np
    
    [yinds,xinds] = np.where(np.invert(landmask))
    [NT,Ninds] = np.shape(data)
    [NY,NX] = np.shape(landmask)
    if len(yinds) != Ninds:
        data = data.T
        [NT,Ninds] = np.shape(data)
    
    if landmask.any():
        data_xy = np.ma.zeros((NT, NY, NX))
    else:
        data_xy = np.zeros((NT, NY, NX))
        
    for i in range(0,Ninds):
        data_xy[:,yinds[i],xinds[i]] = data[:,i]
    
    if landmask.any():
        for t in range(0,NT):
            data_xy[t,:,:] = np.ma.masked_where(landmask,data_xy[t,:,:])
        
    return data_xy


# In[ ]:


def eof_func(NT,NX,data_vec):
    
    import scipy
    import scipy.signal as signal
    
    import sklearn
    from sklearn import preprocessing
    from sklearn.decomposition import pca
    from sklearn.preprocessing import StandardScaler
    
    import matplotlib 
    from matplotlib import pyplot as plt
    
    skpca = pca.PCA()
    skpca.fit(data_vec)
    
    f, ax = plt.subplots(figsize=(5,5))
    ax.plot(skpca.explained_variance_ratio_[0:10]*100)
    ax.plot(skpca.explained_variance_ratio_[0:10]*100,'ro')
    ax.set_title("% of variance explained", fontsize=14)
    ax.grid()
    print('Variance in mode: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10')
    print(np.floor(100*skpca.explained_variance_ratio_[0:10]*100)/100)
    
    cumvar = 0.95
    ipc = np.where(skpca.explained_variance_ratio_.cumsum() >= cumvar)[0][0]
    print('Number of modes to contain',100*cumvar,'% of variance:',ipc)
    vals = skpca.explained_variance_ratio_
    
    PCs = skpca.transform(data_vec)
    PCs = PCs[:,:ipc]
    
    EOFs = skpca.components_
    EOFs = EOFs[:ipc,:]
    vals = vals[:ipc]
    
    scaler_PCs = StandardScaler()
    scaler_PCs.fit(PCs)
    vecs = scaler_PCs.transform(PCs)
    
    
        
    return vals, vecs, EOFs


# In[ ]:


def eof_2D(datas, x, y, t, landmask):
    
    import numpy as np
    
    import sklearn
    from sklearn import preprocessing
    from sklearn.decomposition import pca
    from sklearn.preprocessing import StandardScaler

    scaler  = preprocessing.StandardScaler()
    
    NT = len(t)
    NY = len(y)
    NX = len(x)
    
    [yinds,xinds] = np.where(np.invert(landmask))
    Ninds = len(yinds)
    

    ## Re-arrange the 3d time-x-y matrix into a 2d time-space matrix
    data_vec = np.zeros((NT,Ninds))
    for i in range(0,Ninds):
        data_vec[:,i] = datas[:,yinds[i],xinds[i]]     
        
    data_scaler = scaler.fit(data_vec)
    data_vec = data_scaler.transform(data_vec)

    ## Calculate the EOF of the restructed data
    [vals, vecs, eof] = eof_func(NT,Ninds,data_vec)

    nmodes = len(vals)
    
    ## Re-construct the eof from a 1 space matrix back into a 2d
    ## x-y matrix for plotting
    #eof_xy = data_2D_reconstruct(landmask, eof)
    
    eof_xy = np.ones((nmodes, NY, NX)) * -999
    for i in range(nmodes): 
        eof_xy[i,yinds,xinds] = eof[i,:]
    eof_xy = np.ma.masked_where(eof_xy == -999.,eof_xy)

    return vals, vecs, eof, eof_xy


# In[ ]:


def eof_data_reconstruct(data_vec,vecs,modes):
    
    import scipy.signal as signal
    
    [NT,Ninds] = data_vec.shape
    for i in range(0,Ninds):
        data_vec[:,i] = signal.detrend(data_vec[:,i])
    
    tvecs = np.matmul(data_vec, vecs)
    recon = np.matmul(tvecs[:,modes], vecs.T[modes,:])
    
    return recon


# In[ ]:


def eof_datarecon_bymodes(rawdata, datamask, eof, modes):
    
    [yinds,xinds] = np.where(np.invert(datamask))
    datamean = np.nanmean(rawdata,axis=0)
    
    tempdata = rawdata[:,yinds,xinds] @ eof[modes,:].T
    recondata = tempdata @ eof[modes,:]
    
    recon_xy = np.ma.copy(rawdata)
    recon_xy[:,yinds,xinds] = recondata
    recon_xy = recon_xy + datamean[np.newaxis,:,:]
    
    return recon_xy


# In[ ]:


def eof_data_reconstruct_2D(datas,landmask,vecs,modes):
    
    import numpy as np
    
    [yinds,xinds] = np.where(np.invert(landmask))
    Ninds = len(yinds)
    [NT,NY,NX] = datas.shape

    ## Re-arrange the 3d time-x-y matrix into a 2d time-space matrix
    data_vec = np.zeros((NT,Ninds))
    for i in range(0,Ninds):
        data_vec[:,i] = datas[:,yinds[i],xinds[i]]
    
    recondata = eof_data_reconstruct(data_vec,vecs,modes)
    
    recond2d = data_2D_reconstruct(landmask, recondata)
    
    return recon2d


# In[ ]:


def integrator(zmin,zmax,z,fraw):
    
    import numpy as np
    import gc
    
    f = np.ma.copy(fraw)
    f = np.ma.masked_where(np.isnan(f),f)
    fshape = np.asarray(f.shape)
    ndims = len(fshape)
    if ndims == 1:
        f = np.ma.expand_dims(f,axis=1)
        fshape = np.asarray(f.shape)
        ndims = len(fshape)
        
    NZ = len(z)
    Haxis = np.where(fshape == NZ)[0][0].astype(int)
    noHaxis = np.arange(0,ndims).astype(int)[np.invert(np.arange(0,ndims).astype(int) == Haxis)].astype(int)
    
    zexpand = np.copy(z)
    if ndims > 1:
        for n in range(0,ndims-1):
            expanddim = noHaxis[n]
            zexpand = np.expand_dims(zexpand,axis=noHaxis[n])
    else:
        zexpand = np.expand_dims(zexpand,axis=noHaxis)
    if np.ma.is_masked(zmin):
        zminmask = zmin.mask
        zmin = zmin.data
        zmin[zminmask] = np.nan
        
    if type(zmax) == np.ma.core.MaskedArray:
        zmaxmask = zmax.mask
        zmax = zmax.data
        zmax[zmaxmask] = np.nan
            
    if type(zmin) == np.ndarray:
        zminshape = np.asarray(zmin.shape)
    else:
        zminshape = 1

    if ((type(zmax) == np.ndarray) or (type(zmax) == np.ma.core.MaskedArray)):
        zmaxshape = np.asarray(zmax.shape)
    else:
        zmaxshape = 1
        
    if not(type(zminshape) == tuple):
        if ndims-1 == 0:
            zminmap = np.array([zmin])
        else:
            zminmap = zmin*np.ma.ones(fshape[noHaxis])
    else:
        zminmap = np.ma.copy(zmin)
    if not(type(zmaxshape) == tuple):
        if ndims-1 == 0:
            zmaxmap = np.array([zmax])
        else:
            zmaxmap = zmax*np.ma.ones(fshape[noHaxis])
    else:
        zmaxmap = np.ma.copy(zmax)

    if np.ma.is_masked(f):
        fmask = np.copy(f.mask)
        f[fmask] = np.nan
        
        depmax = np.argmax(fmask, axis=Haxis)
        allzero = np.invert(np.any(fmask,axis=Haxis))
        if allzero.any():
            if ndims > 1:
                depmax[allzero] = NZ
            else:
                depmax = NZ
        
        depmax = depmax - 1
        if ndims > 1:
            depmax[depmax < 0] = 0
        else:
            if depmax < 0:
                depmax = 0
        
        if ndims > 1:
            depmaxinds = np.where(depmax >= 0)
            znew = np.copy(zmaxmap)
            zmaxmap[depmaxinds] = z[depmax[depmaxinds]]
            zmaxmap[np.where(zmaxmap > znew)] = znew[np.where(zmaxmap > znew)]
            del znew
        else:
            zmaxmap = z[depmax]
            if zmaxmap > zmax:
                zmaxmap = zmax
                
        del depmax
        del depmaxinds
    gc.collect()

    ## Find the min and max indices
    zminind = np.argmin(abs(zexpand-np.expand_dims(zminmap,axis=Haxis)),axis=Haxis)
    zminind[np.where(z[zminind] > zminmap)] = zminind[np.where(z[zminind] > zminmap)] - 1
    zminind[zminind < 0] = 0
        
    zmaxind = np.argmin(abs(zexpand-np.expand_dims(zmaxmap,axis=Haxis)),axis=Haxis)
    zmaxind[np.where(z[zmaxind] < zmaxmap)] = zmaxind[np.where(z[zmaxind] < zmaxmap)] + 1
    zmaxind[zmaxind == NZ] = zmaxind[zmaxind == NZ] - 1
    zmaxind[zmaxind <= zminind] = zmaxind[zmaxind <= zminind] + 1
    zmaxind[zmaxind == NZ] = zmaxind[zmaxind == NZ] - 1
        
    ##
    zmininds = np.unique(zminind)
    zmaxinds = np.unique(zmaxind)
    zinds = np.zeros((len(zmininds)*len(zmaxinds),2))
    for j in range(len(zmininds)):
        for i in range(len(zmaxinds)):
            zinds[j*len(zmaxinds) + i,0] = zmininds[j]
            zinds[j*len(zmaxinds) + i,1] = zmaxinds[i]
    goodinds = np.where(zinds[:,0] < zinds[:,1])
    badinds = np.where(zinds[:,0] >= zinds[:,1])
    nanzinds = zinds[badinds[0],:].astype(int)
    zinds = zinds[goodinds[0],:].astype(int)
    [ninds,temp] = zinds.shape

    
    ##
    F = np.ma.zeros(fshape[noHaxis])
    
    for n in range(ninds):
        curzmin = zinds[n,0]
        curzmax = zinds[n,1]
        curinds = np.where(np.logical_and(zminind == curzmin,
                                          zmaxind == curzmax))

        if len(zminind[curinds]) == 0:
            continue
        
        slc = [slice(None)] * len(f.shape)
        slc[Haxis] = slice(curzmin,curzmax+1)

        frange = np.ma.copy(f[slc])
        fshape = np.asarray(frange.shape)
        zrange = np.empty(frange.shape)
        ztemp = z[curzmin:curzmax+1]
        NZ = len(ztemp)
        slc = [slice(None)] * len(frange.shape)
        zslc = [slice(None)] * len(zmaxmap.shape)
        for i in range(0,NZ):
            slc[Haxis] = slice(i,i+1)
            zslice = ztemp[i]*np.ones(frange[slc].shape)
            zrange[slc] = zslice

        slc1 = [slice(None)] * len(f.shape)
        slc0 = [slice(None)] * len(f.shape)
        
        
        ## Expand the f and z range to include z min range
        if (zminmap[curinds] < z[curzmin]).any():
            if curzmin == 0:
                dz = np.squeeze(z[curzmin]-zminmap[curinds])
                slc1[Haxis] = slice(curzmin+1,curzmin+2)
                slc0[Haxis] = slice(curzmin,curzmin+1)
                df = np.squeeze((f[slc1]-f[slc0])/(z[curzmin+1]-z[curzmin]),axis=Haxis) 
            else:
                dz = np.squeeze(z[curzmin]-zminmap[curinds],axis=Haxis)
                slc1[Haxis] = slice(curzmin,curzmin+1)
                slc0[Haxis] = slice(curzmin-1,curzmin)
                df = np.squeeze((f[slc1]-f[slc0])/(z[curzmin]-z[curzmin-1]),axis=Haxis)
            if not(bool(dz.shape)):
                dz = dz*np.ma.ones(df[curinds].shape)
            
            f0 = np.copy(np.squeeze(f[slc0],axis=Haxis))
            if ndims > 2:
                lowinds = np.where(zminmap[curinds] <= z[curzmin])[0]
                highinds = np.where(zminmap[curinds] > z[curzmin])[0]

                if not(lowinds.size == 0):
                    if not(bool(f0.shape)):
                        f0 = f0 - df*dz
                    else:
                        ftest = f0[curinds]
                        ftest[lowinds] = ftest[lowinds] - df[curinds][lowinds]*dz[lowinds]
                        f0[curinds] = ftest
                if not(highinds.size == 0):
                    if not(bool(f0.shape)):
                        f0 = f0 + df*dz
                    else:
                        ftest = f0[curinds]
                        ftest[lowinds] = ftest[lowinds] + df[curinds][lowinds]*dz[lowinds]
                        f0[curinds] = ftest
            elif (zminmap[curinds] > z[curzmin]).any():
                f0[curinds] = f0[curinds] + df[curinds]*dz
            else:
                f0[curinds] = f0[curinds] - df[curinds]*dz
            f0 = np.expand_dims(f0,axis=Haxis)
            
            frange = np.append(f0,frange,axis=Haxis)
            ztemp = np.expand_dims(zminmap,axis=Haxis)
            zrange = np.append(ztemp,zrange,axis=Haxis)
        else:
            dz = np.squeeze(zminmap[curinds] - z[curzmin])
            slc1[Haxis] = slice(curzmin+1,curzmin+2)
            slc0[Haxis] = slice(curzmin,curzmin+1)
            df = np.squeeze((f[slc1]-f[slc0])/(z[curzmin+1]-z[curzmin]),axis=Haxis)
            if not(bool(dz.shape)):
                dz = dz*np.ma.ones(df[curinds].shape)
            f0 = np.copy(np.squeeze(f[slc0],axis=Haxis))
            if not(bool(f0.shape)):
                f0 = f0 + df*dz
            else:
                f0[curinds] = f0[curinds] + df[curinds]*dz
            f0 = np.expand_dims(f0,axis=Haxis)
            slc = [slice(None)] * len(frange.shape)
            slc[Haxis] = slice(0,1)
            frange[slc] = f0
            ztemp = np.squeeze(zrange[slc])
            if not(bool(ztemp.shape)):
                ztemp = zminmap[curinds]
            else:
                ztemp[curinds] = zminmap[curinds]
            zrange[slc] = np.expand_dims(ztemp,axis=Haxis)
        
        
        ## Expand the f and z range to include the z max values
        if (z[curzmax] < zmaxmap[curinds]).any():
            if curzmax >= NZ-1:
                dz = np.squeeze(zmaxmap[curinds] - z[curzmax])
                slc1[Haxis] = slice(curzmax,curzmax+1)
                slc0[Haxis] = slice(curzmax-1,curzmax)
                df = np.squeeze((f[slc1]-f[slc0])/(z[curzmax]-z[curzmax-1]),axis=Haxis)
            else:
                dz = np.squeeze(zmaxmap[curinds] - z[curzmax],axis=Haxis)
                slc1[Haxis] = slice(curzmax+1,curzmax+2)
                slc0[Haxis] = slice(curzmax,curzmax+1)
                df = np.squeeze((f[slc1]-f[slc0])/(z[curzmax+1]-z[curzmax]),axis=Haxis)
            if not(bool(dz.shape)):
                dz = dz*np.ma.ones(df[curinds].shape)
                
            f0 = np.copy(np.squeeze(f[slc0],axis=Haxis))
            if ndims > 2:
                lowinds = np.where(zmaxmap[curinds] <= z[curzmax])[0]
                highinds = np.where(zmaxmap[curinds] > z[curzmax])[0]

                if not(lowinds.size == 0):
                    if not(bool(f0.shape)):
                        f0 = f0 - df*dz
                    else:
                        ftest = f0[curinds]
                        ftest[lowinds] = ftest[lowinds] - df[curinds][lowinds]*dz[lowinds]
                        f0[curinds] = ftest
                if not(highinds.size == 0):
                    if not(bool(f0.shape)):
                        f0 = f0 + df*dz
                    else:
                        ftest = f0[curinds]
                        ftest[lowinds] = ftest[lowinds] - df[curinds][lowinds]*dz[lowinds]
                        f0[curinds] = ftest
            else:
                f0[curinds] = f0[curinds] - df[curinds]*dz

            f0 = np.expand_dims(f0,axis=Haxis)
            frange = np.append(frange,f0,axis=Haxis)
            ztemp = np.expand_dims(zmaxmap,axis=Haxis)
            zrange = np.append(zrange,ztemp,axis=Haxis)
            
        elif(z[curzmax] > zmaxmap[curinds]).any():
            dz = z[curzmax] - zmaxmap[curinds]
            slc1[Haxis] = slice(curzmax,curzmax+1)
            slc0[Haxis] = slice(curzmax-1,curzmax)
            df = np.squeeze((f[slc1]-f[slc0])/(z[curzmax]-z[curzmax-1]),axis=Haxis)
            if not(bool(dz.shape)):
                dz = dz*np.ma.ones(df[curinds].shape)
            f0 = np.copy(np.squeeze(f[slc1],axis=Haxis))
            f0[curinds] = f0[curinds] - df[curinds]*dz
            f0 = np.expand_dims(f0,axis=Haxis)
            slc = [slice(None)] * len(frange.shape)
            curshape = frange.shape
            
            slc[Haxis] = slice(curshape[Haxis]-1,curshape[Haxis])
            frange[slc] = f0
            ztemp = np.squeeze(zrange[slc])
            if not(bool(ztemp.shape)):
                ztemp = zmaxmap[curinds]
            else:
                ztemp[curinds] = zmaxmap[curinds]
            zrange[slc] = np.expand_dims(ztemp,axis=Haxis)

            
        ## Integrate the data
        slc1 = [slice(None)] * len(frange.shape)
        slc1[Haxis] = slice(1,frange.shape[Haxis])
        slc2 = [slice(None)] * len(frange.shape)
        slc2[Haxis] = slice(0,frange.shape[Haxis]-1)
        
        if ndims > 2:
            F[curinds] = np.squeeze(np.nansum((frange[slc1]+frange[slc2])*
                                              (zrange[slc1]-zrange[slc2])/2,axis=Haxis))[curinds]
        else:
            F[curinds] = np.squeeze(np.nansum((frange[slc1]+frange[slc2])*
                                              (zrange[slc1]-zrange[slc2])/2,axis=Haxis))
    [ninds,temp] = nanzinds.shape
    for n in range(ninds):
        curzmin = nanzinds[n,0]
        curzmax = nanzinds[n,1]
        curinds = np.where(np.logical_and(zminind == curzmin,
                                          zmaxind == curzmax))
        F[curinds] = np.nan   
    
    F = F*(zmax-zmin)/(zmaxmap-zminmap)
    F = np.squeeze(F)  

    return F


