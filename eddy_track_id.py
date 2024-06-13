import numpy as np
import matplotlib.pyplot as plt 
import xarray as xr 
from scipy.ndimage import morphology 
from scipy.ndimage.filters import maximum_filter
import math

def logical_subtract(A, B):
    return A.astype(int) - B.astype(int) == 1

def find_local_max(x, y, var):
    var_abs = np.abs(var)
    neighborhood = morphology.generate_binary_structure(len(np.shape(var_abs)),2)
    local_max = maximum_filter(var_abs,footprint=neighborhood)==var_abs
    background = (var_abs==0)
    eroded_background = morphology.binary_erosion(background, 
                                                  structure=neighborhood, 
                                                  border_value=1)

    detected_maxima = logical_subtract(local_max, eroded_background)

    NX = len(x)
    NY = len(y)

    [yinds,xinds] = np.nonzero(detected_maxima)

    local_count = 0
    for n in range(0,len(yinds)):
        if var[yinds[n],xinds[n]]:
            local_count = local_count + 1
            vmax_temp = var[yinds[n],xinds[n]]
            pole_temp = var[yinds[n],xinds[n]]/var_abs[yinds[n],xinds[n]]
            x_ind_temp = int(xinds[n])
            y_ind_temp = int(yinds[n])
            if local_count == 1:
                vmax = vmax_temp
                pole = pole_temp
                x_vmax_ind = x_ind_temp
                y_vmax_ind = y_ind_temp
            else:
                vmax = np.append(vmax, vmax_temp)
                pole = np.append(pole, pole_temp)
                x_vmax_ind = np.append(x_vmax_ind, x_ind_temp)
                y_vmax_ind = np.append(y_vmax_ind, y_ind_temp) 
    return local_count, vmax, pole, x_vmax_ind, y_vmax_ind 

def local_eddy_check(NX,NY,x_ind,y_ind,eddy_track_map):
    
    if x_ind == 0:
        if y_ind == 0:
            if ((eddy_track_map[y_ind,x_ind+1] == 0) &
                (eddy_track_map[y_ind+1,x_ind] == 0)):
                eddy_go = 1
            else:
                eddy_go = 0
        elif y_ind == NY-1:
            if ((eddy_track_map[y_ind,x_ind+1] == 0) &
                (eddy_track_map[y_ind-1,x_ind] == 0)):
                eddy_go = 1
            else:
                eddy_go = 0
        else:
            if ((eddy_track_map[y_ind,x_ind+1] == 0) &
                (eddy_track_map[y_ind+1,x_ind] == 0) & (eddy_track_map[y_ind-1,x_ind] == 0)):
                eddy_go = 1
            else:
                eddy_go = 0
    elif x_ind == NX-1:
        if y_ind == 0:
            if ((eddy_track_map[y_ind,x_ind-1] == 0) &
                (eddy_track_map[y_ind+1,x_ind] == 0)):
                eddy_go = 1
            else:
                eddy_go = 0
        elif y_ind == NY-1:
            if ((eddy_track_map[y_ind,x_ind-1] == 0) &
                (eddy_track_map[y_ind-1,x_ind] == 0)):
                eddy_go = 1
            else:
                eddy_go = 0
        else:
            if ((eddy_track_map[y_ind,x_ind-1] == 0) &
                (eddy_track_map[y_ind+1,x_ind] == 0) & (eddy_track_map[y_ind-1,x_ind] == 0)):
                eddy_go = 1
            else:
                eddy_go = 0
    else:
        if y_ind == 0:
            if ((eddy_track_map[y_ind,x_ind+1] == 0) & (eddy_track_map[y_ind,x_ind-1] == 0) &
                (eddy_track_map[y_ind+1,x_ind] == 0)):
                eddy_go = 1
            else:
                eddy_go = 0
        elif y_ind == NY-1:
            if ((eddy_track_map[y_ind,x_ind+1] == 0) & (eddy_track_map[y_ind,x_ind-1] == 0) &
                (eddy_track_map[y_ind-1,x_ind] == 0)):
                eddy_go = 1
            else:
                eddy_go = 0
        else:
            if ((eddy_track_map[y_ind,x_ind+1] == 0) & (eddy_track_map[y_ind,x_ind-1] == 0) &
                (eddy_track_map[y_ind+1,x_ind] == 0) & (eddy_track_map[y_ind-1,x_ind] == 0)):
                eddy_go = 1
            else:
                eddy_go = 0
                
    return eddy_go


def eddy_grow(var, eddy_track, x_range, y_range, vmax,
              delt, nmin, nmax, dmax):
    
    # Initialize the eddy testing region
    [cent_y, cent_x] = np.where(var == vmax)
    var_test = np.zeros(np.shape(var))
    var_test[var_test == 0] = np.nan
    var_test[cent_y, cent_x] = var[cent_y, cent_x]
    [NY_test, NX_test] = np.shape(var_test)
    
    var_test_old = var_test
    eddy_map = np.zeros(np.shape(var_test))
    eddy_map[eddy_map == 0] = np.nan
    
    # Define parameters for checking the eddy range
    l0 = math.floor(vmax/delt)
    h0 = l0*delt
    l = 0
    hl = h0 - l*delt
    hl_max = vmax
    lmin = l0
    if vmax < 0.3:
        lmin = math.floor(2*0.3/delt)
    [y,x] = np.nonzero(~np.isnan(var_test))
    n = len(x)
    ncount = n
    eddy_grow = 1
    eddy_flag = np.zeros(np.shape(var_test)) 
    
    # Perform a single iteration of the eddy growing procedure
    [n,eddy_map, 
     eddy_flag_first, eddy_ended] = eddy_iteration(var, var_test, eddy_track, eddy_flag, 
                                                   x_range, y_range, hl, hl_max, 
                                                   l, lmin, nmin, nmax, dmax)
    
    var_test[np.nonzero(~np.isnan(eddy_map))] = var[np.nonzero(~np.isnan(eddy_map))]
    E = np.atleast_3d(eddy_map)
    eddy_flag = np.atleast_3d(eddy_flag_first)
    del eddy_flag_first
    #### u_step = edge_speed(var,var_test,x_range,y_range)
    
    [edge_x, edge_y, interior_x, interior_y, 
     edge_count, interior_count] = find_edges(var_test,x,y)
    
    if eddy_ended > 1:
        eddy_grow = 0
        
    if n > nmax:
        eddy_grow = 0
        
    #if ((len(interior_x) < nmin) and (l < lmin)):
    #    eddy_grow = 1
        
    
    ## Go through the eddy growing procedure, checking each criterion on
    ## each growth iteration
    while eddy_grow == 1:
        l = l+1
        hl = h0 - l*delt
        
        var_test_old = var_test
        [n,new_eddy_map, 
         new_eddy_flag, eddy_ended] = eddy_iteration(var, var_test,
                                                     eddy_track,
                                                     eddy_flag[:,:,-1], 
                                                     x_range, y_range, hl, 
                                                     hl_max, l, lmin,
                                                     nmin, nmax, dmax)
        
        E = np.append(np.atleast_3d(E),
                      np.atleast_3d(new_eddy_map),axis = 2)
        eddy_flag = np.append(eddy_flag, 
                              np.atleast_3d(new_eddy_flag), axis = 2)
        
        del var_test
        var_test = np.zeros(np.shape(var))
        var_test[var_test == 0] = np.nan
        var_test[np.nonzero(~np.isnan(new_eddy_map))] = var[np.nonzero(~np.isnan(new_eddy_map))]

        #### u_step = np.append(u_step,
        ####                   edge_speed(var,var_test,x_range,y_range))        
        
        if eddy_ended >= 1:
            eddy_grow = 0
            
    ## After the eddy growth procedure has stopped, go back and check
    ## the flags for consistency
    if eddy_ended == 4:
        [flag_yinds,flag_xinds] = np.nonzero(eddy_flag[:,:,-1])
        if len(flag_yinds) > 0:
            min_l = int(np.min(np.min(eddy_flag[flag_yinds,flag_xinds,-1])))-1
        else:
            min_l = l-1
        if min_l < 1:
            min_l = 1
        
        goback = True
        while goback:
            eddy_map = np.squeeze(E[:,:,min_l])
            del var_test
            var_test = np.zeros(np.shape(var))
            var_test[np.nonzero(var_test == 0)] = np.nan
            var_test[np.nonzero(~np.isnan(eddy_map))] = var[np.nonzero(~np.isnan(eddy_map))]

            [y,x] = np.nonzero(~np.isnan(var_test))
            [edge_x, edge_y, interior_x, interior_y, 
             edge_count, interior_count] = find_edges(var_test,x,y)
            if interior_count >= nmin:
                interior_ones = np.zeros(np.shape(var_test))
                for i in range(0,len(interior_x)):
                    interior_ones[int(interior_y[i]),
                                  int(interior_x[i])] = 1
            
                int_connected = is_contiguous(interior_ones)
                if not(int_connected) and min_l > 0:
                    min_l = min_l - 1
                elif not(int_connected):
                    eddy_map[:,:] = np.nan
                    goback = False
                else:
                    goback = False
            else:
                eddy_map[:,:] = np.nan
                goback = False
    else:
        min_l = l-1
        if min_l < 1:
            min_l = 1
        eddy_map = np.squeeze(E[:,:,min_l])
        
    del var_test
    var_test = np.zeros(np.shape(var))
    var_test[np.nonzero(var_test == 0)] = np.nan
    var_test[np.nonzero(~np.isnan(eddy_map))] = var[np.nonzero(~np.isnan(eddy_map))]
        
    
    #var_test[np.nonzero(np.isnan(eddy_map))] = float('nan')

    [y,x] = np.nonzero(~np.isnan(var_test))
    [edge_x, edge_y, interior_x, interior_y, 
     edge_count, interior_count] = find_edges(var_test,x,y)

    if h0 - (l+1)*delt < -vmax:
        eddy_grow = 0
            
            
    ##
    eddy_range = var_test
    [y,x] = np.nonzero(~np.isnan(var_test))
    [edge_x, edge_y, interior_x, interior_y, 
     edge_count, interior_count] = find_edges(var_test,x,y)
    #### if len(np.nonzero(np.isnan(u_step[0:min_l]))[0]) == len(u_step[0:min_l]):
    ####    u_eddy = np.nan
    #### else:
    ####    u_eddy = np.nanmax(u_step[0:min_l])
    #### leng_eddy = np.empty(np.shape(u_eddy))  
    #### if not(np.isnan(u_eddy)):
    ####    u_ind = max(np.nonzero(u_step == u_eddy)[0])        
    ####    leng_eddy = speed_length(eddy_map,y_range)
    
    #### return (eddy_range, u_eddy, leng_eddy, 
    ####        edge_x, edge_y, interior_x, interior_y,
    ####        edge_count, interior_count)
    return (eddy_range, edge_x, edge_y, interior_x, interior_y,
            edge_count, interior_count)


def eddy_iteration(var, var_test, eddy_track, eddy_flag, 
                   x_range, y_range, hl, hl_max, l, lmin, nmin, nmax, dmax):
        
    crit1 = 0
    crit2 = 0
    crit3 = 0
    crit4 = 0
    crit5 = 0
    crit6 = 0
    new_eddy_flag = np.zeros(np.shape(eddy_flag))
    
    #Store the original eddy iteration locations
    var_old = var_test
    [NY, NX] = np.shape(var)
    eddy_grow = 1
    eddy_ended = 0
    [y,x] = np.nonzero(~np.isnan(var_test))
    n = len(x)
    
    while eddy_grow == 1:
        ncount = n
        
        ##
        # Go through all the currently grouped cells of the eddy, and define
        # them as edge or interior cells
        # Interior cells have all neighbor cells as part of the eddy, edge
        # cells have at least one non-eddy neighbor cell
        [edge_x, edge_y, interior_x, interior_y, 
         edge_count, interior_count] = find_edges(var_test,x,y)
        
        
        ##
        # Check all the adjacent cells to the current iteration of edge cells,
        # and see if they meet the criteria to be part of the eddy
        for f in range(0,edge_count):
            if edge_x[f] != 0:
                if ((var[edge_y[f],edge_x[f]-1] >= hl) and 
                    (var[edge_y[f],edge_x[f]-1] <= hl_max) and
                    (np.isnan(var_test[edge_y[f],edge_x[f]-1])) and
                    (eddy_track[edge_y[f],edge_x[f]-1] == 0)):
                    
                    var_test[edge_y[f],
                             edge_x[f]-1] = var[edge_y[f],
                                                edge_x[f]-1]
                    
            if edge_x[f] != NX-1:
                if ((var[edge_y[f],edge_x[f]+1] >= hl) and 
                    (var[edge_y[f],edge_x[f]+1] <= hl_max) and
                    (np.isnan(var_test[edge_y[f],edge_x[f]+1])) and
                    (eddy_track[edge_y[f],edge_x[f]+1] == 0)):
                    
                    var_test[edge_y[f],
                             edge_x[f]+1] = var[edge_y[f],
                                                edge_x[f]+1]
                    
            if edge_y[f] != 0:                
                if ((var[edge_y[f]-1,edge_x[f]] >= hl) and 
                    (var[edge_y[f]-1,edge_x[f]] <= hl_max) and
                    (np.isnan(var_test[edge_y[f]-1,edge_x[f]])) and
                    (eddy_track[edge_y[f]-1,edge_x[f]] == 0)):
                    
                    var_test[edge_y[f]-1,
                             edge_x[f]] = var[edge_y[f]-1,
                                              edge_x[f]]
                    
            if edge_y[f] != NY-1:
                if ((var[edge_y[f]+1,edge_x[f]] >= hl) and 
                    (var[edge_y[f]+1,edge_x[f]] <= hl_max) and
                    (np.isnan(var_test[edge_y[f]+1,edge_x[f]])) and
                    (eddy_track[edge_y[f]+1,edge_x[f]] == 0)):
                    
                    var_test[edge_y[f]+1,
                             edge_x[f]] = var[edge_y[f]+1,
                                              edge_x[f]]                   
                    
        [y,x] = np.nonzero(~np.isnan(var_test))
        n = len(x)
        del edge_x
        del edge_y
        del interior_x
        del interior_y
        [edge_x, edge_y, interior_x, interior_y, 
         edge_count, interior_count] = find_edges(var_test,x,y)        
                
        
        if n == ncount:
            eddy_grow = 0
        if n == 0:
            eddy_grow = 0
            eddy_ended = 1
            
    ##
    # Check if there is an isoy_rangeed cell (all surrounding cells are NaN)
    if interior_count > nmin:
        for f in range(0,edge_count):
            if edge_x[f] == 0:
                if edge_y[f] == 0:
                    if (np.isnan(var_test[edge_y[f],edge_x[f]+1]) and
                        np.isnan(var_test[edge_y[f]+1,edge_x[f]])):
                        var_test[edge_y[f],edge_x[f]] = np.nan
                elif edge_y[f] == NY-1:
                    if (np.isnan(var_test[edge_y[f],edge_x[f]+1]) and
                        np.isnan(var_test[edge_y[f]-1,edge_x[f]])):
                        var_test[edge_y[f],edge_x[f]] = np.nan
                else:
                    if (np.isnan(var_test[edge_y[f],edge_x[f]+1]) and
                        np.isnan(var_test[edge_y[f]+1,edge_x[f]]) and 
                        np.isnan(var_test[edge_y[f]-1,edge_x[f]])):
                        var_test[edge_y[f],edge_x[f]] = np.nan
            elif edge_x[f] == NX-1:
                if edge_y[f] == 0:
                    if (np.isnan(var_test[edge_y[f],edge_x[f]-1]) and
                        np.isnan(var_test[edge_y[f]+1,edge_x[f]])):
                        var_test[edge_y[f],edge_x[f]] = np.nan
                elif edge_y[f] == NY-1:
                    if (np.isnan(var_test[edge_y[f],edge_x[f]-1]) and
                        np.isnan(var_test[edge_y[f]-1,edge_x[f]])):
                        var_test[edge_y[f],edge_x[f]] = np.nan
                else:
                    if (np.isnan(var_test[edge_y[f],edge_x[f]-1]) and
                        np.isnan(var_test[edge_y[f]+1,edge_x[f]]) and 
                        np.isnan(var_test[edge_y[f]-1,edge_x[f]])):
                        var_test[edge_y[f],edge_x[f]] = np.nan
            else:                    
                if edge_y[f] == 0:
                    if (np.isnan(var_test[edge_y[f],edge_x[f]+1]) and
                        np.isnan(var_test[edge_y[f],edge_x[f]-1]) and
                        np.isnan(var_test[edge_y[f]+1,edge_x[f]])):
                        var_test[edge_y[f],edge_x[f]] = np.nan
                elif edge_y[f] == NY-1:
                    if (np.isnan(var_test[edge_y[f],edge_x[f]+1]) and
                        np.isnan(var_test[edge_y[f],edge_x[f]-1]) and
                        np.isnan(var_test[edge_y[f]-1,edge_x[f]])):
                        var_test[edge_y[f],edge_x[f]] = np.nan
                else:
                    if (np.isnan(var_test[edge_y[f],edge_x[f]+1]) and
                        np.isnan(var_test[edge_y[f],edge_x[f]-1]) and
                        np.isnan(var_test[edge_y[f]+1,edge_x[f]]) and 
                        np.isnan(var_test[edge_y[f]-1,edge_x[f]])):
                        var_test[edge_y[f],edge_x[f]] = np.nan        

    [y,x] = np.nonzero(~np.isnan(var_test))
    n = len(x)
    del edge_x
    del edge_y
    del interior_x
    del interior_y
    [edge_x, edge_y, interior_x, interior_y, 
     edge_count, interior_count] = find_edges(var_test,x,y)
        
    
    
    ##
    ##
    ## Check the five criteria for eddy growth to continue
    if n != 0:
        # CRITERION #1
        # Check that the number of pixels in the eddy does not exceed the
        # maximum size of the eddy
        if n >= nmax:
            crit1 = 1
            
            
        # CRITERION #2
        # Check that there are a minimum number of interior points after
        # having run a minimum number of iterations
        if ((interior_count < nmin) and (l > lmin)):
            crit2 = 1
            
        
        # CRITERION #3
        # Check that the eddy has no neighbor points belonging to another eddy
        # Check all the adjacent cells to the current iteration of edge cells,
        # and make sure that they are not bordering the edge of another
        # eddy 
        n_preborder = n
        var_preborder = var_test
        for f in range(0,edge_count):
            if edge_x[f] != 0:
                if eddy_track[edge_y[f],edge_x[f]-1] != 0:
                    crit3 = 1
            if edge_x[f] != NX-1:
                if eddy_track[edge_y[f],edge_x[f]+1] != 0:
                    crit3 = 1
            if edge_y[f] != 0:
                if eddy_track[edge_y[f]-1,edge_x[f]] != 0:
                    crit3 = 1
            if edge_y[f] != NY-1:
                if eddy_track[edge_y[f]+1,edge_x[f]] != 0:
                    crit3 = 1
                    
                    
        # CRITERION #4
        # Check if the eddy is simply connected (there are no holes in the
        # eddy) (only do this after the first lmin steps)
        for i in range(0,NX):
            for j in range(0,NY):
                if (((np.nansum((edge_x==i+1)*(edge_y==j)) + 
                     np.nansum((edge_x==i-1)*(edge_y==j)) + 
                     np.nansum((edge_x==i)*(edge_y==j+1)) + 
                     np.nansum((edge_x==i)*(edge_y==j-1))) >= 3) and 
                    np.isnan(var_test[j,i])):               
                    
                    new_eddy_flag[j,i]= l
                    crit4 = 1;                    
        
        if interior_count >= nmin:
            interior_ones = np.zeros(np.shape(var_test))
            for i in range(0,len(interior_x)):
                interior_ones[int(interior_y[i]),
                              int(interior_x[i])] = 1
            
            int_connected = is_contiguous(interior_ones)
            if not(int_connected):
                new_eddy_flag[interior_y, interior_x] = l;
                crit4 = 1;       
                
            for i in range(0,len((edge_x))):
                xind = edge_x[i]
                yind = edge_y[i]
                if ((np.nansum((interior_x==xind+1)*(interior_y==yind)) + 
                     np.nansum((interior_x==xind-1)*(interior_y==yind)) + 
                     np.nansum((interior_x==xind)*(interior_y==yind+1)) + 
                     np.nansum((interior_x==xind)*(interior_y==yind-1))) == 0):                                           
                    
                    new_eddy_flag[yind,xind]= l
                    crit4 = 1
            
                        
        # CRITERION #4 END TEST
        if crit4 == 1:
            [newflag_yinds, newflag_xinds] = np.nonzero(new_eddy_flag)
            newmap = np.zeros(np.shape(new_eddy_flag))
            newmap[newflag_yinds, newflag_xinds] = 1
            
            [oldflag_yinds, oldflag_xinds] = np.nonzero(eddy_flag)
            oldmap = np.zeros(np.shape(new_eddy_flag))
            oldmap[oldflag_yinds, oldflag_xinds] = 1
            [matched_yinds, matched_xinds] = np.nonzero(newmap*oldmap)
            
            oldmap_check = np.zeros(np.shape(new_eddy_flag))
            oldmap_check[oldflag_yinds, oldflag_xinds] = 1
            oldmap_check[matched_yinds, matched_xinds] = 0
            [rem_yinds, rem_xinds] = np.nonzero(oldmap_check)
            
            new_eddy_flag[matched_yinds,
                          matched_xinds] = eddy_flag[matched_yinds, matched_xinds]
            new_eddy_flag[rem_yinds,rem_xinds] = 0
            
            [newflag_yinds, newflag_xinds] = np.nonzero(new_eddy_flag)
            if len(newflag_yinds) == 0:
                crit4 = 0
                        
                
                
                
        # CRITERION #5
        # Check the distance between all cells
        # If any cell distance exceeds the maximum allowable distance, revert
        # to the prior set of eddy cells
        dists = np.zeros((edge_count,edge_count))
        for i in range(0,edge_count):
            for j in range(0,edge_count):
                dy = y_range[edge_y[j]] - y_range[edge_y[i]]
                dx = x_range[edge_x[j]] - x_range[edge_x[i]]
                dists[i,j] = np.sqrt(dx**2 + dy**2)
        
        if np.nansum(np.nansum(dists >= dmax)) > 0:
            crit5 = 1

        del dists
        
        
        # CRITERION #6
        # Check that the centroid point is within the 
        # range of eddy cells
        cent_x = np.nanmean(x_range[x])
        cent_y = np.nanmean(y_range[y])
        cent_x = x_range[min(abs(x_range-cent_x)) == abs(x_range-cent_x)][0]
        cent_y = y_range[min(abs(y_range-cent_y)) == abs(y_range-cent_y)][0]
        if (not(any([x_range[ii] == cent_x for ii in x])) or
            not(any([y_range[ii] == cent_y for ii in y]))):
            crit6 = 1
            
        
        ## Check if any of the ending criteria has been met, and if so, end the
        ## eddy-growing procedure
        if (((crit4 == 1) and (crit1+crit2+crit3+crit5+crit6 > 0)) 
            or ((crit4 == 1) and (l > lmin))):
            eddy_ended = 4
            #del var_test
            #var_test = var_old
        elif crit1+crit2+crit3+crit5+crit6 > 0:
            if (crit3 == 1):
                if (l > lmin):
                    eddy_ended = 1   
                    #del var_test
                    #var_test = var_old     
            else:
                eddy_ended = 1   
                #del var_test
                #var_test = var_old                     
            
            
        if eddy_ended > 0:  
            #print('Crit1:',crit1, ',  Crit2:',crit2,',  Crit3:',crit3,
            #     ',  Crit4:',crit4,',  Crit5:',crit5)
            
            for i in range(0,len(edge_x)):
                if ((edge_x[i] != 0) and (edge_x[i] != NX-1) and
                    (edge_y[i] != 0) and (edge_y[i] != NY-1)):
                    if (np.isnan(var_test[edge_y[i],edge_x[i]+1]) and
                        ~np.isnan(var_test[edge_y[i]+1,edge_x[i]+1]) and
                        ~np.isnan(var_test[edge_y[i]-1,edge_x[i]+1]) and
                        (eddy_track[edge_y[i],edge_x[i]+1] == 0) and
                        (var[edge_y[i], edge_x[i]+1])):
                        var_test[edge_y[i],
                                 edge_x[i]+1] = var[edge_y[i],
                                                    edge_x[i]+1]
                    if (np.isnan(var_test[edge_y[i],edge_x[i]-1]) and
                        ~np.isnan(var_test[edge_y[i]+1,edge_x[i]-1]) and
                        ~np.isnan(var_test[edge_y[i]-1,edge_x[i]-1]) and
                        (eddy_track[edge_y[i],edge_x[i]-1] == 0) and 
                        (var[edge_y[i], edge_x[i]-1])):
                        var_test[edge_y[i],
                                 edge_x[i]-1] = var[edge_y[i],
                                                    edge_x[i]-1]
    
           
    [y,x] = np.nonzero(~np.isnan(var_test))
    n = len(x)
    del edge_x
    del edge_y
    del interior_x
    del interior_y
    [edge_x, edge_y, interior_x, interior_y, 
     edge_count, interior_count] = find_edges(var_test,x,y)        
    
    neweddy = var_test  
        
    return n, neweddy, new_eddy_flag, eddy_ended    


def is_contiguous(grid):
    
    items = {(x, y) for x, row in enumerate(grid) for y, f in enumerate(row) if f}
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    neighbours = {(x, y): [(x+dx, y+dy) for dx, dy in directions if (x+dx, y+dy) in items]
                  for x, y in items}

    closed = set()
    fringe = [next(iter(items))]
    while fringe:
        i = fringe.pop()
        if i in closed:
            continue
        closed.add(i)
        for n in neighbours[i]:
            fringe.append(n)

    return items == closed

def find_edges(var,x,y):
    
    [NY,NX] = np.shape(var)
    edge_count = 0
    interior_count = 0
    
    for f in range(0,len(x)):
        if ((x[f] == 0) or (x[f] == NX-1) or (y[f] == 0) or (y[f] == NY-1)):
            edge_count = edge_count + 1
            if edge_count == 1:
                edge_x = [x[f]]
                edge_y = [y[f]]
            else:
                edge_x = np.append(edge_x, x[f])
                edge_y = np.append(edge_y, y[f]) 
        else:
            if (np.isnan(var[y[f],x[f]+1]) or np.isnan(var[y[f],x[f]-1]) or
                np.isnan(var[y[f]+1,x[f]]) or np.isnan(var[y[f]-1,x[f]])):
                edge_count = edge_count + 1
                if edge_count == 1:
                    edge_x = [x[f]]
                    edge_y = [y[f]]
                else:
                    edge_x = np.append(edge_x, x[f])
                    edge_y = np.append(edge_y, y[f])
                
            else:
                interior_count = interior_count + 1
                if interior_count == 1:
                    interior_x = [x[f]]
                    interior_y = [y[f]]
                else:
                    interior_x = np.append(interior_x, x[f])
                    interior_y = np.append(interior_y, y[f])
            
                    
    if edge_count == 0:
        edge_x = np.empty(1)
        edge_y = np.empty(1)
    if interior_count == 0:
        interior_x = np.empty(1)
        interior_y = np.empty(1)
                
    return edge_x, edge_y, interior_x, interior_y, edge_count, interior_count   

def eddy_id_singletime(x, y, var, sz_x=45, shore_x=50, szthresh=0.5, offthresh=0.1, delt = 0.009, dmax = 4, nmin = 50):
    # find local minimums and maximums 
    count, vmax, pole, x_ind, y_ind = find_local_max(x, y, var)

    # remove local minimums and maximums below threshold
    x_thresh = offthresh*np.ones(len(x))
    sz_ind = np.where(min(abs(x-sz_x)) == abs(x-sz_x))[0][0]
    sh_ind = np.where(min(abs(x-shore_x)) == abs(x-shore_x))[0][0]
    sz_slope = (szthresh-offthresh)/(shore_x-sz_x)
    x_thresh[sz_ind:sh_ind] = (offthresh + 
                               (sz_slope*(x[sz_ind:sh_ind]-x[sz_ind])))
    x_thresh[sh_ind:] = szthresh
    
    x_ind_thresh = x_thresh[x_ind]
    good_thresh_inds = np.where([abs(vmax[ii]) >= x_ind_thresh[ii] 
                                 for ii in range(0,len(vmax))])[0]
    
    pole = pole[good_thresh_inds]
    x_ind = x_ind[good_thresh_inds]
    y_ind = y_ind[good_thresh_inds]
    vmax = vmax[good_thresh_inds]

    count = len(vmax)

    # sort maximum and minimums, in decreasing magnitude 
    vmax_abs = np.abs(vmax)
    local_data = np.array([vmax_abs[:], vmax[:], pole[:], x_ind[:], y_ind[:]])
    local_sorted = local_data[:,np.argsort(-vmax_abs)]

    vmax_abs_sort = local_sorted[0,:]
    vmax_sort = local_sorted[1,:]
    pole_sort = local_sorted[2,:]
    x_sort_ind = local_sorted[3,:].astype(int)
    y_sort_ind = local_sorted[4,:].astype(int)
    x_sort = x[x_sort_ind]
    y_sort = y[y_sort_ind]

    # Initialize eddy tracking maps
    eddy_tracknum = 0
    eddy_map = np.zeros(np.shape(var))
    eddy_map[eddy_map == 0] = np.nan
    eddy_track_map = np.zeros(np.shape(var))

    # Define eddy growing parameters to be used
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xmax_search = np.ceil(dmax/dx)
    ymax_search = np.ceil(dmax/dy)
    nmax = (xmax_search * ymax_search)

    eddy_go = 0

    leng = []; spin = []; xc = []; yc = []
    eddy_num = 1
    Nx = var.shape[-1]
    Ny = var.shape[-2]

    for e in range(0,count):
        # If the current local min./max. is not located in a previous eddy, start the growing procedure
        # Do this by checking the current cell, and all adjacent cells to ensure that
        # they are not already defined as part of another eddy
        if eddy_track_map[y_sort_ind[e],x_sort_ind[e]] == 0:
            eddy_go = local_eddy_check(Nx,Ny,x_sort_ind[e],y_sort_ind[e],eddy_track_map)
        else:
            eddy_go = 0

        if eddy_go == 1:
            # Increase the number of tracked eddies
            eddy_tracknum = eddy_tracknum + 1     

            # Define the eddy search region parameters
            eddy_centx = x_sort[e]
            eddy_centy = y_sort[e]

            vmax_cur_abs = vmax_abs_sort[e]
            vmax_cur = vmax_sort[e]
            xcent = x_sort_ind[e]
            ycent = y_sort_ind[e]

            xmin = xcent - xmax_search
            xmax = xcent + xmax_search
            if xmin < 0:
                xmin = 0
            if xmax > Nx-1:
                xmax = Nx-1

            ymin = ycent - ymax_search
            ymax = ycent + ymax_search
            if ymin < 0:
                ymin = 0
            if ymax > Ny-1:
                ymax = Ny-1

            # convert max search region to int
            xmin = int(xmin); xmax = int(xmax); ymin = int(ymin); ymax = int(ymax)

            # Extract the eddy search region, based on the above parameters
            x_range = x[xmin:xmax]
            y_range = y[ymin:ymax]
            var_range = var[ymin:ymax,xmin:xmax]*pole_sort[e]            
            eddy_track_range = eddy_track_map[ymin:ymax, xmin:xmax]

            [range_centx, range_centy] = np.where(var_range == vmax_cur_abs)

            ## Grow the eddy, using the above parameters and range     
            [var_range_eddy, edge_x, edge_y, 
             interior_x, interior_y, 
             edge_count, interior_count] = eddy_grow(var_range, 
                                                     eddy_track_range, 
                                                     x_range, y_range, 
                                                     vmax_cur_abs, delt, 
                                                     nmin, nmax, dmax)

            [Ny_range, Nx_range] = np.shape(var_range_eddy)

            [eddy_yinds, eddy_xinds] = np.where(~np.isnan(var_range_eddy) == 1)

            leng_temp = len(eddy_yinds)*dx*dy/np.pi 

            ## Check that the eddy meets certain criteria, and if it does,
            ## save relevant info on the eddy
            if (interior_count >= nmin):
                xc_sum = 0
                yc_sum = 0
                h_sum = 0 
                leng.append(leng_temp)
                spin.append(pole_sort[e])

                for f in range(0,len(eddy_xinds)):
                    xc_sum = xc_sum + x[eddy_xinds[f]+xmin]*np.abs(var[eddy_yinds[f]+ymin, eddy_xinds[f]+xmin])
                    yc_sum = yc_sum + y[eddy_yinds[f]+ymin]*np.abs(var[eddy_yinds[f]+ymin, eddy_xinds[f]+xmin])
                    h_sum = h_sum + np.abs(var[eddy_yinds[f]+ymin, eddy_xinds[f]+xmin])

                xc.append(xc_sum/h_sum)
                yc.append(yc_sum/h_sum)

                for f in range(0,len(eddy_xinds)):
                    #eddy_map[eddy_yinds[f]+ymin,
                    #         eddy_xinds[f]+xmin] = var[eddy_yinds[f]+ymin,
                    #                                   eddy_xinds[f]+xmin]
                    eddy_track_map[eddy_yinds[f]+ymin,
                                   eddy_xinds[f]+xmin] = int(eddy_num)
                eddy_num += 1
    
    return eddy_track_map, leng, spin, xc, yc