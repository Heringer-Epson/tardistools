#!/usr/bin/env python

import os                                                               
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy

from scipy.optimize import curve_fit, OptimizeWarning
from astropy import constants as const

import astropy
import math         
from math import factorial    

warnings.simplefilter('error',OptimizeWarning)

class Utility(object):
    """ Utility functions to assist on computing spectral features.
    Check separation.
    """
    
    def __init__(self, sep=20.):
        self.utility = True
        self.sep = sep
        self.keys = ['6', '7']
        self.keys_to_fit = ['6', '7']
        
    def rms(self,y_data,y_pred):
        rms_array = math.sqrt(((y_data - y_pred)**2.).mean())
        rms_array = np.asarray(rms_array).astype('float')
        rms_array[rms_array == 0] = 1.e-5 
        return rms_array

    def make_parabola(self, x_ref):
        def parabola(x,a,b,c):
            return a*(x-x_ref)**2.+b*(x-x_ref)+c
        return parabola     
    
class Analyse_Spectra(Utility):

    """ Computes a set of spectral features, given an input dictioanry
    containing ['wavelength_raw'] and ['flux_raw'].

    Parameters
    ----------
    dataframe : ~pandas dataframe
        Each row of the dataframe corresponds to a spectrum and thus needs to
        contain 'wavelength_raw' and 'flux_raw' columns.

    smoothing_mode : ~str
       'savgol' will use the Savitzky-Golay filter and is the default option.
       Other filters are not implemented at the moment.
                           
    smoothing_window : ~float
        Window to be used by the Savitzky-Golay filter to smooth the spectra.
        
    deredshift_and_normalize : boolean
        Flag to whether or not de-redshift the spectrum
    
    verbose : ~boolean
        Flag to whether or not print extra information.        
                    
    Returns
    -------
    self.DF : ~pandas dataframe
        Input dataframe where spectral features have been added as new columns. 
    """

    def __init__(self, dataframe, smoothing_mode='savgol', smoothing_window=21,
                 deredshift_and_normalize=True, verbose=False):
        
        Utility.__init__(self)

        self.time = time.time()   
        self.DF = dataframe
        self.smoothing_mode = smoothing_mode
        self.smoothing_window = smoothing_window
        self.deredshift_and_normalize = deredshift_and_normalize
        self.verbose = verbose
                
        #Boundaries of line regions. From Silverman+ 2012 (paper II).
        self.MD = {}    
        
        self.MD['rest_f1'] = [3945.28]
        self.MD['blue_lower_f1'], self.MD['blue_upper_f1'] =3400., 3800.
        self.MD['red_lower_f1'], self.MD['red_upper_f1'] = 3800., 4100.
             
        self.MD['rest_f2'] = [4129.73]
        self.MD['blue_lower_f2'], self.MD['blue_upper_f2'] = 3850., 4000.
        self.MD['red_lower_f2'], self.MD['red_upper_f2'] = 4000., 4150.
           
        #rest flux is the upper red bound for uniform selection criteria.
        self.MD['rest_f3'] = [4700.]
        self.MD['blue_lower_f3'], self.MD['blue_upper_f3'] = 4000., 4150.
        self.MD['red_lower_f3'], self.MD['red_upper_f3'] = 4350., 4700. 
                
        #rest flux is the upper red bound for uniform selection criteria.
        self.MD['rest_f4'] = [5550.]
        self.MD['blue_lower_f4'], self.MD['blue_upper_f4'] = 4350., 4700.
        self.MD['red_lower_f4'], self.MD['red_upper_f4'] = 5050., 5550. 
         
        self.MD['rest_f5'] = [5624.32]
        self.MD['blue_lower_f5'], self.MD['blue_upper_f5'] = 5100., 5300.
        self.MD['red_lower_f5'], self.MD['red_upper_f5'] = 5450., 5700.
               
        self.MD['rest_f6'] = [5971.85]
        self.MD['blue_lower_f6'], self.MD['blue_upper_f6'] = 5400., 5700.
        self.MD['red_lower_f6'], self.MD['red_upper_f6'] = 5750., 6000.
     
        self.MD['rest_f7'] = [6355.21]
        self.MD['blue_lower_f7'], self.MD['blue_upper_f7'] = 5750., 6060.
        self.MD['red_lower_f7'], self.MD['red_upper_f7'] = 6200., 6600.
               
        self.MD['rest_f8'] = [7773.37]
        self.MD['blue_lower_f8'], self.MD['blue_upper_f8'] = 6800., 7450.
        self.MD['red_lower_f8'], self.MD['red_upper_f8'] = 7600., 8000.
        
        self.MD['rest_f9'] = [8498., 8542., 8662.]
        self.MD['blue_lower_f9'], self.MD['blue_upper_f9'] = 7500., 8100.
        self.MD['red_lower_f9'], self.MD['red_upper_f9'] = 8200., 8900.

        if self.verbose:
            print '\n*STARTING SPECTRAL ANALAYSIS.'
        
    def deredshift_spectrum(self):
        """Data downloaded from BSNIP is not in rest-wavelength."""
        start_time = time.time()
        def remove_redshift(wavelength, redshift):      
            try:
                wavelength = np.asarray(wavelength).astype(np.float)
                redshift = float(redshift)
                wavelength = wavelength / (1. + redshift)
            except:
                wavelength = np.full(len(wavelength), np.nan)
            return wavelength
       
        self.DF['wavelength_raw'] = self.DF.apply(
          lambda row: pd.Series([remove_redshift(row['wavelength_raw'],
          row['host_redshift'])]), axis=1) 
        
        if self.verbose:
            print ('  -RAN: De-redshifting the spectra. FINISHED IN ('
                   +str(format(time.time()-start_time, '.1f'))+'s)')

    def normalize_flux(self):
        """ Normalize the flux to relative units so that methods such as
        wavelet smoothing can be applied if requested.
        """ 
        start_time = time.time()

        def get_normalized_flux(wavelength, flux):          
            aux_wavelength = np.asarray(wavelength).astype(np.float)
            aux_flux = np.asarray(flux).astype(np.float)                    
            
            #Wavelength window where the mean flux is computed.
            window_condition = ((wavelength >= 4000.) & (wavelength <= 9000.))             
            
            flux_window = aux_flux[window_condition]
            #normalization_factor = np.mean(flux_window)     
            normalization_factor = max(aux_flux)     
            
            aux_flux = aux_flux / normalization_factor
            aux_flux = list(aux_flux)
            return aux_flux
       
        self.DF['flux_normalized'] = self.DF.apply(
          lambda row: get_normalized_flux(row['wavelength_raw'],
        row['flux_raw']), axis=1)   
       
        if self.verbose:
            print ('  -RAN: Normalizing flux to maximum. FINISHED IN ('
              +str(format(time.time()-start_time, '.1f'))+'s)')

    def smooth_spectrum(self):
        """  Smooth the spectrum using either the savgol-golay.
        Other options not implemented at the moment.
        """                                 
        start_time = time.time()

        def savitzky_golay(y, window_size, order, deriv=0, rate=1):     
            """This was taken from 
            http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
            The package that can be directly imported was conflicting with
            ?numpy?.
            """
            try:
                window_size = np.abs(np.int(window_size))
                order = np.abs(np.int(order))
            except ValueError, msg:
                raise ValueError("window_size and order have to be of type int")
            if window_size % 2 != 1 or window_size < 1:
                raise TypeError("window_size size must be a positive odd number")
            if window_size < order + 2:
                raise TypeError("window_size is too small for the polynomials order")
            order_range = range(order+1)
            half_window = (window_size -1) // 2
            # precompute coefficients
            b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
            m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
            # pad the signal at the extremes with
            # values taken from the signal itself
            firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
            lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
            y = np.concatenate((firstvals, y, lastvals))
            return np.convolve( m[::-1], y, mode='valid')

        def do_smooth(wavelength,flux):
            """Smooth the spectrum so that the boundaries of features
            can be more easily identified.
            """
            wavelength = np.asarray(wavelength).astype(np.float)
            flux = np.asarray(flux).astype(np.float)
            
            #f_smoothed = savgol_filter(flux, self.smoothing_window, 3)
            f_smoothed = savitzky_golay(flux, self.smoothing_window, 3)
            
            df = np.asarray([f - f_next for f, f_next in zip(
              f_smoothed,f_smoothed[1:])])
            dw = np.asarray([w - w_next for w, w_next in zip(
              wavelength,wavelength[1:])])                      
            derivative = savitzky_golay(np.asarray([np.nan]+list(df/dw)),
              self.smoothing_window, 3)            
            return f_smoothed, derivative

        flux_smoothed = self.DF.apply(
          lambda row: pd.Series(do_smooth(row['wavelength_raw'],
          row['flux_normalized'])), axis=1)
        
        flux_smoothed.columns = ['flux_smoothed', 'derivative']
        self.DF = self.DF.join(flux_smoothed)            
        self.DF['wavelength_smoothed'] = self.DF['wavelength_raw']                             
        
        if self.verbose:
            print ('  -RAN: Smoothing flux. FINISHED IN ('
                   +str(format(time.time()-start_time, '.1f'))+'s)')

    def find_zeros_in_features(self):
        """ Find where the deepest minimum in the feature region is. Then
        selected the closest maxima to the red and blue as the boundaries of
        the feature. If the deepest minimum has no maximum either to the red or
        to the blue, then select the next deepest minimum. Once the 'true'
        minimum is determined, if there are more than one maximum to the red
        or blue, then check if the nearest maxima are not shoulders by checking
        for the presence another minimum withing the self.sep window of the
        nearest maximum. If the maximum is deemed as a shoulder and if
        there is another bluer/redder minimum bounded by another maximum,
        then determine this minimum as the true one.
        """         
        start_time = time.time()
        def get_zeros(wavelength, flux, derivative, key):
            wavelength = np.asarray(wavelength).astype(np.float)
            flux = np.asarray(flux).astype(np.float)
            derivative = np.asarray(derivative).astype(np.float)

            window_condition = ((wavelength >= self.MD['blue_lower_f'+key])
                                & (wavelength <= self.MD['red_upper_f'+key]))          
                  
            w_window = wavelength[window_condition]
            f_window = flux[window_condition]
            #print f_window
            der_window = derivative[window_condition]    
            
            idx_minima_window = [idx for idx, (w,f,der,der_next)
              in enumerate(zip(w_window,f_window,der_window,der_window[1:]))
              if (np.sign(der) != np.sign(der_next)) and (der_next > 0.)
              and w < max(self.MD['rest_f'+key])]  
                    
            idx_maxima_window = [idx for idx, (w,f,der,der_next)
              in enumerate(zip(w_window,f_window,der_window,der_window[1:]))
              if (np.sign(der) != np.sign(der_next)) and (der_next < 0.)]           
    
            w_minima_window = np.asarray([w_window[idx] for idx
              in idx_minima_window])         
            f_minima_window = np.asarray([f_window[idx] for idx
              in idx_minima_window])            
           
            w_maxima_window = np.asarray([w_window[idx] for idx
              in idx_maxima_window])
            f_maxima_window = np.asarray([f_window[idx] for idx
              in idx_maxima_window])

            #Find where the true minimum of the feature is.
            #Iterate over the minima to find the deepest point
            #that contains a maximum to the right and to the left.          
            
            copy_w_minima_window = w_minima_window[:]
            copy_f_minima_window = f_minima_window[:]

            for i in range(len(w_minima_window)):
                if len(copy_w_minima_window) > 0:
                    
                    #Get the deepest minimum.
                    w_min = copy_w_minima_window[copy_f_minima_window.argmin()]
                    f_min = min(copy_f_minima_window)
                    
                    #Trimming minima and maxima in feature window:
                    #Select only minima/maxima in the left (right) side of the
                    #true minimum for the blue (red) window. These are bounded
                    #by the pre-fixed limits for the window and the position
                    #of the true minimum. 
                   
                    min_blue_condition = (w_minima_window < w_min)
                    min_red_condition = (w_minima_window > w_min)
                                     
                    max_blue_condition = (w_maxima_window < w_min)
                    max_red_condition = (w_maxima_window > w_min)

                    minima_window_blue_condition = (min_blue_condition
                      & (w_minima_window <= self.MD['blue_upper_f'+key])
                      & (w_minima_window >= self.MD['blue_lower_f'+key]))
                              
                    maxima_window_blue_condition = (max_blue_condition
                    & (w_maxima_window <= self.MD['blue_upper_f'+key])
                    & (w_maxima_window >= self.MD['blue_lower_f'+key]))              
                   
                    minima_window_red_condition = (min_red_condition
                      & (w_minima_window <= self.MD['red_upper_f'+key])
                      & (w_minima_window >= self.MD['red_lower_f'+key]))                      
                   
                    maxima_window_red_condition = (max_red_condition
                      & (w_maxima_window <= self.MD['red_upper_f'+key])
                      & (w_maxima_window >= self.MD['red_lower_f'+key]))             
                                    
                    w_minima_window_blue = w_minima_window[
                      minima_window_blue_condition]
                    f_minima_window_blue = f_minima_window[
                      minima_window_blue_condition]  
                                    
                    w_maxima_window_blue = w_maxima_window[
                      maxima_window_blue_condition]
                    f_maxima_window_blue = f_maxima_window[
                      maxima_window_blue_condition]  
                                 
                    w_minima_window_red = w_minima_window[
                      minima_window_red_condition]
                    f_minima_window_red = f_minima_window[
                      minima_window_red_condition]    
                    
                    w_maxima_window_red = w_maxima_window[
                      maxima_window_red_condition]
                    f_maxima_window_red = f_maxima_window[
                      maxima_window_red_condition]    

                    #Select the maxima to the right and to the left of the
                    #Minimum determined above.
                    try:
                        w_max_blue = w_maxima_window_blue[-1]
                        f_max_blue = f_maxima_window_blue[-1]
                   
                        w_max_red = w_maxima_window_red[0]
                        f_max_red = f_maxima_window_red[0]
           
                    except:
                        w_max_blue, f_max_blue = np.nan, np.nan
                        w_max_red, f_max_red = np.nan, np.nan                            

                    #If there is no maximum to either the left or to the right,
                    #remove the minimum from the list of minima and
                    #try the next deepest minimum.
                    if not np.isnan(w_max_blue) and not np.isnan(w_max_red):
                        break
                    else:
                        copy_w_minima_window = np.asarray(
                          filter(lambda x : x != w_min, copy_w_minima_window))
                        copy_f_minima_window = np.asarray(
                          filter(lambda x : x != f_min, copy_f_minima_window))  

            if len(copy_w_minima_window) == 0: 
                w_min, f_min = np.nan, np.nan
                w_max_blue, f_max_blue = np.nan, np.nan
                w_max_red, f_max_red = np.nan, np.nan     

            #Once the true minimum is known, check whether the nearest maxima
            #are just shoulders.
            if not np.isnan(w_max_blue) and len(w_maxima_window_blue)>1:   
                
                #Compute wavelength separation between minima to the maximum.
                d_minima_window_blue = w_minima_window_blue - w_max_blue   
                
                #Select only the minima which are bluer than the maximum
                #and within the separation window.
                d_minima_window_blue = d_minima_window_blue[
                  (d_minima_window_blue < 0.)
                  & (d_minima_window_blue > -1.*self.sep)]
                  
                #Take the bluest of the minima and check
                #if there is another maximum bluer than that.
                if len(d_minima_window_blue)>0:
                    condition = (w_maxima_window_blue < w_max_blue
                                 + min(d_minima_window_blue))                  
                    w_maxima_window_blue = w_maxima_window_blue[condition]
                    f_maxima_window_blue = f_maxima_window_blue[condition]
                    if len(w_maxima_window_blue) >= 1:
                        w_max_blue = w_maxima_window_blue[-1]
                        f_max_blue = f_maxima_window_blue[-1]

            if not np.isnan(w_max_red) and len(w_maxima_window_red)>1: 
                
                #Compute wavelength separation between minima to the maximum.
                d_minima_window_red = w_minima_window_red - w_max_red  
               
                #Select only the minima which are redder than the maximum
                #and within the separation window.
                d_minima_window_red = d_minima_window_red[
                  (d_minima_window_red > 0.)
                  & (d_minima_window_red < 1.*self.sep)]
              
                #Take the reddest of the minima and check
                #if there is another maximum bluer than that.
                if len(d_minima_window_red)>0:
                    condition = (w_maxima_window_red > w_max_red
                                 + max(d_minima_window_red))
                    w_maxima_window_red = w_maxima_window_red[condition]
                    f_maxima_window_red = f_maxima_window_red[condition]
                    if len(w_maxima_window_red) >= 1:
                        w_max_red = w_maxima_window_red[0]
                        f_max_red = f_maxima_window_red[0]
                                        
            return float(w_min), float(f_min), float(w_max_blue), \
                   float(f_max_blue), float(w_max_red), float(f_max_red)
        
        for key in self.keys:
                       
            feature_zeros = self.DF.apply(
              lambda row: pd.Series(get_zeros(
              row['wavelength_smoothed'],row['flux_smoothed'],
              row['derivative'],key)), axis=1)
              
            feature_zeros.columns = [
              'wavelength_minima_f'+key, 'flux_minima_f'+key,
              'wavelength_maxima_blue_f'+key, 'flux_maxima_blue_f'+key,
              'wavelength_maxima_red_f'+key, 'flux_maxima_red_f'+key]
              
            self.DF = self.DF.join(feature_zeros)                           
        if self.verbose:
            print ('  -RAN: Determining zeros that define the pseudo spectrum.'
                   +' FINISHED IN ('+str(format(time.time()-start_time, '.1f'))
                   +'s)')
        
    def grab_feature_regions(self):
        """ Store the region of the features (boundaries determined at
        find_zeros_in_features) in order to facilitate computing features. 
        """
        start_time= time.time()
    
        def isolate_region(wavelength, flux_normalized, flux_smoothed,
                           blue_boundary, red_boundary):       
                                        
            wavelength = np.asarray(wavelength).astype(np.float)
            flux_normalized = np.asarray(flux_normalized).astype(np.float)
            flux_smoothed = np.asarray(flux_smoothed).astype(np.float)
            
            if not np.isnan(blue_boundary) and not np.isnan(red_boundary): 
                
                region_condition = ((wavelength >= blue_boundary)
                                    & (wavelength <= red_boundary))      
               
                wavelength_region = wavelength[region_condition]
                flux_normalized_region = flux_normalized[region_condition]
                flux_smoothed_region = flux_smoothed[region_condition]           
        
            else:
                wavelength_region = [np.nan]
                flux_normalized_region = [np.nan]
                flux_smoothed_region = [np.nan]   
           
            return wavelength_region, flux_normalized_region, \
                   flux_smoothed_region
        
        for key in self.keys:
           
            feature_region = self.DF.apply(
              lambda row: pd.Series(isolate_region(
              row['wavelength_raw'], row['flux_normalized'], 
              row['flux_smoothed'], row['wavelength_maxima_blue_f'+key],
              row['wavelength_maxima_red_f'+key])), axis=1)
                  
            feature_region.columns = [
              'wavelength_region_f'+key, 'flux_normalized_region_f'+key,
              'flux_smoothed_region_f'+key]
              
            self.DF = self.DF.join(feature_region)      
        
        if self.verbose:
            print ('  -RAN: Grabing the region of each feature.'
                   +'FINISHED IN ('+str(format(time.time()-start_time, '.1f'))
                   +'s)')

    def make_pseudo_continuum(self):
        """ The pseudo continuum slope is simply a line connecting the
        feature region boundaries. It depends only on the wavelength array and
        boundary values. The latter coming from smoothed quantities,
        regardless of method chosen (raw or smoothed.)
        """         
        start_time = time.time()
        def get_psedo_continuum_flux(w,x1,y1,x2,y2,f_smoothed):
           
            try:    
                w = np.asarray(w)
                f_smoothed = np.asarray(f_smoothed)
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
             
                def pseudo_cont(x):
                    return slope*x+intercept
              
                pseudo_flux = pseudo_cont(w)            

                #Check whether the continuum is always higher than the
                #**smoothed** flux and the array contains more than one element.
                boolean_check = [(f_s-f_c)>0.01 for (f_c,f_s)
                                 in zip(pseudo_flux, f_smoothed)]
                       
                if True in boolean_check or len(pseudo_flux) < 1:
                    pseudo_flux = 'Failed'

            except:
                pseudo_flux = 'Failed'
            return pseudo_flux                                          
        
        for key in self.keys:
            
            pseudo_cont_flux = self.DF.apply(
              lambda row: pd.Series([get_psedo_continuum_flux(
              row['wavelength_region_f'+key],
              row['wavelength_maxima_blue_f'+key],
              row['flux_maxima_blue_f'+key],
              row['wavelength_maxima_red_f'+key],
              row['flux_maxima_red_f'+key],
              row['flux_smoothed_region_f'+key])]), axis=1)  
                      
            pseudo_cont_flux.columns = ['pseudo_cont_flux_f'+key]
            self.DF = self.DF.join(pseudo_cont_flux)
       
        if self.verbose:
            print ('  -RAN: Making the pseudo continuum.'
                   +' FINISHED IN ('+str(format(time.time()-start_time, '.1f'))
                   +'s)')

    def compute_pEW(self):
        """ Compute the pEW of features.
        """
        start_time = time.time()
        def get_pEW(wavelength_region, flux_region, pseudo_flux):
            try:
                if len(wavelength_region) > 1:
                    pEW = 0.
                    
                    #Last entry not included so that differences can be computed.
                    for i, (w, f, p) in enumerate(zip(
                      wavelength_region[0:-1],
                      flux_region[0:-1],
                      pseudo_flux[0:-1])): 
                          
                        delta_lambda = abs(
                          wavelength_region[i+1] - wavelength_region[i])
                        
                        pEW += delta_lambda * (p - f) / p
                        
                else:
                    pEW = np.nan                                    
           
            except:
                pEW = np.nan        
            
            return pEW

        for key in self.keys:
           
            pEW_value = self.DF.apply(
              lambda row: pd.Series(get_pEW(
              row['wavelength_region_f'+key], 
              row['flux_normalized_region_f'+key],
              row['pseudo_cont_flux_f'+key])), axis=1)   
              
            pEW_value.columns = ['pEW_f'+key]
            self.DF = self.DF.join(pEW_value)
       
        if self.verbose:
            print ('  -RAN: Computing the pEW. FINISHED IN ('
                   +str(format(time.time()-start_time, '.1f'))+'s)')

    def compute_smoothed_velocity_and_depth(self):
        """ Compute the velocity of the features according to the rest
        wavelength of the line forming the feature.
        The velocity is computed by fitting a parabola to the minimum of the
        feature.
        """         
        start_time = time.time()
       
        def get_smoothed_velocity(wavelength_region, flux_region,
                                  pseudo_cont_flux,rest_wavelength):
                                      
            try:
                wavelength_region = np.asarray(wavelength_region).astype(np.float)
                flux_region = np.asarray(flux_region).astype(np.float)
                pseudo_cont_flux = np.asarray(pseudo_cont_flux).astype(np.float)           
                                
                flux_at_min = min(flux_region)
                wavelength_at_min = wavelength_region[flux_region.argmin()]
                pseudo_cont_at_min = pseudo_cont_flux[flux_region.argmin()] 
                
                wavelength_par_window = wavelength_region[
                  (wavelength_region >= wavelength_at_min - self.sep)
                  & (wavelength_region <= wavelength_at_min + self.sep)]
               
                flux_par_window = flux_region[
                  (wavelength_region >= wavelength_at_min - self.sep)
                  & (wavelength_region <= wavelength_at_min + self.sep)]
                             
                par_to_fit = self.make_parabola(wavelength_at_min)                     
                popt, covt = curve_fit(par_to_fit, wavelength_par_window, 
                                       flux_par_window)
                        
                rest_wavelength = sum(rest_wavelength) / len(rest_wavelength)
                wavelength_par_min = wavelength_at_min - popt[1] / (2*popt[0])
                flux_par_min = par_to_fit(wavelength_par_min, popt[0], popt[1],
                                          popt[2])        
                velocity = (const.c.to('km/s').value
                  * ((wavelength_par_min / rest_wavelength)**2. - 1.)
                  / ((wavelength_par_min / rest_wavelength)**2. + 1.)
                  / 1.e3)
                depth = 1. - flux_par_min / pseudo_cont_at_min
                                
                if popt[0] < 0. or velocity > 0. or velocity < -30000.:         
                    velocity = np.nan                    
            
            except:                 
                wavelength_par_min, flux_par_min = np.nan, np.nan
                velocity, depth = np.nan, np.nan            
            
            return wavelength_par_min, flux_par_min, velocity, depth    
    
        for key in self.keys_to_fit:
          
            velocity_from_smoothing = self.DF.apply(
              lambda row: pd.Series(get_smoothed_velocity(
              row['wavelength_region_f'+key], 
              row['flux_normalized_region_f'+key], 
              row['pseudo_cont_flux_f'+key], 
              self.MD['rest_f'+key])), axis=1)   
                       
            velocity_from_smoothing.columns = [
              'wavelength_at_min_f'+key, 'flux_at_min_f'+key, 
              'velocity_f'+key, 'depth_f'+key]
       
            self.DF = self.DF.join(velocity_from_smoothing)
    
        if self.verbose:
            print ('  -RAN: Computing line velocities from minima in smoothed '
                   +'spectra. FINISHED IN ('
                   +str(format(time.time()-start_time, '.1f'))+'s)')
    
    def run_analysis(self):
        if self.deredshift_and_normalize:
            self.deredshift_spectrum()
            self.normalize_flux()    
        self.smooth_spectrum()  
        self.find_zeros_in_features()
        self.grab_feature_regions()
        self.make_pseudo_continuum()
        self.compute_pEW()
        self.compute_smoothed_velocity_and_depth()
        
        if self.verbose:
            print ("    -TOTAL TIME IN SPECTRAL ANALYSIS: "
                   +str(format(time.time()-self.time, '.1f'))+'s')            
            print '    *** RUN COMPLETED SUCCESSFULLY ***\n'                        
       
        return self.DF  

class Compute_Uncertainty(Utility):
    """Uses a MC approach to compute the uncertainty of spectral features.
    As a guideline, this follows Liu+ 2016
    [[http://adsabs.harvard.edu/abs/2016ApJ...827...90L]].

    Parameters
    ----------
    dataframe : ~pandas dataframe
        Each row of the dataframe corresponds to a spectrum and thus needs to
        contain 'wavelength_raw' and 'flux_raw' columns.

    smoothing_mode : ~str
       'savgol' will use the Savitzky-Golay filter and is the default option.
       Other filters are not implemented at the moment.
                           
    smoothing_window : ~float
        Window to be used by the Savitzky-Golay filter to smooth the spectra.
        
    N_MC_runs : ~float
        Number of spectra with noise artificially added for the MC run.   

    verbose : ~boolean
        Flag to whether or not print extra information. 

    Notes
    -----
    1) The uncertainties estimated using the MC combined with noise estimation
    from rms is systematically different than the 'true' uncertainty obtained
    when running multiple TARDIS simulations with different seeds. This
    discrepancy is different than correcting the rms because of the smoothing
    procedure and is not corrected in this code. Therefore it has to be taken
    into account when quoting the uncertainties computed here.
                        
    Returns
    -------
    self.DF : ~pandas dataframe
        Input dataframe where spectral features have been added as new columns. 
    """

    def __init__(self, dataframe, smoothing_mode='savgol',
                 smoothing_window=21, N_MC_runs=3000, verbose=False):
        
        self.time = time.time()
        
        print '\n*STARTING CALCULATION OF UNCERTAINTIES.'
        
        Utility.__init__(self)
            
        self.df = dataframe
        self.smoothing_mode = smoothing_mode
        self.smoothing_window = smoothing_window
        self.N_MC_runs = N_MC_runs
        self.verbose = verbose

        #Relatively small correction needed due to the fact that the smoothed
        #spectra 'follows' the noise, leading to a smaller than expected rms noise.
        if smoothing_window == 21:
            self.smoothing_correction = (1. / 0.93)
        elif smoothing_window == 51:
            self.smoothing_correction = 1. / 0.96   
        else:
            raise ValueError("Smoothing correction not defined for this"
                             +"smoothing window.")

    def compute_flux_rms(self):
        """ Estimate the flux noise in each pixel using a simple rms
        in a bin defined by the self.sep parameter.
        """                                 
       
        print '  -RUNNING: Computing flux rms pixel-wise.'     
        start_time = time.time()
        
        def get_rms(wavelength, flux_normalized, flux_smoothed):
            
            try:
                wavelength = np.asarray(wavelength).astype(np.float)
                flux_normalized = np.asarray(flux_normalized).astype(np.float)
                flux_smoothed = np.asarray(flux_smoothed).astype(np.float)     
          
                flux_normalized_bins = [
                  flux_normalized[(wavelength >= w - self.sep)
                  & (wavelength <= w + self.sep)] for w in wavelength]
               
                flux_smoothed_bins = [
                  flux_smoothed[(wavelength >= w - self.sep)
                  & (wavelength <= w + self.sep)] for w in wavelength]      
                
                rms = [self.rms(f_raw_bins, f_smoothed_bins)
                       *self.smoothing_correction
                       for (f_raw_bins, f_smoothed_bins)
                       in zip(flux_normalized_bins,flux_smoothed_bins)]
           
            except:
                rms = np.nan
        
            if self.verbose:
                print ('    -Median of the smoothing corrected rms = '+
                       str(np.median(rms)))
                print ('    -Medan of the smoothing corrected rms = '+
                       str(np.mean(rms)))
          
            return rms
    
        flux_rms = self.df.apply(
          lambda row: pd.Series([get_rms(
          row['wavelength_raw'], row['flux_normalized'], 
          row['flux_smoothed'])]), axis=1)    
          
        flux_rms.columns = ['flux_rms']
        self.df = self.df.join(flux_rms)        
    
        print '    -DONE ('+str(format(time.time()-start_time, '.1f'))+'s)'          

    def compute_mock_spectra(self, index):
        """
        Create N_runs mock spectra for each spectrum in the dataframe.
        The are to be used to compute uncertainties in the pEW and
        velocity measurements using a MC technique.
        """     
        try:
            wavelength_raw = np.asarray(
              self.df.loc[index]['wavelength_raw']).astype(np.float)
            flux_normalized = np.asarray(
              self.df.loc[index]['flux_normalized']).astype(np.float)
            flux_rms = np.asarray(
              self.df.loc[index]['flux_rms']).astype(np.float)        
             
            #Create random MC realisations of the spectra using the estimated
            #rms noise.
            random_flux_draw = [
              [np.random.normal(flux,noise) for (flux,noise)
              in zip(flux_normalized,flux_rms)]
              for i in range(self.N_MC_runs)]
            
            wavelength = [wavelength_raw for i in range(self.N_MC_runs)]       
           
            #Create a mock dataframe with each of the MC realisations.
            mock_dict = pd.DataFrame(
              {'wavelength_raw': wavelength,
              'flux_normalized': random_flux_draw})
        
        except:
            mock_dict = {}        
    
        return mock_dict

    def compute_uncertainty(self,quantity,quantity_value,bin_size):
        def gaussian(x,A,mu,sigma):
            return A*np.exp(-(x-mu)**2./(2.*sigma**2.))

        if not np.isnan(quantity).all():
          
            flag = False
            
            quantity = np.asarray(quantity)
            quantity = quantity[~np.isnan(quantity)]     
            quantity = quantity[quantity != 0]       
            quantity_median = np.median(quantity)   
            quantity_mean = quantity.mean()

            bin_edges = np.arange(
              math.floor(min(quantity)), math.ceil(max(quantity)) + bin_size,
              bin_size)
           
            center_bins = np.asarray([edge + (edge_next - edge)/2.
              for edge, edge_next in zip(bin_edges,bin_edges[1:])])  
           
            pEW_histogram, edges = np.histogram(quantity,bins=bin_edges)
            
            try:                
                popt, pcov = curve_fit(
                  gaussian, center_bins, pEW_histogram, p0 = [
                  self.N_MC_runs / 6., quantity_median, abs(quantity_median / 5.)])
                
                gaussian_mean = popt[1]
                
                unc = abs(popt[2])
                #If uncertainty is smaller than bin_size, something is likely
                #wrong, so the object is flagged and the uncertainty becomes
                #the bin_size.
                if unc < bin_size:
                    unc = bin_size
                    flag = True  

                #If the measured feature is nan, then assess whether the
                #feature value is relatively close to the median of the MC
                #realisations. If the difference is larger than the estimated
                #uncertainty or the bin size, then flag this object --
                #meaning that the measured value might not be reliable.   
                if not np.isnan(quantity_value):
                    if abs(quantity_value - quantity_median) > max([unc,bin_size]): 
                        flag = True  
                
                #Conversely, if the originally measured feature was nan, but
                #the meadian of feature from the MC realisations is not nan,
                #then attribute the median value to the feature, but flag it
                #as unreliable. 
                elif np.isnan(quantity_value) and not np.isnan(quantity_median):    
                    flag = True
                    quantity_value = quantity_median
                        
            #If the code fails to fit a gaussian and estimate the uncertainty,
            #then flag the value as suspicius and attribute the value - mean
            #as an estimated uncertainty.
            except:
                flag = True
                unc = 2.*abs(quantity_value - quantity_mean)    

        #If every single feature value measured from the mock spectra in the MC
        #run is nan, then flag the object and attribute a nan uncertainty.
        else:
            unc, flag = np.nan, True  
        
        return unc, flag, quantity_value

    def run_uncertainties(self):
        """Main function to run the modules in this class to estimate the
        uncertainties.
        """
        
        #First, estimate the noise by computing the rms in a wavelength window.
        self.compute_flux_rms()  
         
        for idx in self.df.index.values:
            
            print '  -PROCESSING OBJECT WITH INDEX: '+str(idx)      
            
            #Second, cosntruct a dataframe of mock spectra with artificial noise.
            mock_spectra_dict = self.compute_mock_spectra(idx)

            #Check whether the mock spectra were successfully created.
            if any(mock_spectra_dict):
                
                mock_spectra_dict = Analyse_Spectra(
                  mock_spectra_dict, smoothing_mode='savgol', 
                  smoothing_window = self.smoothing_window,
                  deredshift_and_normalize=False, verbose=False).run_analysis() 
                               
                #Compute pEW uncertainty.
                for key in self.keys:
                    
                    pEW_unc, pEW_flag, pEW_value = self.compute_uncertainty(
                      mock_spectra_dict['pEW_f'+key].tolist(), 
                      self.df.loc[idx]['pEW_f'+key], 0.5)   
                                 
                    self.df.loc[idx, 'pEW_unc_f'+key]= pEW_unc
                    self.df.loc[idx, 'pEW_flag_f'+key] = pEW_flag
                    self.df.loc[idx, 'pEW_f'+key] = pEW_value
                
                #Compute velocity and depth uncertainties.
                for key in self.keys_to_fit:                
                    
                    v_unc, v_flag, v_value = self.compute_uncertainty(
                      mock_spectra_dict['velocity_f'+key].tolist(), 
                      self.df.loc[idx]['velocity_f'+key], 0.1)                
                       
                    self.df.loc[idx, 'velocity_unc_f'+key] = v_unc
                    self.df.loc[idx, 'velocity_flag_f'+key] = v_flag
                    self.df.loc[idx, 'velocity_f'+key] = v_value

                    d_unc, d_flag, d_value = self.compute_uncertainty(
                      mock_spectra_dict['depth_f'+key].tolist(),
                      self.df.loc[idx]['depth_f'+key], 0.01)             
                           
                    self.df.loc[idx, 'depth_unc_f'+key] = d_unc
                    self.df.loc[idx, 'depth_flag_f'+key] = d_flag
                    self.df.loc[idx, 'depth_f'+key] = d_value
            
            #If the mock spectra were not successfully created, then attribute
            #nan values to the uncertainties. This needs checking as might be
            #redundant to a similar check when computing the uncertainties.
            else:
                for key in self.keys:
                    self.df.loc[idx, 'pEW_unc_f'+key] = np.nan
                    self.df.loc[idx, 'pEW_flag_f'+key] = np.nan
                    self.df.loc[idx, 'pEW_f'+key] = np.nan
               
                for key in self.keys_to_fit:                
                    self.df.loc[idx, 'velocity_unc_f'+key] = np.nan
                    self.df.loc[idx, 'velocity_flag_f'+key] = np.nan
                    self.df.loc[idx, 'velocity_f'+key] = np.nan
                    
                    self.df.loc[idx, 'depth_unc_f'+key] = np.nan
                    self.df.loc[idx, 'depth_flag_f'+key] = np.nan
                    self.df.loc[idx, 'depth_f'+key] = np.nan
            
            #Cleaning up the mock spectra to make sure the memory is been freed.
            del mock_spectra_dict
                            
        print ("    -TOTAL TIME IN COMPUTING UNCERTAINTIES: "
               +str(format(time.time()-self.time, '.1f'))+'s')         
        print '    *** RUN COMPLETED SUCCESSFULLY ***\n'

        return self.df  

