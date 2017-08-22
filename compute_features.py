#!/usr/bin/env python

import os                                                               
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import scipy

from PyAstronomy import pyasl
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import savgol_filter
from astropy import constants as const

import astropy
import math
from math import factorial         

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

warnings.simplefilter('error', OptimizeWarning)

class Utility(object):
    """ Utility functions to assist on computing spectral features.
    Check separation.
    """
    
    def __init__(self, sep=20.):
        self.sep = sep
        self.keys = ['6', '7', 'C']
        self.c = const.c.to('km/s').value
        
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
        self.MD['blue_lower_f6'], self.MD['blue_upper_f6'] = 5400., 5750. #5700 originally
        self.MD['red_lower_f6'], self.MD['red_upper_f6'] = 5750., 6060. #6000. originally
     
        self.MD['rest_f7'] = [6355.21]
        self.MD['blue_lower_f7'], self.MD['blue_upper_f7'] = 5750., 6060.
        self.MD['red_lower_f7'], self.MD['red_upper_f7'] = 6200., 6600.
               
        self.MD['rest_f8'] = [7773.37]
        self.MD['blue_lower_f8'], self.MD['blue_upper_f8'] = 6800., 7450.
        self.MD['red_lower_f8'], self.MD['red_upper_f8'] = 7600., 8000.
        
        self.MD['rest_f9'] = [8498., 8542., 8662.]
        self.MD['blue_lower_f9'], self.MD['blue_upper_f9'] = 7500., 8100.
        self.MD['red_lower_f9'], self.MD['red_upper_f9'] = 8200., 8900.   
    
        #Below, the line boundaries are not really given the BSNIP paper IV;
        #For the blue side, using the same limits as the red side of f7 and
        #for the red side the regions was obtained by trial and error.
        self.MD['rest_fC'] = [6580.]
        self.MD['blue_lower_fC'], self.MD['blue_upper_fC'] = 6200., 6600. 
        self.MD['red_lower_fC'], self.MD['red_upper_fC'] = 6400., 6800.
            
class Analyse_Spectra(Utility):

    """ Computes a set of spectral features, given an input dictioanry
    containing ['wavelength_raw'] and ['flux_raw'].

    Parameters
    ----------
    wavelength : ~np.array
        Array containing the wavelength values of the spectra.

    flux : ~np.array
       Array containing the flux values of the spectra. Same length of the
       wavelength array.
                           
    redshift : float
        Redshift of the host galaxy. Usually the observed spectra is corrected
        by redshift and therefore syntethic spectra should use redshift=0.
    
    extinction : float
        Extinction to be corrected. Usually the observed spectra is not
        corrected for extinction and the syntethic spectra is reddened using
        a negative value for extinction.
    
    smoothing_window : ~float
        Window to be used by the Savitzky-Golay filter to smooth the spectra.
        Usually smoothing_window=51 for observed spectra and
        smoothing_window=21 for syntethic spectra.
        
    deredshift_and_normalize : boolean
        Flag to whether or not de-redshift the spectrum.     
                    
    Returns
    -------
    self.D : ~pandas dataframe
        Input dataframe where spectral features have been added as new columns. 
    """

    def __init__(self, wavelength, flux, redshift=0., extinction=0., D={},
                 smoothing_window=21, deredshift_and_normalize=True,
                 verbose=False):
        
        Utility.__init__(self)

        self.wavelength = wavelength
        self.flux = flux
        self.redshift = redshift
        self.extinction = extinction
        self.D = D

        self.smoothing_window = smoothing_window
        self.deredshift_and_normalize = deredshift_and_normalize
        self.verbose = verbose
                
    #@profile
    def perform_checks(self):
        
        def check_type(var, var_name, wanted_type):
            """Raise errors when variable don't have the required type."""
            if not isinstance(var, wanted_type):
                raise TypeError(
                  'Input ' + var_name + ' must be a ' + wanted_type.__name__
                  + ', not ' + type(var).__name__ + '.')              
        
        #Check whether variable types are as requeired.
        check_type(self.wavelength, 'wavelength_raw', np.ndarray)
        check_type(self.flux, 'flux_raw', np.ndarray)
        check_type(self.redshift, 'host_redshift', float)
        check_type(self.extinction, 'extinction', float)        
        
        #Once variables are checked to be ok, then store them in the dict.
        self.D['wavelength_raw'] = self.wavelength
        self.D['flux_raw'] = self.flux
        self.D['host_redshift'] = self.redshift
        self.D['extinction'] = self.extinction
                        
    #@profile
    def deredshift_spectrum(self):
        """Correct the wavelength for redshift. Note that the data downloaded
        from BSNIP is not in rest-wavelength."""   
        self.D['wavelength_corr'] = (self.D['wavelength_raw']
                                     / (1. + self.D['host_redshift']))                                                         

    def normalize_flux_and_correct_extinction(self):
        """ Normalize the flux to relative units so that methods such as
        wavelet smoothing can be applied if requested. Note that this cannot
        be further optimized as pyasl seems not work with series.
        """ 
        
        #@profile
        def get_normalized_flux(w, f, e):          
            #Redden/Un-redden the spectra.
            aux_flux = pyasl.unred(w, f, ebv=e, R_V=3.1)
                        
            #Wavelength window where the mean flux is computed.
            window_condition = ((w >= 4000.) & (w <= 9000.))             
            flux_window = aux_flux[window_condition]           
            normalization_factor = np.mean(flux_window)     
            aux_flux /= normalization_factor
            return aux_flux
       
        self.D['flux_normalized'] = get_normalized_flux(
          self.D['wavelength_corr'], self.D['flux_raw'], self.D['extinction'])   

    #@profile
    def smooth_spectrum(self):
        """Smooth the spectrum using either the savgol-golay.
        Other options not implemented at the moment.
        """                                 
        
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
        
        #Smooth flux
        self.D['flux_smoothed'] = savitzky_golay(
          self.D['flux_normalized'], self.smoothing_window, 3)

        #Smooth the derivative of the smoothed flux.
        def smooth_derivative(wavelength, f_smoothed):
            dw = np.diff(wavelength)
            df = np.diff(f_smoothed)
            der = savitzky_golay(np.divide(df, dw), self.smoothing_window, 3)         
            return np.append(np.array([np.nan]), der)
        
        self.D['derivative'] = smooth_derivative(self.D['wavelength_corr'],
                                                 self.D['flux_smoothed'])
        
        #This comment chunck perfomers the exact same calculation, but
        #is 20% slower. However it does not require reproducing the scipy code.
        '''
        self.D['flux_smoothed'] = savgol_filter(
          self.D['flux_normalized'], self.smoothing_window, 3)

        def smooth_derivative(wavelength, f_smoothed):
            dw = np.diff(wavelength)
            df = np.diff(f_smoothed)
            der = np.append(np.array([np.nan]), savgol_filter(
              np.divide(df, dw), self.smoothing_window, 3))
            return der                
        
        self.D['derivative'] = smooth_derivative(self.D['wavelength_corr'],
                                                 self.D['flux_smoothed'])        
        '''
    #@profile
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
        def get_zeros(wavelength, flux, derivative, key):
            
            #Retrieve all maxima and minima that are within the feature range.
            window_condition = ((wavelength >= self.MD['blue_lower_f'+key])
                                & (wavelength <= self.MD['red_upper_f'+key]))          
                  
            w_window = wavelength[window_condition]
            f_window = flux[window_condition]
            der_window = derivative[window_condition]    
            
            #Find the points where the sign of the derivative changes.
            #These are used as the conditions to determine maxima and
            #minima candidates. Note that the minima are forced to be to the
            #left of the rest wavelength of the feature, as to ensure
            #a negative velocity due to the Doppler effect -- to check
            #whether this messes up flagging shoulders afterwards.
            minima_cond = ((der_window[0:-1] < 0.) & (der_window[1:] > 0.)
                            & (w_window[0:-1] < max(self.MD['rest_f'+key]))) 
                            
            maxima_cond = ((der_window[0:-1] > 0.) & (der_window[1:] < 0.)) 
    
            #Condition array has len = len(w_window) - 1 as it uses consecutive
            #elements. Below it could be used w_window[1:], differences in the
            #computed quantities are not significant (usually < 1ang in pEW.)
            w_minima_window = w_window[0:-1][minima_cond]
            f_minima_window = f_window[0:-1][minima_cond]
            w_maxima_window = w_window[0:-1][maxima_cond]            
            f_maxima_window = f_window[0:-1][maxima_cond]                        
                                    
            def guess_minimum(potential_w, potential_f):
                """ In low noise spectra, get minimum at wavelength where the
                line would have been shifted due to a typical ejecta
                velocity of ~ -11,000 km/s. Maybe need some improvement to also
                consider the deepest minimum.
                """
                if len(potential_w) <= 4:
                    rest_w = np.mean(self.MD['rest_f' + key])
                    typical_v = -11000.
                    
                    typical_w = (rest_w * np.sqrt(1. + typical_v / self.c) / 
                      np.sqrt(1. - typical_v / self.c))
                                          
                    w_diff = np.absolute(potential_w - typical_w)
                    w_guess = potential_w[w_diff.argmin()]
                    f_guess = potential_f[w_diff.argmin()]
                                        
                #In noisy spectra, get the deepest minimum.
                elif len(potential_w) > 4:
                    f_guess = min(potential_f) 
                    w_guess = potential_w[potential_f.argmin()]
                                        
                return w_guess, f_guess    
                      
            copy_w_minima_window = np.copy(w_minima_window)
            copy_f_minima_window = np.copy(f_minima_window)

            for i in range(len(w_minima_window)):
                if len(copy_w_minima_window) > 0:
                    
                    #Assign a minimum.
                    w_min, f_min = guess_minimum(copy_w_minima_window,
                                                 copy_f_minima_window)

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
            if not np.isnan(w_max_blue) and len(w_maxima_window_blue) > 1:   
                
                #Compute wavelength separation between minima to the maximum.
                d_minima_window_blue = w_minima_window_blue - w_max_blue
                                
                #For each minimum, compute the largest relative fluxe
                #in the window between current maximum and the minimum.
                #This will assess whether the spectra is flat in this region.
                r_minima_window_blue = []
                for w_mwb in w_minima_window_blue:
                    try:
                        condition = ((wavelength <= w_max_blue)
                                     & (wavelength >= w_mwb))
                        
                        r_max = max([abs(f_step - f_max_blue) / f_max_blue for 
                                    f_step in flux[condition]])
                        
                        r_minima_window_blue.append(r_max)
                    except:
                        r_minima_window_blue.append(np.nan)
                                        
                #ASelect only the minima which are bluer than the maximum
                #and within the separation window or within 0.5% of the maximum
                #flux. This avoids tricky situations where there ahppens to be
                #a shoulder from a neighbor feature at the same level.                 
                d_minima_window_blue = np.asarray(
                  [d for (d, r) in zip(d_minima_window_blue, r_minima_window_blue)
                  if d < 0. and ((d > -1. * self.sep) or (r <= 0.01))])                
                                  
                #If there are shoulders, select the largest peak
                #that is bluer than the shoulder as the new maximum.
                if len(d_minima_window_blue) > 0:
                    condition = (w_maxima_window_blue <= w_max_blue)                  
                    w_maxima_window_blue = w_maxima_window_blue[condition]
                    f_maxima_window_blue = f_maxima_window_blue[condition]
                    if len(w_maxima_window_blue) >= 1:
                        f_max_blue = max(f_maxima_window_blue)
                        w_max_blue = w_maxima_window_blue[f_maxima_window_blue.argmax()]
            
            if not np.isnan(w_max_red) and len(w_maxima_window_red) > 1: 
                
                #Compute wavelength separation between minima to the maximum.
                d_minima_window_red = w_minima_window_red - w_max_red  

                #For each minimum, compute the largest relative fluxe
                #in the window between current maximum and the minimum.
                #This will assess whether the spectra is flat in this region.
                r_minima_window_red = []
                for w_mwr in w_minima_window_red:
                    try:
                        condition = ((wavelength >= w_max_red)
                                     & (wavelength <= w_mwr))
                        
                        r_max = max([abs(f_step - f_max_red) / f_max_red for 
                                    f_step in flux[condition]])
                        
                        r_minima_window_red.append(r_max)
                    except:
                        r_minima_window_red.append(np.nan)

               
                #Select only the minima which are bluer than the maximum
                #and within the separation window or within 0.5% of the maximum
                #flux. This avoids tricky situations where there ahppens to be
                #a shoulder from a neighbor feature at the same level. 
                d_minima_window_red = np.asarray(
                  [d for (d, r) in zip(d_minima_window_red, r_minima_window_red)
                  if d > 0. and ((d < 1. * self.sep) or (r <= 0.01))])
              
                #If there are shoulders, select the largest peak
                #that is redr than the shoulder as the new maximum.
                if len(d_minima_window_red) > 0:
                    condition = (w_maxima_window_red >= w_max_red)                  
                    w_maxima_window_red = w_maxima_window_red[condition]
                    f_maxima_window_red = f_maxima_window_red[condition]
                    if len(w_maxima_window_red) >= 1:
                        f_max_red = max(f_maxima_window_red)
                        w_max_red = w_maxima_window_red[f_maxima_window_red.argmax()]

            return float(w_min), float(f_min), float(w_max_blue), \
                   float(f_max_blue), float(w_max_red), float(f_max_red)

        for key in self.keys:
            v1, v2, v3, v4, v5, v6 = get_zeros(
              self.D['wavelength_corr'], self.D['flux_smoothed'],
              self.D['derivative'], key)
            
            self.D['wavelength_minima_f' + key] = v1
            self.D['flux_minima_f' + key] = v2
            self.D['wavelength_maxima_blue_f' + key] = v3
            self.D['flux_maxima_blue_f' + key] = v4
            self.D['wavelength_maxima_red_f' + key] = v5
            self.D['flux_maxima_red_f' + key] = v6            

    #@profile    
    def grab_feature_regions(self):
        """ Store the region of the features (boundaries determined at
        find_zeros_in_features) in order to facilitate computing features. 
        """
    
        def isolate_region(wavelength, flux_normalized, flux_smoothed,
                           blue_boundary, red_boundary):       
                                                                   
            if not np.isnan(blue_boundary) and not np.isnan(red_boundary): 
                
                region_condition = ((wavelength >= blue_boundary)
                                    & (wavelength <= red_boundary))      
                               
                wavelength_region = wavelength[region_condition]
                flux_normalized_region = flux_normalized[region_condition]
                flux_smoothed_region = flux_smoothed[region_condition]           
                
            else:
                wavelength_region = np.array([np.nan])
                flux_normalized_region = np.array([np.nan])
                flux_smoothed_region = np.array([np.nan])
           
            return wavelength_region, flux_normalized_region, \
                   flux_smoothed_region
        
        for key in self.keys:
                       
            c1, c2, c3 = isolate_region(
              self.D['wavelength_corr'], self.D['flux_normalized'],
              self.D['flux_smoothed'],
              self.D['wavelength_maxima_blue_f' + key],
              self.D['wavelength_maxima_red_f' + key])
            
            self.D['wavelength_region_f' + key] = c1
            self.D['flux_normalized_region_f' + key] = c2
            self.D['flux_smoothed_region_f' + key] = c3     
            
    #@profile
    def make_pseudo_continuum(self):
        """ The pseudo continuum slope is simply a line connecting the
        feature region boundaries. It depends only on the wavelength array and
        boundary values. The latter coming from smoothed quantities,
        regardless of method chosen (raw or smoothed.)
        """         
        def get_psedo_continuum_flux(w, x1, y1, x2, y2, f_smoothed):
           
            if len(f_smoothed) > 1:    
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
             
                def pseudo_cont(x):
                    return slope * x + intercept
              
                pseudo_flux = pseudo_cont(w)            

                #Check whether the continuum is always higher than the
                #**smoothed** flux and the array contains more than one element.
                boolean_check = (f_smoothed - pseudo_flux > 0.15
                                 * (max(f_smoothed) - min(f_smoothed)))
                            
                if True in boolean_check or len(boolean_check) < 1:
                    pseudo_flux = np.array([np.nan])

            else:
                pseudo_flux = np.array([np.nan])
            
            return pseudo_flux                                          
        
        for key in self.keys:
                        
            self.D['pseudo_cont_flux_f' + key] = get_psedo_continuum_flux(
              self.D['wavelength_region_f' + key],
              self.D['wavelength_maxima_blue_f' + key],
              self.D['flux_maxima_blue_f' + key],
              self.D['wavelength_maxima_red_f' + key],
              self.D['flux_maxima_red_f' + key],
              self.D['flux_smoothed_region_f' + key])  
                                      
    #@profile
    def compute_pEW(self):
        """ Compute the pEW of features.
        """
        def get_pEW(wavelength_region, flux_region, pseudo_flux):           
            if len(pseudo_flux) > 1:
                pEW = sum(np.multiply(
                  np.diff(wavelength_region),
                  np.divide(pseudo_flux[0:-1] - flux_region[0:-1], pseudo_flux[0:-1])))
            else:
                pEW = np.nan
            return pEW

        for key in self.keys:
            self.D['pEW_f' + key] = get_pEW(
              self.D['wavelength_region_f' + key], 
              self.D['flux_normalized_region_f' + key],
              self.D['pseudo_cont_flux_f' + key]) 
                    
    #@profile
    def compute_smoothed_velocity_and_depth(self):
        """ Compute the velocity of the features according to the rest
        wavelength of the line forming the feature.
        The velocity is computed by fitting a parabola to the minimum of the
        feature.
        """         
       
        def make_parabola(x_ref):
            def parabola(x, a, b, c):
                return a * (x - x_ref)**2. + b * (x - x_ref) + c
            return parabola  

        #@profile
        def get_smoothed_velocity(wavelength_region, flux_region,
                                  pseudo_flux, rest_wavelength):
                                      
            if len(pseudo_flux) > 1:                
                flux_at_min = min(flux_region)
                wavelength_at_min = wavelength_region[flux_region.argmin()]
                pseudo_cont_at_min = pseudo_flux[flux_region.argmin()] 
                
                wavelength_par = wavelength_region[
                  (wavelength_region >= wavelength_at_min - self.sep)
                  & (wavelength_region <= wavelength_at_min + self.sep)]
               
                flux_par = flux_region[
                  (wavelength_region >= wavelength_at_min - self.sep)
                  & (wavelength_region <= wavelength_at_min + self.sep)]
                
                #Note that using polyfit is significant faster than curve_fit.
                popt = np.polyfit(wavelength_par, flux_par, 2)                    
                rest_wavelength = np.mean(rest_wavelength)
                wavelength_par_min = - popt[1] / (2 * popt[0])
                flux_par_min = np.polyval(popt, wavelength_par_min)
                
                #Velocity is given in units of [1000 km/s].
                velocity = (self.c / 1.e3
                  * ((wavelength_par_min / rest_wavelength)**2. - 1.)
                  / ((wavelength_par_min / rest_wavelength)**2. + 1.))
                
                depth = 1. - flux_par_min / pseudo_cont_at_min
                                
                if popt[0] < 0. or velocity > 0. or velocity < -30000.:         
                    velocity = np.nan                    
                    
            else:                 
                wavelength_par_min, flux_par_min = np.nan, np.nan
                velocity, depth = np.nan, np.nan            
            
            return wavelength_par_min, flux_par_min, velocity, depth    
    
        for key in self.keys:
            
            a1, a2, a3, a4 = get_smoothed_velocity(
              self.D['wavelength_region_f' + key],
              self.D['flux_normalized_region_f' + key],
              self.D['pseudo_cont_flux_f' + key],
              self.MD['rest_f' + key])
              
            self.D['wavelength_at_min_f' + key] = a1
            self.D['flux_at_min_f' + key] = a2
            self.D['velocity_f' + key] = a3
            self.D['depth_f' + key] = a4
                        
    #@profile
    def run_analysis(self):
        """Main routine to call the functions of this class."""
        
        #'if' condition is useful when producing mock spectra to compute the
        #uncertainty -- it prevents repeating the calculation to normalize and
        #de-redshift the spectrum.
        self.perform_checks()
        if self.deredshift_and_normalize:
            self.deredshift_spectrum()
            self.normalize_flux_and_correct_extinction()    
        else:
            self.D['wavelength_corr'] = self.D['wavelength_raw'] 
            self.D['flux_normalized'] = self.D['flux_raw'] 
        self.smooth_spectrum()  
        self.find_zeros_in_features()
        self.grab_feature_regions()
        self.make_pseudo_continuum()
        self.compute_pEW()
        self.compute_smoothed_velocity_and_depth()
        
        return self.D  

class Compute_Uncertainty(Utility):
    """Uses a MC approach to compute the uncertainty of spectral features.
    As a guideline, this follows Liu+ 2016
    [[http://adsabs.harvard.edu/abs/2016ApJ...827...90L]].

    Parameters
    ----------
    dataframe : ~pandas dataframe
        Each row of the dataframe corresponds to a spectrum and thus needs to
        contain 'wavelength_corr' and 'flux_raw' columns.

    smoothing_mode : ~str
       'savgol' will use the Savitzky-Golay filter and is the default option.
       Other filters are not implemented at the moment.
                           
    smoothing_window : ~float
        Window to be used by the Savitzky-Golay filter to smooth the spectra.
        
    N_MC_runs : ~float
        Number of spectra with noise artificially added for the MC run.   

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
    self.D : ~pandas dataframe
        Input dataframe where spectral features have been added as new columns. 
    """

    def __init__(self, D, smoothing_window=21, N_MC_runs=3000):
                                
        Utility.__init__(self)
            
        self.D = D
        self.smoothing_window = smoothing_window
        self.N_MC_runs = N_MC_runs

        #Relatively small correction needed due to the fact that the smoothed
        #spectra 'follows' the noise, leading to a smaller than expected rms noise.
        if smoothing_window == 21:
            self._corr = 1. / 0.93
        elif smoothing_window == 51:
            self._corr = 1. / 0.96   
        else:
            raise ValueError('Smoothing correction not defined for this'
                             + 'smoothing window.')

    #@profile
    def compute_flux_rms(self, wave, fnor, fsmo):
        """ Estimate the flux noise in each pixel using a simple rms
        in a bin defined by the self.sep parameter.
        """                                 
    
        def rms(y_data, y_smot):
            """Given a noisy and a smoothed data, compute an array of the
            squared differences and take the square-root of its mean.
            Used as a proxy of the noise.
            """
            rms_out = np.sqrt(((y_data - y_smot)**2.).mean())
            if rms_out < 1.e-10: rms_out = 1.e-5     
            return rms_out

        #Compute the rms as a proxy of the noise of the flux point-wise.
        #Although point-wise, the noise of a given point is determined by
        #computing the rms including also the nearby points -- this prevents
        #funky values from being generated. In the loop below, for each point
        #'w' in the wavelength array, created a mini array containing the
        #nearby normalized and smoothed fluxes, which are then used as inputs
        #to the rms function.
        rms = np.asarray([rms(
          fnor[(wave >= w - self.sep) & (wave <= w + self.sep)],
          fsmo[(wave >= w - self.sep) & (wave <= w + self.sep)])
          * self._corr for w in wave])
      
        return rms
        
    #@profile
    def compute_uncertainty(self, quantity, quantity_value, bin_size):
        def gaussian(x,A,mu,sigma):
            return A*np.exp(-(x-mu)**2./(2.*sigma**2.))

        if not np.isnan(quantity).all():
            flag = False
            
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
                #If uncertainty is smaller than bin_size, the object is not 
                #not flagged, but the uncertainty becomes the bin_size.
                if unc < bin_size:
                    unc = bin_size

                #If the measured feature is nan, then assess whether the
                #feature value is relatively close to the median of the MC
                #realisations. If the difference is larger than the estimated
                #uncertainty or the bin size, then flag this object --
                #meaning that the measured value might not be reliable.   

                if not np.isnan(quantity_value):
                    if abs(quantity_value - quantity_median) > unc:
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
                unc = 2. * abs(quantity_value - quantity_mean)    

        #If every single feature value measured from the mock spectra in the MC
        #run is nan, then flag the object and attribute a nan uncertainty.
        else:
            unc, flag = np.nan, True  
        
        return unc, flag, quantity_value

    #@profile
    def run_uncertainties(self):
        """Main function to run the modules in this class to estimate the
        uncertainties.
        """
        
        #Estimate the noise by computing the rms in a wavelength window.
        self.D['flux_rms'] = self.compute_flux_rms(self.D['wavelength_corr'],
                                                   self.D['flux_normalized'], 
                                                   self.D['flux_smoothed']) 
                                 
        #Initialize dictionary to store compute quantities (such as pEW)
        #for the mock runs
        _store_D = {}
        for key in self.keys: 
            _store_D['pEW_f' + key] = []
            _store_D['velocity_f' + key] = []
            _store_D['depth_f' + key] = []
        
        #Compute quantities using mock spectra and store their values.
        for i in range(self.N_MC_runs):
            mock_flux = np.random.normal(self.D['flux_normalized'],
                                         self.D['flux_rms'])
            
            mock_D = {}
            mock_D = Analyse_Spectra(
              wavelength=self.D['wavelength_corr'], flux=mock_flux,
              redshift=self.D['host_redshift'], extinction=self.D['extinction'],
              D={}, smoothing_window=self.smoothing_window,
              deredshift_and_normalize=False, verbose=False).run_analysis()     
        
            for key in self.keys:                
                _store_D['pEW_f' + key].append(mock_D['pEW_f' + key])
                _store_D['velocity_f' + key].append(mock_D['velocity_f' + key])
                _store_D['depth_f' + key].append(mock_D['depth_f' + key])
        
        #Compute uncertainties.
        for key in self.keys:
            for var, bin_size in zip(['pEW', 'velocity', 'depth'],
                                     [0.5, 0.1, 0.01]):
                                         
                unc, flag, updated_value = self.compute_uncertainty(
                  np.asarray(_store_D[var + '_f' + key]),
                  self.D[var + '_f' + key], bin_size)
                
                self.D[var + '_unc_f' + key] = unc          
                self.D[var + '_flag_f' + key] = flag          
                self.D[var + '_f' + key] = updated_value          

        return self.D  

class Plot_Spectra(Utility):
    """
    """
    
    def __init__(self, D, outfile=False, show_fig=False, save_fig=False):

        Utility.__init__(self)

        self.D = D
        self.outfile = outfile
        self.show_fig = show_fig
        self.save_fig = save_fig
       
        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14       
        
        self.make_plots()

    def set_fig_frame(self, ax):
        x_label = r'$\lambda \ \mathrm{[\AA}]}$'
        y_label = r'$\mathrm{f}_{\lambda}/ \langle \mathrm{f}_{\lambda} \rangle$'
        ax.set_xlabel(x_label, fontsize=self.fs_label)
        ax.set_ylabel(y_label, fontsize=self.fs_label)
        ax.set_xlim(1500.,10000.)
        ax.set_ylim(0.,5.)
        ax.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)
        ax.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        ax.minorticks_on()
        ax.tick_params('both', length=8, width=1, which='major')
        ax.tick_params('both', length=4, width=1, which='minor')
        ax.xaxis.set_minor_locator(MultipleLocator(500.))
        ax.xaxis.set_major_locator(MultipleLocator(1000.))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))

    def add_feature_shade(self, ax, w, f, f_c, color, alpha):
        try:
            ax.plot(w, f_c, ls='--', c=color, alpha=alpha)
            ax.fill_between(w, f, f_c, color=color, alpha=alpha)
        except:
            pass    

    def add_boundaries(self, ax, w_max_blue, f_max_blue, w_max_red,
                       f_max_red, w_min, f_min, color):
        
        ax.plot(w_max_blue, f_max_blue, color=color, marker='+', markersize=12.)
        ax.plot(w_max_red, f_max_red, color=color, marker='+', markersize=12.)
        ax.plot(w_min, f_min, color=color, marker='x', markersize=12.)

    def save_figure(self, extension='png', dpi=360):
        if self.save_fig:
            extension = self.outfile.split('.')[-1]
            plt.savefig(self.outfile, format=extension, dpi=dpi)                                   
    
    def show_figure(self):
        if self.show_fig:
            plt.show()        
            
    def make_plots(self):
        
        colors = ['b', 'r', 'g']
        alpha = 0.5
                          
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)

        self.set_fig_frame(ax)

        ax.plot(self.D['wavelength_corr'], self.D['flux_normalized'],
                color='k', alpha=alpha, lw=1.)
        
        ax.plot(self.D['wavelength_corr'], self.D['flux_smoothed'],
                color='k', alpha=1., lw=2.)

        for i, key in enumerate(self.keys):
        
            self.add_feature_shade(ax, self.D['wavelength_region_f' + key],
                                   self.D['flux_normalized_region_f' + key],
                                   self.D['pseudo_cont_flux_f' + key],
                                   color=colors[i], alpha=alpha)            
        
                                 
            self.add_feature_shade(ax, self.D['wavelength_region_f' + key],
                                   self.D['flux_normalized_region_f' + key],
                                   self.D['pseudo_cont_flux_f' + key],
                                   color=colors[i], alpha=alpha)                                                                                         
            
            self.add_boundaries(ax, self.D['wavelength_maxima_blue_f' + key], 
                                self.D['flux_maxima_blue_f' + key],
                                self.D['wavelength_maxima_red_f' + key], 
                                self.D['flux_maxima_red_f' + key],
                                self.D['wavelength_minima_f' + key], 
                                self.D['flux_minima_f' + key], color=colors[i])

        
        ax.grid(True)
        plt.tight_layout()
        self.save_figure()
        self.show_figure()
        
        plt.close(fig)
