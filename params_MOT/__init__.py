import numpy as np
from numpy import loadtxt
import os
import math
import matplotlib.pyplot as plt
import emcee
import scipy.ndimage.filters as filters

def gaussian_1d(z, center_z, sigma_z, amplitude):
    return amplitude*np.exp(-(z-center_z)**2/(2*sigma_z**2))
	
def gaussian_2d(x, y, center_x, center_y, amplitude, sigma_x, sigma_y):
    return amplitude*np.exp(-(x-center_x)**2/(2*sigma_x**2))*np.exp(-(y-center_y)**2/(2*sigma_y**2))

def Image_with_CCD_readout_charge(image, readout_charge):
    charge=(np.cumsum(image[::-1],axis=0)/readout_charge)
    return image + charge[::-1]

def MOT_bare_model(x, y, theta):
    center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g = theta
    return gaussian_2d(x, y, center_x, center_y, amplitude, sigma_x, sigma_y) + background_offset

def MOT_model(x, y, theta):
    # Use 40 for the readout_charge for now
    return Image_with_CCD_readout_charge(MOT_bare_model(x, y, theta), 40)

def background(image_size,offset,scattered_light):
    N=image_size
    return np.add([[scattered_light for i in range(N)] for j in range(N)],offset)

def detected(model):
    return np.random.poisson(model)
	
# Functions for Bayesian inference:
	
def log_likelihood(theta, x, y, data):
    '''
    x, y: independent data (arrays of size 50)
    data: measurements (brightness of pixel) 
    sigma_m: uncertainty in the model chosen
    sigma_g: uncertainty from scattered light background
    theta: model parameters
    '''
    center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g = theta
      
    #return -0.5*(np.sum((data[x-1][y-1] - MOT_model(x, y, theta))**2/(sigma_m**2 + sigma_g**2) + np.log(sigma_m**2 + sigma_g**2)))
    for i in x:
        for j in y:
            return -0.5*(np.sum((data[int(i)-1][int(j)-1] - MOT_model(x, y, theta))**2/(sigma_m**2 + sigma_g**2) - np.log(sigma_m**2 + sigma_g**2)))
    
def log_prior(theta):
    """
    returns log of prior probability distribution
    
    Parameters:
        theta: model parameters (specified as a tuple)
    """
    # unpack the model parameters
    center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g = theta
  
    # impose bounds on parameters
    # For now (the model data) impose strong bounds
    if center_x > 30 or center_x < 20:
        return -math.inf
    if center_y > 30 or center_y < 20:
        return -math.inf
    if amplitude > 450 or amplitude < 350:
        return -math.inf
    if sigma_x > 8 or sigma_x < 5:
        return -math.inf
    if sigma_y > 8 or sigma_y < 3:
        return -math.inf
    if sigma_m < 0: 
        return -math.inf
    if sigma_g < 0:
        return -math.inf
    if background_offset > 1000:
        return -math.inf
    
    sigma_sigma_m_Jeff_prior = 1/(sigma_m) 
    sigma_sigma_g_Jeff_prior = 1/(sigma_g)
 
    sigma_x_prior = 100/(50/7.5 - sigma_x)**2
    sigma_y_prior = 100/(50/9 - sigma_y)**2
    
    # Use a pretty strong prior on the sigma_x and sigma_y values (for now)
    if math.isnan(sigma_x_prior) or math.isnan(sigma_y_prior):
        return 0 + np.log(sigma_sigma_m_Jeff_prior) + np.log(sigma_sigma_g_Jeff_prior)
    else:
        return 0 + np.log(sigma_sigma_m_Jeff_prior) + np.log(sigma_sigma_g_Jeff_prior) + np.log(sigma_x_prior) + np.log(sigma_y_prior)
    
def log_posterior(theta, x, y, data):
    '''
    theta: model parameters
    x, y: independent data (arrays of size 250)
    z: measurement (brightness of pixel) 
    sigma_m: uncertainty in the model chosen
    sigma_g: uncertainty from scattered light background
    theta: model parameter
    '''
    
    center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g = theta
    
    return log_prior(theta) + log_likelihood(theta, x, y, data)

# For emcee

def sampler(data, ndim, nwalkers, nsteps, image_size):
    
    # Set up the data
    x = np.linspace(1, image_size, image_size)
    y = np.linspace(1, image_size, image_size)
    
    ndim = ndim
    nwalkers = nwalkers
    nsteps = nsteps

    #initial guess for center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g
    ls_result = [25, 25, 400, 6.6667, 5.5556, 100, 20, 20] # from HBL figure 1 and randomly guessing
    
    starting_positions = [ls_result + 1e-1*np.random.randn(ndim) for i in range(nwalkers)]
    
    # set up the sampler object
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x, y, data))
    
    # run the sampler. We use iPython's %time directive to tell us 
    # how long it took (in a script, you would leave out "%time")
    sampler.run_mcmc(starting_positions, nsteps)
    print('Done')
    
    return sampler
