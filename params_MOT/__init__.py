import numpy as np
from numpy import loadtxt
import os
import math
import matplotlib.pyplot as plt
import emcee
import scipy.ndimage.filters as filters

def gaussian_1d(z, center_z, sigma_z, amplitude):
    '''
    Gaussian function in 1D
    
	Returns a standard Gaussian function with a single peak.
    
    Keyword arguments:
    z			-- Input
	center_z	-- mean
	sigma_z		-- standard deviation
	amplitude	-- amplitude
    '''
    return amplitude*np.exp(-(z-center_z)**2/(2*sigma_z**2))

def gaussian_2d(x, y, center_x, center_y, amplitude, sigma_x, sigma_y):
    '''
    Gaussian function in 2D
    
	Returns a product of two gaussians, one in x and one in y. Uses the function gaussian_1d defined before.
    
    Keyword arguments:
    x,y					-- Inputs
	center_x, center_y	-- means
	sigma_x, sigma_y	-- standard deviation
	amplitude			-- amplitude
    '''
    return amplitude*gaussian_1d(x, center_x, sigma_x, 1)*gaussian_1d(y,center_y,sigma_y,1)

def Image_with_CCD_readout_charge(image, readout_charge):
    '''
	Function to add CCD readout image noise to an image of a captured MOT.
	This noise is accumulated vertically, which is why the array image is flipped over when calculating the variable charge.
	
	Keyword arguments:
	image			-- Input image, in the form of a 2D arrays
	readout_charge	-- parameter controling amount of noise
	'''
    charge=(np.cumsum(image[::-1],axis=0)/readout_charge)
    return image + charge[::-1]

def MOT_bare_model(x, y, theta):
    '''
	Function to unpack the parameter array theta and insert them into the Gaussian 2D model.
	This gives the "bare" or perfect model of the MOT without any noise.
	Also adds a background offset.
	
	Keyword arguments:
	theta	-- model parameter array (center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g).
    These are parameters for the function gaussian_2d (see above) plus background offset, which is a general offset added to the overall data.
	'''
    center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g = theta
	
    return gaussian_2d(x, y, center_x, center_y, amplitude, sigma_x, sigma_y) + background_offset

def background(image_size,offset,scattered_light):
    '''
	Function to generate background, which will be processed later using params_MOT.detected
	
	Keyword arguments:
	image_size		-- size of image data, usually 50*50 in our experiment
	offset			-- overall offset to be added 
	scattered_light	-- parameter characterizing magnitude of scattered light on CCD
	'''
    N=image_size
    return np.add([[scattered_light for i in range(N)] for j in range(N)],offset)
	
def detected(model):
    '''
    Function to add random Poissonian noise to the MOT model.

    Keyword arguments:
    model	-- image of MOT, consisting of a 2D array size 50*50
    '''
    return np.random.poisson(model)

# Functions for Bayesian inference:
	
def log_likelihood(theta, x, y, data):
    '''
    Function to define log-likelihood for Bayesian inference.
    
    Keyword arguments:
    x, y	-- independent data (arrays of size 50)
    data	-- measurements (brightness of pixel), consisting of 2D array of size 50*50
    theta	-- model parameter array (center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g).
    '''
    center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g = theta
    MOT_model = Image_with_CCD_readout_charge(MOT_bare_model(x, y, theta), 40) # Model is the bare model plus some CCD noise added on.
	
    #return -0.5*(np.sum((data[x-1][y-1] - MOT_model(x, y, theta))**2/(sigma_m**2 + sigma_g**2) + np.log(sigma_m**2 + sigma_g**2)))
    for i in x:
        for j in y:
            return -0.5*(np.sum((data[int(i)-1][int(j)-1] - MOT_model)**2/(sigma_m**2 + sigma_g**2) - np.log(sigma_m**2 + sigma_g**2)))
    
def log_prior(theta):
    """
    Function to return log of prior probability distribution.
    
    Keyword arguments:
    theta	-- model parameter array (center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g).
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
    Function to return log of posterior probability distribution. From Bayes' theorem we obtain that the posterior probability is the product of the prior and likelihood, so for the log posterior we sum the log of the prior and log of the likelihood.
	
    Keyword arguments:
    x, y	-- independent data (arrays of size 50)
    data	-- measurements (brightness of pixel), consisting of 2D array of size 50*50
    theta	-- model parameter array (center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g).
    '''
    
    center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g = theta
    
    return log_prior(theta) + log_likelihood(theta, x, y, data)

# For emcee

def sampler(data, ndim, nwalkers, nsteps, image_size, initial_guess):
    '''
    Function which runs the MCMC sampler for parameter search.
	
    Keyword arguments:
    data		-- measurements (brightness of pixel), consisting of 2D array of size 50*50
    ndim		-- number of dimensions in MCMC, corresponding to number of parameters being marginalized over.
    nwalkers	-- number of walkers in MCMC.
    nsteps		-- number of steps in MCMC
    image_size	-- size of image in data.
    init_guess	-- initial guess for optimal parameters, an array of length 8 consisting of (center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g). This is the same structure as variable theta in other functions in this package.
    '''
    # Set up the data
    x = np.linspace(1, image_size, image_size)
    y = np.linspace(1, image_size, image_size)
    
    ndim = ndim
    nwalkers = nwalkers
    nsteps = nsteps

    starting_positions = [initial_guess + 1e-1*np.random.randn(ndim) for i in range(nwalkers)]
    
    # set up the sampler object
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x, y, data))
    
    # run the sampler. We use iPython's %time directive to tell us 
    # how long it took (in a script, you would leave out "%time")
    sampler.run_mcmc(starting_positions, nsteps)
    print('Done')
    
    return sampler
