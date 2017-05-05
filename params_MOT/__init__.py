import numpy as np
from numpy import loadtxt
import os
import math
import matplotlib.pyplot as plt
import emcee
import scipy.ndimage.filters as filters
from params_MOT import MOT_image
import pandas as pd
from scipy.optimize import curve_fit
from params_MOT.model import *

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
    center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, readout_charge = theta
	
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
    x, y	-- independent data (arrays of size 50 for the 50x50 pixel image size)
    data	-- measurements (brightness of pixel), consisting of 2D array of size image_size*image_size
    theta	-- model parameter array (center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g).
    '''
    center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, readout_charge = theta
    MOT_model = Image_with_CCD_readout_charge(MOT_bare_model(x, y, theta), readout_charge) # Model is the bare model plus some CCD noise added on.
	
    
    return np.sum(-0.5*(data - MOT_model)**2/(sigma_m**2) - 0.5*np.log(2*np.pi*(sigma_m**2)))
    
def log_prior(theta):
    """
    Function to return log of prior probability distribution.
    
    Keyword arguments:
    theta	-- model parameter array (center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g).
    """
    # unpack the model parameters
    center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, readout_charge = theta

    # Work with a generally uninformative prior:
    # This means simply impose boundaries that make physical sense (amplitude > 0 or readout_charge > 1)
    # or limited by the apparatus (in a sense also physical; for example the centers or sigmas should be within the confines of the
    # CCD camera, i.e., smaller than image_size, which is taken to be 50)

    # TO DO: have a set of parameters that describe the apparatus also gets passed along, like theta. This way we can set limits on
    # the priors in a more general way (for ex, use the image_size variable to define the limits of the centers or sigmas)

    # NOTE: In practice we observed occasional "bad" sigma fits if we actually choose the range [0, 50]. Go with [3, 40] instead.

    if center_x > 40 or center_x < 3: # Limited by CCD size
        return -math.inf
    if center_y > 40 or center_y < 3: # Limited by CCD size
        return -math.inf
    if amplitude > 1000 or amplitude < 0: # Limited CCD saturation limit
        return -math.inf
    if sigma_x > 40 or sigma_x < 3: # Limited by CCD size
        return -math.inf
    if sigma_y > 40 or sigma_y < 3: # Limited by CCD size
        return -math.inf
    if sigma_m > 1000 or sigma_m < -1000: # Limited CCD saturation limit
        return -math.inf
    if background_offset > 450 or background_offset < -500: # Limited CCD saturation limit
        return -math.inf
    if readout_charge > 2000 or readout_charge < 1: # Limited CCD saturation limit
        return -math.inf
    
    return 0
    
def log_posterior(theta, x, y, data):
    '''
    Function to return log of posterior probability distribution. From Bayes' theorem we obtain that the posterior probability is the product of the prior and likelihood, so for the log posterior we sum the log of the prior and log of the likelihood.
	
    Keyword arguments:
    x, y	-- independent data (arrays of size 50 for the 50x50 pixel image size)
    data	-- measurements (brightness of pixel), consisting of 2D array of size image_size*image_size
    theta	-- model parameter array (center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g).
    '''
    
    center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, readout_charge = theta
    
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
    x, y = np.meshgrid(x, y)

    ndim = ndim
    nwalkers = nwalkers
    nsteps = nsteps

    starting_positions = [initial_guess + 1e-1*np.random.randn(ndim) for i in range(nwalkers)]
    
    # set up the sampler object
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x, y, data))
    
    # run the sampler. We use iPython's %time directive to tell us 
    # how long it took (in a script, you would leave out "%time")
    sampler.run_mcmc(starting_positions, nsteps)
    
    return sampler

def func_quad(x, b, m):
    '''
    Function which defines a quadratic equation (without the linear term)

    Keyword arguments:
    m, b		-- constants.
    x		    -- variable.
    '''
    return m * x**2 + b

def print_results_quad(b, m, covarianceM):
    '''
    Function that prints a quadratic equation, including uncertainties, given its constants and covariance matrix

    Keyword arguments:
    m, b		-- constants.
    covarianceM -- covariance matrix.
    '''
    print ("The covariance matrix is \n",covarianceM)
    print("\n")
    print ("The fitted model, including uncertainties is (%0.4f +- %0.4f)x^2 + (%0.0f +- %0.0f)"
           %(m, np.sqrt(covarianceM[1][1]), b, np.sqrt(covarianceM[0][0])))
    print("\n")

def quad_fit(data, x_name = "time", y_name = "sigma_x", suppressMessages = True):
    '''
    Function that does the fitting to a quadratic function (without the linear part)

    Keyword arguments:
    data		-- panda data frame containing the relevant data
    x_name      -- the name of the column containing the x data
    y_name      -- the name of the column containing the y data
    suppressMessages		-- Boolean which indicates whether or not messages, including plots, should be output.
    '''
    x = data[x_name].as_matrix()
    y = data[y_name].as_matrix()
    sigma = data['sigma_' + y_name].as_matrix()

    popt, cov = curve_fit(func_quad, xdata = x, ydata = y, sigma = sigma, method='lm')

    b, m = popt
    if(not suppressMessages):
        print_results_quad(b, m, cov)

    return [b, m]

def find_MOT_temp (q, pixel_distance_ratio, time_conversion_ratio, max_power, suppressMessages):
    '''
    Function that returns the temperatures corresponding to each direction, as well as a total temperature.

    Keyword arguments:
    q		                -- array containing MOT_object and fitted sigma_x and sigma_y, as returned by find_params_MOT(s)
    pixel_distance_ratio    -- value giving the conversion ratio between pixel and physical distance
    time_conversion_ratio   -- value scaling time
    max_power               -- maximum power; Note that the number as given by the MOT_image power attribute is a fraction of this max_power
    suppressMessages		    -- Boolean which indicates whether or not messages, including plots, should be output.
    '''

    # Create a pandas data frame storing the relevant information
    dataMOT = pd.DataFrame(columns=['time', 'power', 'sigma_x', 'sigma_sigma_x', 'sigma_y', 'sigma_sigma_y'])

    for i in range(len(q)):
        dataMOT.loc[i] = [time_conversion_ratio * float(q[i][0].time), max_power/float(q[i][0].power),
                          pixel_distance_ratio * q[i][1]['sigma_x'][0.50], \
                          pixel_distance_ratio * np.abs(
                              q[i][1]['sigma_x'][0.50] - (q[i][1]['sigma_x'][0.16] + q[i][1]['sigma_x'][0.84]) / 2), \
                          pixel_distance_ratio * q[i][1]['sigma_y'][0.50], \
                          pixel_distance_ratio * np.abs(
                              q[i][1]['sigma_y'][0.50] - (q[i][1]['sigma_y'][0.16] + q[i][1]['sigma_y'][0.84]) / 2)]


    dataMOT['sigma_x_squared'] = dataMOT['sigma_x']**2
    dataMOT['sigma_sigma_x_squared'] = dataMOT['sigma_sigma_x']**2
    dataMOT['sigma_y_squared'] = dataMOT['sigma_y']**2
    dataMOT['sigma_sigma_y_squared'] = dataMOT['sigma_sigma_y']**2

    if(not suppressMessages):
        print(dataMOT)

    quad_fit_sigma_x = quad_fit(dataMOT, x_name = "time", y_name='sigma_x_squared', suppressMessages = suppressMessages)
    quad_fit_sigma_y = quad_fit(dataMOT, x_name="time", y_name='sigma_y_squared', suppressMessages=suppressMessages)

    # Output plots for the fits
    if(not suppressMessages):
        dataMOT.iloc[:].plot(x='time', y='sigma_x_squared', kind='scatter', yerr='sigma_sigma_x_squared', s=30)
        _ = plt.xlabel('time (s)')
        _ = plt.ylabel('sigma_x^2 (m^2)')
        _ = plt.ylim([0.0000001, 0.000015])
        _ = plt.xlim([0, 0.005])
        _ = plt.plot(np.linspace(0, 0.005, 10), quad_fit_sigma_x[1] * np.linspace(0, 0.005, 10) ** 2 + quad_fit_sigma_x[0])
        _ = plt.title("Quadratic fit to data for sigma_x")
        _ = plt.show()

        dataMOT.iloc[:].plot(x='time', y='sigma_y_squared', kind='scatter', yerr='sigma_sigma_y_squared', s=30)
        _ = plt.xlabel('time (s)')
        _ = plt.ylabel('sigma_y^2 (m^2)')
        _ = plt.ylim([0.0000001, 0.000015])
        _ = plt.xlim([0, 0.005])
        _ = plt.plot(np.linspace(0, 0.005, 10), quad_fit_sigma_y[1] * np.linspace(0, 0.005, 10) ** 2 + quad_fit_sigma_y[0])
        _ = plt.title("Quadratic fit to data for sigma_y")
        _ = plt.show()

    # Define constants
    m = 9.80 * 10 ** (-26)
    K_b = 1.38 * 10 ** (-23)

    # The temperatures:
    T_x = quad_fit_sigma_x[1]*m/K_b
    T_y = quad_fit_sigma_y[1]*m/K_b
    T = T_x**(2/3) * T_y**(1/3)

    if(not suppressMessages):
        print("The fitted temepratures: T_x = %f mK, T_y = %f mK, T = %f mK" %(10**3 * T_x, 10**3 * T_y, 10**3 * T))

    return [T_x, T_y, T]

def separate_files_power(data_files):
    '''
    Function that takes an array of data file names and groups them according to the power used.

    Keyword arguments:
        data_files  -- array of data file names

    '''
    data_files_power = [[0 for x in range(1)] for y in range(1000)] #1000 should be larger than any other power fraction we choose

    for f in data_files:
        try:
            f_power = int((f.split('_')[2]).split('power')[0])
        except IndexError:
            raise ValueError('Check that your data file name respects the convention.')

        if data_files_power[f_power] == [0]:
            data_files_power[f_power] = [f]
        else:
            data_files_power[f_power].append(f)

    return [list(row) for row in data_files_power if any(x is not 0 for x in row)]


def temp_vs_power(data_files, data_dir = 'data', image_size = 50, mc_params=(200, 1500, 500), initial_guess=[25, 25, 400, 6.6667, 5.5556, 100, 20, 20], suppressMessages=True):
    '''
        Function that takes an array of data file names and returns temperatures as a function of the laser power used.

        Keyword arguments:
            data_files          -- array of data file names
            data_dir	        -- String denoting name of data directory
		    image_size	        -- Size of MOT image (which is assumed to be a square).
		    mc_params	        -- triplet of MCMC parameters: (number of walkers, number of steps, burn_in_steps).
						        -- walkers = individual traces in the Monte Carlo algorithm
						        -- steps = length of said traces
						        -- burn_in_steps = steps after which the trace settles around a value
		    initial_guess		-- tuple of initial MCMC guesses, consisting of (center_x, center_y, amplitude, sigma_x, sigma_y, background_offset, sigma_m, sigma_g).
		    suppressMessages    -- Boolean which indicates whether or not messages, including plots, should be output.

    '''


    # TO DO: Again, have a set of parameters that describe the appartus (like theta) that we automatically pass around
    # this way max power doesn't need to be set inside the package (and should be variable)
    max_power = 60 * 10 ** (-3)
    dataPowerTemp = pd.DataFrame(columns=['power', 'T_x', 'T_y', 'T'])


    for i in range(len(data_files)):
        q = find_params_MOTs(data_files[i], data_dir, image_size, mc_params, initial_guess, suppressMessages = True)
        power = max_power / float(q[0][0].power)
        temp = find_MOT_temp(q, pixel_distance_ratio=0.4 * 10 ** (-3), time_conversion_ratio=10 ** (-3), max_power=60 * 10 ** (-3), suppressMessages=True)

        dataPowerTemp.loc[i] = [power, temp[0], temp[1], temp[2]]

    return dataPowerTemp





