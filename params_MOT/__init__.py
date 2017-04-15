import numpy as np

def gaussian_1d(z, center_z, sigma_z, amplitude):
    return amplitude*np.exp(-(z-center_z)**2/(2*sigma_z**2))
	
def gaussian_2d(x, y, center_x, center_y, amplitude, sigma_x, sigma_y):
    return amplitude*np.exp(-(x-center_x)**2/(2*sigma_x**2))*np.exp(-(y-center_y)**2/(2*sigma_y**2))

def background(image_size,offset,scattered_light):
    N=image_size
    return np.add([[scattered_light for i in range(N)] for j in range(N)],offset)
	
def model(x, y, theta):
    center_x, center_y, amplitude, sigma_x, sigma_y, background_offset = theta
    return gaussian_2d(x, y, center_x, center_y, amplitude, sigma_x, sigma_y) + background_offset
	
def MOTmodel(x, y, center_x, center_y, amplitude, sigma_x, sigma_y):
    return gaussian_2d(x, y, center_x, center_y, amplitude, sigma_x, sigma_y)
	
def detected(model):
    return np.random.poisson(model)
	
def log_likelihood(x, y, z, sigma_m, sigma_g, mu_f, mu_sct, theta):
    '''
    theta: model parameters
    x, y: independent data (arrays of size 250)
    z: measurement (brightness of pixel) 
    sigma_m: uncertainty in the model chosen
    sigma_g: uncertainty from CCD camera readout noise
    mu_f: average number of photons due to shot noise
    mu_sct = average number of photons due to scattered light
    '''
    center_x, center_y, amplitude, sigma_x, sigma_y, background_offset = theta
    
    sigma = sigma_m + sigma_g
    prob_gaussian = gaussian_1d(z, model(x, y, theta), sigma, 1) 
    prob_poisson = np.random.poisson(mu_f) + np.random.poisson(mu_sct)
    
    return np.sum(np.log(prob_gaussian + prob_poisson)) 