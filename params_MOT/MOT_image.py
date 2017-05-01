import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
from params_MOT import MOT_image

def gaussian_2d(x, y, center_x, center_y, amplitude, sigma_x, sigma_y):
    return amplitude * np.exp(-(x - center_x) ** 2 / (2 * sigma_x ** 2)) * np.exp(
        -(y - center_y) ** 2 / (2 * sigma_y ** 2))

def MOTmodel(x, y, center_x, center_y, amplitude, sigma_x, sigma_y):
    return gaussian_2d(x, y, center_x, center_y, amplitude, sigma_x, sigma_y)

class MOT_image:
    '''
    data: numpy 2D array containing the photon count for the (x,y) pixel
    time: duration since MOT has been released from trap
    '''

    #def __init__(self, data, time):
    def __init__(self, data, time, power, image_size = 50):
        data.reshape(image_size, image_size)
        self.data = data
        self.time = time
        self.power = power

    def show(self, image_size=50, gauss_filter=False):

        if gauss_filter == False:
            plt.figure(1)
            plt.imshow(self.data, cmap="jet", interpolation='none')
            plt.colorbar()
        else:
            plt.figure(1)
            plt.imshow(filters.gaussian_filter(self.data, 1), cmap="jet", interpolation='none')
            plt.colorbar()
