import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters

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

    def __init__(self, data, time):
        self.data = data
        self.time = time

    def show(self, image_size=50, gauss_filter=False):

        image_size = image_size
        x = np.linspace(1, image_size, image_size)
        y = np.linspace(1, image_size, image_size)
        x, y = np.meshgrid(x, y)
        image = MOTmodel(x, y, image_size / 2, image_size / 2, 400, image_size / 7.5, image_size / 9)

        if gauss_filter == False:
            plt.figure(1)
            plt.imshow(image, cmap="jet", interpolation='none')
            plt.colorbar()
        else:
            plt.figure(1)
            plt.imshow(filters.gaussian_filter(image, 1), cmap="jet", interpolation='none')
            plt.colorbar()