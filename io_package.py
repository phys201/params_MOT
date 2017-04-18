import os
from numpy import loadtxt
from params_MOT import MOT_image

def get_data_file_path(filename = 'model_data.csv', data_dir=''):
    # __file__ is the location of the source file currently in use (so
    # in this case io.py). We can use it as base path to construct
    # other paths from that should end up correct on other machines or
    # when the package is installed
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    # If you need to go up another directory (for example if you have
    # this function in your tests directory and your data is in the
    # package directory one level up) you can use
    # up_dir = os.path.split(start_dir)[0]
    data_dir = os.path.join(start_dir, data_dir)
    return os.path.join(start_dir, data_dir, filename)

def load_data(data_file, delim = ' '):
    return loadtxt(data_file, delimiter = delim)

def load_image(data, image_size = 50):
    image_data = data.reshape(image_size, image_size)
    return MOT_image(image_data)
