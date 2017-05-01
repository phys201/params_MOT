import os
from numpy import loadtxt
from params_MOT import MOT_image

def get_data_file_path(filename = 'model_data.csv', data_dir=''):
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    up_dir = os.path.split(start_dir)[0] # Go up one level for the data file path
    data_dir = os.path.join(up_dir, data_dir)
    return os.path.join(start_dir, data_dir, filename)

def load_data(data_file, delim = ','):
    return loadtxt(data_file, delimiter = delim)

def load_time(filename):
    return filename.split('_')[0]

def load_power(filename):
    return (filename.split('_')[2]).split('power')[0]

