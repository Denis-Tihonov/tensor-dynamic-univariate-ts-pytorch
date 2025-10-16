import numpy as np
import pandas as pd
from scipy.linalg import hankel

def delay_embedding_matrix(s, nlags, fit_intercept=False):
    """Make a matrix with delay embeddings.

    Parameters
    ----------
    s : np.array
        The time series data.

    nlags : int
        Size of time lags.

    Returns
    -------
    delay_embedding_matrix : np.array of shape  (len(s) - lags + 1 , lags)
        Matrix with lags.
    """ 
    N = len(s)
    delay_embedding_matrix = hankel(s[ : N - nlags + 1], s[N - nlags : N])
    if fit_intercept:
        delay_embedding_matrix = np.hstack((np.ones((delay_embedding_matrix.shape[0],1)),delay_embedding_matrix))
    return delay_embedding_matrix

def diag_mean(array):
    """Diagonal mean.

    Parameters
    ----------
    array : np.array
        The time series matrix data.


    Returns
    -------
    result_array : np.array
        Diagonal mean for time series.
    """ 
    array = array[::-1]
    result_array = [array.diagonal(j).mean() for j in range(-array.shape[0]+1, array.shape[1])]
    return result_array

def prepare_time_series(path, centred = True):
    """Diagonal mean.

    Parameters
    ----------
    array : np.array
        The time series matrix data.


    Returns
    -------
    result_array : np.array
        Diagonal mean for time series.
    """ 
    
    data = pd.read_csv(path, delimiter =';', decimal=',')
    
    time_series = data[['X_value', 'Y_value', 'Z_value']]
    time_series[['x axis', 'y axis', 'z axis']] = time_series[['X_value', 'Y_value', 'Z_value']]
    time_series = time_series[['x axis', 'y axis', 'z axis']]
    if centred:
        time_series = (time_series - np.mean(time_series, axis = 0))/np.std(time_series, axis = 0)
        
    time_points = (data['time'].values).astype(float).reshape([-1,])
    time_points = np.linspace(0,time_points[-1]-time_points[0],time_series.shape[0])

    return time_series,time_points

def lorenz(xyz, *, s=10, r=28, b= 2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])