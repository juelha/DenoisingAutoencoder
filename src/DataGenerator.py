import numpy as np
import torch

"""
This scripts contains functions for generating datasets
"""

def gen_spiral(n_samples):
    """Generate (x,y) coordinates in the shape of a spiral

    Args:
        n_samples (int): number of coordinates to generate

    Returns:
        (x,y) (tuple(nd.array,nd.array)): tuple of x- and y-coordinates
    """
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    r = theta**2
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    # normalize x and y to [0, 1]
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    return x, y


def gen_line(n_samples):
    """Generate (x,y) coordinates in the shape of a line (x=y)

    Args:
        n_samples (int): number of coordinates to generate

    Returns:
        (x,y) (tuple(nd.array,nd.array)): tuple of x- and y-coordinates
    """
    x = np.random.uniform(-1, 1, n_samples)
    return x, x


def gen_circle(n_samples):
    """Generate (x,y) coordinates in the shape of a circle

    Args:
        n_samples (int): number of coordinates to generate

    Returns:
        (x,y) (tuple(nd.array,nd.array)): tuple of x- and y-coordinates
    """
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    x = np.cos(angles)
    y = np.sin(angles)
    return x, y


def corruption_process(x, y):
    """Corrupt the data given as the coordinates (x,y) with gaussian noise

    Args:
        x (nd.array): x-coordinates  
        y (nd.array): y-coordinates

    Returns:
        (x_corrupted, y_corrupted) (tuple(nd.array,nd.array)): tuple of corrupted x- and y-coordinates
    """
    # gen noise with same size as that of the data
    mu = 0
    sigma = 0.1
    noise_x = np.random.normal(mu, sigma, len(x)) 
    noise_y = np.random.normal(mu, sigma, len(y)) 

    # add the noise to the data (element-wise)
    x_corrupted = x + noise_y 
    y_corrupted = y + noise_x 
    
    return x_corrupted, y_corrupted


def gen_grid_data(n_samples=25, bot_left=(-2,-2), top_right=(2,2)):
    """Generate (x,y) coordinates spanning a grid from bot_left to top_right

    Args:
        n_samples (int): number of coordinates to generate
        bot_left (tuple(int,int), optional): corner coordinate of the grid. Defaults to (-1,-1).
        top_right (tuple(int,int), optional): corner coordinate of the grid. Defaults to (2,2).

    Returns:
        grid_data (torch.tensor): grid data prepared for the autoencoder model
    """
    x = np.linspace (bot_left[0],top_right[0],n_samples)
    y = np.linspace(bot_left[1],top_right[1],n_samples) 
    x,y = np.meshgrid(x,y)

    x = x.flatten()
    y = y.flatten()
    grid_data = np.stack((x,y), axis=1)
    grid_data = torch.tensor(grid_data, dtype=torch.float32)
    return grid_data






