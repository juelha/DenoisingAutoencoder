import matplotlib.pyplot as plt
from src.StorageManager import get_path

"""
This scripts contains functions for plotting datasets and vectors
"""

def plot_data(shape_name, x, y, x_corrupted, y_corrupted, save=True):
    """Plots the coordinates (x,y) and their corrupted versions in a scatter plot

    Args:
        shape_name (str): name of the shape
        x (nd.array): x-coordinates  
        y (nd.array): y-coordinates
        x_corrupted (nd.array): corrupted x-coordinates
        y_corrupted (nd.array): corrupted y-coordinates
        save (boolean, optional): whether to save the plot. Defaults to True.
    """
    plt.scatter(x_corrupted, y_corrupted, c='salmon', marker='.')
    plt.scatter(x, y, c='teal', marker='.')
    if save:
        path = get_path(f'{shape_name}.png', relative_path=f'/reports/{shape_name}/')
        plt.savefig(path)
    plt.show()
    plt.clf()


def plot_vectorfield(shape_name, grid, prediction, save=True):
    """Plots the vectors between the coordinates (x,y) and their corrupted versions in a vector field 

    Args:
        shape_name (str): name of the shape
        grid (torch.Tensor): x- and y-coordinates spanning a grid
        prediction (torch.Tensor): x- and y-coordinates from the trained model 
        save (boolean, optional): whether to save the plot. Defaults to True.
    """
    # transform to numpy
    prediction = prediction.detach().numpy()
    grid = grid.detach().numpy()
    
    x = grid[:,0]
    y = grid[:,1]
    x_pred = prediction[:,0]
    y_pred = prediction[:,1]

    plt.quiver(x, y, x_pred-x, y_pred-y)
    if save:
        path = get_path(f'{shape_name}_vectorfield.png', relative_path=f'/reports/{shape_name}/')
        plt.savefig(path)
    plt.show()
    plt.clf()


def plot_acc(trainer, shape_name, save=True):
    """Plotting the accuracy curve of the trainer

    Args:
        trainer (_type_): Instance of Trainer() 
        shape_name (str): name of the shape
        save (boolean, optional): whether to save the plot. Defaults to True.
    """
    plt.plot(trainer.train_accuracies, label="train")
    plt.plot(trainer.test_accuracies, label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    if save:
        path = get_path('AccuracyCurve.png', relative_path=f'/reports/{shape_name}/')
        plt.savefig(path)
    plt.show()
    plt.clf()


def plot_loss(trainer, shape_name, save=True):
    """Plotting the loss curve of the trainer

    Args:
        trainer (_type_): Instance of Trainer() 
        shape_name (str): name of the shape
        save (boolean, optional): whether to save the plot. Defaults to True.
    """
    plt.plot(trainer.train_losses, label="train")
    plt.plot(trainer.test_losses, label="test")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    if save:
        path = get_path('LossCurve.png', relative_path=f'/reports/{shape_name}/')
        plt.savefig(path)
    plt.show()
    plt.clf()
