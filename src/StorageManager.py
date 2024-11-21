import numpy as np
import os 
import yaml 
from yaml.loader import UnsafeLoader, SafeLoader, Loader
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import src.Autoencoder

"""
This scripts contains functions for loading and saving actions
"""

def get_path(file_name, relative_path, notes=None):
    """Get full path for loading

    Args:
        file_name (str): name of file
        relative_path (str): path taken from file to source of data
        notes (str, optional): Defaults to None.

    Returns:
        full_path (str): full path for loading directly
    """
    save_path = os.getcwd() + relative_path # https://stackoverflow.com/questions/39125532/file-does-not-exist-in-jupyter-notebook
    #save_path = os.path.dirname(__file__) +  relative_path
    if not notes == None:
        file_name += "_" + notes
    full_path = os.path.join(save_path, file_name)
    return full_path


def save_data(shape_name, x, y, x_corrupted, y_corrupted):
    """Save the coordinates (x,y) and their corrupted versions in a .csv file 

    Args:
        shape_name (str): name of the shape
        x (nd.array): x-coordinates  
        y (nd.array): y-coordinates
        x_corrupted (nd.array): corrupted x-coordinates
        y_corrupted (nd.array): corrupted y-coordinates
    """
    # saving as csv
    file_name = f"{shape_name}.csv"
    path = get_path(file_name, relative_path='/data/')
    data_dict = {'x_corrupted': x_corrupted, 'y_corrupted': y_corrupted, 'x': x, 'y': y}
    df = pd.DataFrame(data_dict)
    df.to_csv(path, index=False)


def load_data(shape_name):
    """Load the train and test datasets for a given shape

    Args:
        shape_name (str): name of shape

    Returns:
        X_train, y_train, X_test, y_test (tuple of pytorch tensors): datasets for training and testing
    """

    path = get_path(f'{shape_name}.csv', relative_path='/data/')
    data = pd.read_csv(path)
    data = data.to_numpy()
    data = np.hsplit(data, [2, 6])
    x,y = data[0], data[1]

    # convert into PyTorch tensors
    X = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32) 

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)
    return X_train, y_train, X_test, y_test


def save_model(model, shape_name):
    path = get_path(f'./{shape_name}.pth', relative_path=f'/models/')
    torch.save(model.state_dict(), path)

def load_model(shape_name):
    path = get_path(f'./{shape_name}.pth', relative_path=f'/models/')
    model = Autoencoder()
    model.load_state_dict(torch.load(path, weights_only=True))
   # net.to(device) # "send to gpu"
    return model



def save_hps(hps, df_name, trainer_name):
    """"""
    file_name = f"hps_{trainer_name}.yml"
    relative_path = f"/config/{df_name}/"
    path = get_path(file_name, relative_path)

    with open(path, 'w') as yaml_file:
        yaml.dump(hps, yaml_file, default_flow_style=False)

def load_hps(df_name, trainer_name):
    file_name = f"hps_{trainer_name}.yml"
    relative_path = f"/config/{df_name}/"
    path = get_path(file_name, relative_path)

    assert os.path.exists(path), f'File {file_name} not found'
    with open(path, 'r') as config_file:
    # Converts yaml document to np array object
        params = yaml.load(config_file, Loader=UnsafeLoader)#get_loader())#UnsafeLoader)
        return params
    

# def trainer_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Trainer:
#   """Construct a trainer."""
#   return Trainer(**loader.construct_mapping(node))

# def get_loader():
#   """Add constructors to PyYAML loader."""
#   loader = yaml.SafeLoader
#   loader.add_constructor("!Trainer", trainer_constructor)
#   return loader

    