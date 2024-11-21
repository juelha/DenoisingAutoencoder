import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.preprocessing import OneHotEncoder
import os

# utils 
from src import *
 
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(2, 10)
#         self.act = nn.ReLU()
#         self.output = nn.Linear(10, 2)
 
#     def forward(self, x):
#         x = self.act(self.hidden(x))
#         x = self.output(x)
#         return x
    

def main():
    x,y = gen_circle(1000)
    x_c, y_c = corruption_process(x,y)
    save_data("circle",x,y,x_c, y_c)

 
if __name__=="__main__":    

    main()


    shape_name = "circle"
    X_train, y_train, X_test,  y_test = load_data(shape_name)

    # loss metric and optimizer
    model = Autoencoder()

    
    hps = {
        'criterion': nn.MSELoss(),
        'learning_rate': 0.001,
        'optimizer': optim.Adam,
        'n_epochs': 75,
        'batch_size': 64
    }

    
    train = Trainer(model, hps)

    train.training_loop(X_train, y_train, X_test,  y_test )
    
    # Plot the loss and accuracy
    plot_acc(train, shape_name)
    plot_loss(train, shape_name)


    


    path = save_model(model, shape_name)

        # path = save_model(model)
    model = load_model("circle")

    #######################################
    # test_data 



    grid = gen_grid_data()

    prediction = model(grid)

    print("out", prediction)

    print("out", prediction[:,0])

    prediction = prediction.detach().numpy()

    u = grid[:,0] - prediction[:,0]
    v = grid[:,1] - prediction[:,1]
    fig, ax = plt.subplots()
    plt.quiver(grid[:,0],grid[:,1],-u,-v)

    plt.show()
