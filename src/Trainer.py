import numpy as np
import torch
from tqdm import tqdm


class Trainer():

    def __init__(self, model=None, hps=None):
        """_summary_

        Args:
            n_inputs (_type_): _description_
            n_outputs (_type_): _description_
            optimizer_func (_type_, optional): _description_. Defaults to "Adam".
            learning_rate (_type_, optional): _description_. Defaults to None.
            n_epochs (_type_, optional): _description_. Defaults to None.

        Attributes:
            test_accuracies (list(float)): keeping track of test accuracies during training
            test_losses (list(float)): keeping track of test losses during training
            train_losses (list(float)): keeping track of train losses during training    
            train_accuracies (list(float)): keeping track of train accuracies during training    
        """

        self.model = model
        
        # # hyperparamers
        self.n_epochs = 40
        self.learning_rate =  1
        self.batch_size = 10
        self.criterion = None

        for param_name in hps.keys():
            setattr(self, param_name, hps[param_name])

        self.optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)

        # to track performance
        self.train_accuracies = []
        self.train_losses = []
        self.test_accuracies = []
        self.test_losses = []


    def training_loop(self, x_train, y_train, x_test, y_test):
        """Training loop for mini batch training and iterating over whole dataset in one epoch

        Args:
            x_train (torch.Tensor): of shape [n_train_samples, n_features]
            y_train (torch.Tensor): of shape [n_train_samples, n_outputs]
            x_test (torch.Tensor): of shape [n_test_samples, n_features]
            y_test (torch.Tensor): of shape [n_test_samples, n_outputs]
        """
        
        # iter over epochs
        for epoch in tqdm(range(self.n_epochs), desc="Training"):

            # reset temp aggregators
            self.epoch_loss = []
            self.epoch_acc = []

            # set model to training mode 
            self.model.train()

            # iter over all batches in dataset
            for i in range(len(x_train) // self.batch_size):
                # get batch
                start = i * self.batch_size
                x_batch = x_train[start:start+self.batch_size]
                y_batch = y_train[start:start+self.batch_size]
                self.train_step(x_batch, y_batch)

            self.test(x_test, y_test)
            self.train_losses.append(np.mean(self.epoch_loss))
            self.train_accuracies.append(np.mean(self.epoch_acc))

        print(f"Epoch {epoch}: Loss={self.test_losses[-1]:.2f}, Accuracy={self.test_accuracies[-1]*100:.1f}%")
        

    def train_step(self, x_batch, y_batch ):
        """Performing train step on one batch

        Args:
            x_batch (torch.Tensor): of shape [batch_size, n_features]
            y_batch (torch.Tensor): of shape [batch_size, n_outputs]
        """
        # forward pass
        y_pred = self.model(x_batch) 
        loss = self.criterion(y_pred, y_batch)

        # backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # track performance
        rounded_preds = torch.round(y_pred, decimals=1)
        rounded_tarts = torch.round(y_batch, decimals=1)
        acc = (rounded_preds == rounded_tarts).float().mean().item()
        self.epoch_loss.append(float(loss))
        self.epoch_acc.append(float(acc))

        
        
    def test(self, x_test, y_test):
        """_summary_

        Args:
            x_test (torch.Tensor): of shape [n_test_samples, n_features]
            y_test (torch.Tensor): of shape [n_test_samples, n_features]
        """
        # set model to evaluation mode 
        self.model.eval()
        
        # dont need to calc any gradients
        with torch.no_grad():

            # forward pass
            y_pred = self.model(x_test)

            # track performance
            rounded_preds = torch.round(y_pred, decimals=1)
            rounded_tarts = torch.round(y_test, decimals=1)
            acc = (rounded_preds == rounded_tarts).float().mean().item()
            loss = float(self.criterion(y_pred, y_test))
            self.test_losses.append(loss)
            self.test_accuracies.append(acc)


