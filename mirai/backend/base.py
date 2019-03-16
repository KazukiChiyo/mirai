import torch
from tqdm import tqdm

class Train(object):
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.
    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.
    """
    def __init__(self, model, data_loader, optim, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.device = device

    def run_epoch(self):
        """Runs an epoch of training.
        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.
        Returns:
        - The epoch loss (float).
        """
        self.model.train()
        epoch_loss = 0.0
        for step, batch_data in enumerate(tqdm(self.data_loader), 1):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # Forward propagation
            outputs = self.model(inputs)

            # Loss computation
            loss = self.criterion(outputs, labels)

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

        return epoch_loss/len(self.data_loader)


class Test(object):
    """Tests the ``model`` on the specified test dataset using the
    data loader, and loss criterion.
    Keyword arguments:
    - model (``nn.Module``): the model instance to test.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.
    """
    def __init__(self, model, data_loader, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device

    def run_epoch(self):
        """Runs an epoch of validation.
        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.
        Returns:
        - The epoch loss (float), and the values of the specified metrics
        """
        self.model.eval()
        epoch_loss = 0.0
        for step, batch_data in enumerate(tqdm(self.data_loader), 1):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            with torch.no_grad():
                # Forward propagation
                outputs = self.model(inputs)

                # Loss computation
                loss = self.criterion(outputs, labels)

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

        return epoch_loss/len(self.data_loader)
