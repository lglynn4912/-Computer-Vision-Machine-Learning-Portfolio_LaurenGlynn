# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        self.input_shape = input_shape
        self.conv_1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)
        self.relu_1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu_1 = nn.ReLU()

        self.conv_2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.denselayer_1 = nn.Linear(16 * 5 * 5, 256)
        self.relu_3 = nn.ReLU()
        self.denselayer_2 = nn.Linear(256, 128)
        self.relu_4 = nn.ReLU()
        self.denselayer_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        shape_dict = {}
        x = self.maxpool_1(self.relu_1(self.conv_1(x)))
        shape_dict[1] = torch.tensor(x.size())

        x = self.maxpool_2(self.relu_2(self.conv_2(x)))
        shape_dict[2] = torch.tensor(x.size())

        x = self.flatten(x)
        shape_dict[3] = torch.tensor(x.size())

        x = self.relu_3(self.denselayer_1(x))
        shape_dict[4] = torch.tensor(x.size())

        x = self.relu_4(self.denselayer_2(x))
        shape_dict[5] = torch.tensor(x.size())

        x = self.denselayer_3(x)
        shape_dict[6] = torch.tensor(x.size())

        out = x

        return out, shape_dict


def count_model_params():
    """
    return the number of trainable parameters of LeNet.
    """
    model = LeNet()
    model_params = 0.0

    for name, params in model.named_parameters():
        if params.requires_grad:
            model_params += params.numel()

    return model_params / 1e6


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch + 1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch + 1, 100. * test_acc))

    return test_acc
