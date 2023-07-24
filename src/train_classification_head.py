import torch
from torch.utils.data import DataLoader
from utils.data_loaders import ScanObjectNNDataset
from model.grnet import GRNet
from torch.utils.tensorboard import SummaryWriter

def train(model, train_loader, val_loader, device, config):
    loss_criterion = torch.nn.CrossEntropyLoss()
    loss_criterion.to(device)

    train_parameters = []
    train_parameters += list(model.fc11.parameters())
    train_parameters += list(model.fc12.parameters())
    train_parameters += list(model.fc13.parameters())
    train_parameters += list(model.fc14.parameters())
    train_parameters += list(model.fc15.parameters())