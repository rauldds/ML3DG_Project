import torch
from torch.utils.data import DataLoader
from utils.data_loaders import ScanObjectNNDataset
from model.grnet import GRNet
from pathlib import Path
import utils.data_loaders


def train(model, train_dataloader, val_dataloader, device, config):
    loss_criterion = torch.nn.SmoothL1Loss()
    loss_criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.train()

    train_loss_running = 0.

    for epoch in range(config['max_epochs']):
        for batch_idx, batch in enumerate(train_dataloader):
            ScanObjectNNDataset.send_data_to_device(batch, device)

            optimizer.zero_grad()

            reconstruction = model(batch["incomplete_view"])
            target_sdf = batch["target_sdf"]

            loss =loss_criterion(reconstruction, target_sdf)
            loss.backward()
            optimizer.step()

            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx

            if iteration % config["print_every_n"] == (config["print_every_n"] - 1):
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss_running / config["print_every_n"]:.6f}')
                train_loss_running = 0.


def main(config):

    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')


    # Create Dataloaders
    train_dataset = ScanObjectNNDataset('train' if not config['is_overfit'] else 'overfit')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=14,   # Data is usually loaded in parallel by num_workers
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
    )

    model = GRNet()
    model.to(device)

    Path(f'exercise_3/runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    train(model=model,
          train_dataloader=train_dataloader,
          device=device,
          config=config,
          val_dataloader=None)