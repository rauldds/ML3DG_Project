from math import gamma
import torch
from torch.utils.data import DataLoader
from utils.data_loaders import ScanObjectNNDataset
from model.grnet import GRNet
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import utils.data_loaders
import os


def train(model, train_dataloader, val_dataloader, device, config):
    loss_criterion = torch.nn.SmoothL1Loss()
    loss_criterion.to(device)

    train_parameters = []
    train_parameters += list(model.gridding.parameters())
    train_parameters += list(model.conv1.parameters())
    train_parameters += list(model.conv2.parameters())
    train_parameters += list(model.conv3.parameters())
    train_parameters += list(model.conv4.parameters())
    train_parameters += list(model.fc5.parameters())
    train_parameters += list(model.fc6.parameters())
    train_parameters += list(model.dconv7.parameters())
    train_parameters += list(model.dconv8.parameters())
    train_parameters += list(model.dconv9.parameters())
    train_parameters += list(model.dconv10.parameters())
    train_parameters += list(model.gridding_rev.parameters())

    optimizer = torch.optim.Adam(train_parameters,
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'],
                                 betas=config["betas"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["milestones"],gamma=config["gamma"])

    model.train()

    train_loss_running = 0.

    tb = SummaryWriter()
    for epoch in range(config['max_epochs']):
        for batch_idx, batch in enumerate(train_dataloader):
            ScanObjectNNDataset.send_data_to_device(batch, device)

            optimizer.zero_grad()

            reconstruction, class_pred = model(batch["incomplete_view"])
            # TODO: understand the following mask:
            #  reconstruction[batch_val['input_sdf'][:, 1] == 1] = 0
            #  target[batch_val['input_sdf'][:, 1] == 1] = 0
            target_sdf = batch["target_sdf"]

            loss =loss_criterion(reconstruction, target_sdf)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx

            tb.add_scalar("Train_Loss", train_loss_running, epoch)

            if iteration % config["print_every_n"] == (config["print_every_n"] - 1):
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss_running / config["print_every_n"]:.6f}')
                train_loss_running = 0.
            
            best_loss = 100
            #Path(f'/ckpts').mkdir(exist_ok=True, parents=True)
            if epoch%config["save_freq"] == 0 or train_loss_running<best_loss:
                file_name = 'ckpt-best.pth' if train_loss_running<best_loss else 'ckpt-epoch-%03d.pth' % epoch
                output_path = "./ckpts/"+file_name
                torch.save({
                    'epoch_index': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, output_path)  # yapf: disable

                print(f'Saved checkpoint to {output_path}')
                if train_loss_running<best_loss:
                    best_loss = train_loss_running


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
        num_workers=config['num_workers'],   # Data is usually loaded in parallel by num_workers
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
    )

    model = GRNet()
    model.to(device)

    train(model=model,
          train_dataloader=train_dataloader,
          device=device,
          config=config,
          val_dataloader=None)
