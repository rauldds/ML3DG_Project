from model.color_net import EncoderDecoder
from utils.color_class import ColoredMeshesDataset, collate_fn
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from config import color_net_config
def train(config, train_dataloader, val_dataloader, num_epochs, device):
    model = EncoderDecoder()
    model = model.to(torch.double)

    # Define your loss function
    criterion = torch.nn.SmoothL1Loss()

    # Define your optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    tb = SummaryWriter()
    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            target_mesh = batch['target_mesh'].to(torch.double)
            incomplete_mesh = batch['incomplete_mesh'].to(torch.double)
            #print((incomplete_mesh.dtype))

            # Forward pass
            outputs = model(incomplete_mesh)
            loss = criterion(outputs, target_mesh)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track the loss
            running_loss += loss.item()

            # validation
            iteration = epoch * len(train_dataloader) + batch_idx
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                model.eval()
                val_loss = 0
                for batch_val in val_dataloader:
                    ColoredMeshesDataset.send_data_to_device(batch_val, device)
                    with torch.no_grad():
                        predicted_color = model(batch_val["incomplete_mesh"].to(torch.double))
                        target_color = batch_val["target_mesh"].to(torch.double)
                        loss = criterion(predicted_color, target_color)
                        val_loss += loss.item()
                val_loss /= len(val_dataloader)
                print(f'[{epoch:03d}/{batch_idx:05d}] val_loss: {val_loss:.6f}')
                tb.add_scalar("val_loss", val_loss, epoch)
                model.train()


        # Print the average loss for the epoch
        epoch_loss = running_loss / len(train_dataloader)
        tb.add_scalar("Train_Loss", epoch_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        if epoch % config["save_freq"] == 0:
            file_name = 'ckpt-epoch-%03d-' % epoch
            file_name = file_name + ".pth"
            output_path = "./ckpts/colored" + "/" + file_name
            torch.save({
                'epoch_index': epoch,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, output_path)  # yapf: disable
    # After training, you can use the trained model for inference

if __name__ == "__main__":
    device = torch.device('cpu')
    config = color_net_config.config
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')
    config = color_net_config.config
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-dp", "--dataset_path",
                            help="dataset path",
                            type=str,
                            default="/media/rauldds/TOSHIBA EXT/ML3G/Full_Project_Dataset")
    args = argParser.parse_args()
    dataset = ColoredMeshesDataset("overfit",args.dataset_path)
    train_dataloader = torch.utils.data.DataLoader(dataset, 
                                                   batch_size=config["batch_size"],
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=config["batch_size"],
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
    train(config,train_dataloader, val_dataloader, config["max_epochs"], device)