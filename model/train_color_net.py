from model.color_net import EncoderDecoder
from utils.color_class import ColoredMeshesDataset, collate_fn
import torch
import argparse

def train(config, train_dataloader,num_epochs):
    model = EncoderDecoder()
    model = model.to(torch.double)

    # Define your loss function
    criterion = torch.nn.SmoothL1Loss()

    # Define your optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_dataloader:
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

        # Print the average loss for the epoch
        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # After training, you can use the trained model for inference

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-dp", "--dataset_path",
                            help="dataset path",
                            type=str,
                            default="/media/rauldds/TOSHIBA EXT/ML3G/Full_Project_Dataset")
    args = argParser.parse_args()
    dataset = ColoredMeshesDataset("overfit",args.dataset_path)
    train_dataloader = torch.utils.data.DataLoader(dataset, 
                                                   batch_size=2, 
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
    train("config",train_dataloader,200)