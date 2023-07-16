import torch
from utils.data_loaders import ScanObjectNNDataset
from data_e3.shapenet import ShapeNet
from model.grnet import GRNet_clas, GRNet_comp
import skimage
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import utils.data_loaders

torch.autograd.set_detect_anomaly(True)


class log_space_L1_loss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(log_space_L1_loss, self).__init__()

    def forward(self, reconstruction, target):
        target_log = torch.log(torch.abs(target) + 1)

        loss = torch.abs(torch.abs(reconstruction) - target_log)

        # sign_mask = torch.sign(reconstruction) * torch.sign(target) < 0
        # loss = torch.where(sign_mask, 2 * loss, loss)

        return torch.mean(loss)

def train(model_comp, model_clas, train_dataloader, val_dataloader,
          device, config):

    # Defining Losses
    completion_loss_criterion = log_space_L1_loss()
    classification_loss_criterion = torch.nn.BCEWithLogitsLoss()
    
    # Moving Loss to the corresponding device
    classification_loss_criterion.to(device)
    completion_loss_criterion.to(device)
    
    # Resume training from checkpoint in case the resume flag is True
    if config["resume"]:
        # Pulling different checkpoints depending on what's being trained
        if config["train_mode"] == "completion":
            ckpt = torch.load("./ckpts/ScanObjectNN/ckpt-best-completion.pth")
            model_comp.load_state_dict(ckpt["model_comp"])
        elif config["train_mode"] == "classification":
            ckpt = torch.load("./ckpts/ScanObjectNN/ckpt-best-classification.pth")
            model_clas.load_state_dict(ckpt["model_clas"])
        elif config["train_mode"] == "all":
            try:
                ckpt = torch.load("./ckpts/ScanObjectNN/ckpt-best-all.pth")
                model_comp.load_state_dict(ckpt["model_comp"])
                model_clas.load_state_dict(ckpt["model_clas"])
            except Exception:
                ckpt = torch.load("./ckpts/ScanObjectNN/ckpt-best-completion.pth")
                model_comp.load_state_dict(ckpt["model_comp"])
                cmp_optim_dict = ckpt["cmp_optim"]
                cmp_scheduler_dict = ckpt["cmp_scheduler"]
                ckpt = torch.load("./ckpts/ScanObjectNN/ckpt-best-classification.pth")
                model_clas.load_state_dict(ckpt["model_clas"])
                ckpt["cmp_optim"] = cmp_optim_dict
                ckpt["cmp_scheduler"] = cmp_scheduler_dict

    #Weighted loss to train both completion part and classifcation parts together
    weight_CE = torch.tensor(0.5, requires_grad=True)
    weight_L1 = torch.tensor(0.5, requires_grad=True)

    # Defining the optimizers for each network
    cmp_optim = torch.optim.Adam(model_comp.parameters(),
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'],
                                 betas=config["betas"])
    cls_optim = torch.optim.Adam(model_clas.parameters(),
                                 lr=config["cls_net"]['learning_rate'],
                                 weight_decay=config["cls_net"]['weight_decay'])
    
    # Defining the lr schedulers for each network
    cmp_scheduler = torch.optim.lr_scheduler.MultiStepLR(cmp_optim, 
                                                         milestones = config["milestones"],
                                                         gamma = config["gamma"])
    cls_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(cls_optim, min_lr=1e-10)

    # Setting nets in training mode based on the value of the train_mode flag
    if config["train_mode"] == "completion":
        model_comp.train()
        model_clas.eval()
    elif config["train_mode"] == "classification":
        model_comp.eval()
        model_clas.train()
    elif config["train_mode"] == "all":
        model_comp.train()
        model_clas.train()

    # Load the scheduler and optimizer state from checkpoint when the resume flag is True
    if config["resume"]:
        cmp_optim.load_state_dict(ckpt["cmp_optim"])
        cmp_scheduler.load_state_dict(ckpt["cmp_scheduler"])
        cls_optim.load_state_dict(ckpt["cls_optim"])
        cls_scheduler.load_state_dict(ckpt["cls_scheduler"])

    # Start Tensorboard Writer
    tb = SummaryWriter()
    # Initialize the best loss with very high value
    best_loss = 100

    #Train for the number of epochs defined in config file
    for epoch in range(config['max_epochs']):

        # Reset batch loss and accuracy after every epoch
        batch_loss = 0
        train_accuracy = 0

        #Iterate through the training split batches
        for batch_idx, batch in enumerate(train_dataloader):
            #Move the current batch data to the corresponding device
            ScanObjectNNDataset.send_data_to_device(batch, device)

            #Reset gradients after each batch
            cmp_optim.zero_grad()
            cls_optim.zero_grad()

            #Forward pass through model depending on dataset and training mode flags
            if config["dataset"] =="Shapenet":
                reconstruction, skip = model_comp(batch["incomplete_view"])
            else:
                if config["train_mode"] == "classification":
                    # Ideally the classifcation net would receive complete versions of an object
                    # that's the reason why during classification pretraining we provide target
                    # SDFs instead of the incomplete SDFs
                    reconstruction = batch["target_sdf"]
                    # The classification net extracts relevant features from the completion net
                    # However, as during pretraining we don't have those features, we only provide
                    # One Tensors as placeholders
                    skip = {
                        '32_r': torch.ones([config["batch_size"], 32, 32, 32, 32],device=device),
                        '16_r': torch.ones([config["batch_size"], 64, 16, 16, 16],device=device),
                        '8_r': torch.ones([config["batch_size"], 128, 8, 8, 8],device=device)
                    }
                    class_pred = model_clas(reconstruction,skip)
                else:
                    reconstruction, skip = model_comp(batch["incomplete_view"])
                    cls_recon = torch.exp(reconstruction)-1
                    # Had to detach to stop the classification net from trying to compute 
                    # gradients all the way back in the completion net
                    skip_detached = {key: value.detach() for key, value in skip.items()}
                    class_pred = model_clas(cls_recon.detach(),skip_detached)

            # Extract targets for loss computation
            target_sdf = batch["target_sdf"]
            if config["dataset"] !="Shapenet":
                class_target = batch["class"]
            
            # Applying a mask to only compare new data from the SDFs
            # when computing the completion loss
            reconstruction[batch["incomplete_view"] > 0] = 0
            target_sdf[batch["incomplete_view"] > 0] = 0
            
            # Compute loss based on training mode
            if config["train_mode"] == "completion":
                loss = completion_loss_criterion(reconstruction, target_sdf)
            elif config["train_mode"] == "classification":
                loss = classification_loss_criterion(class_pred,class_target)
            elif config["train_mode"] == "all":
                loss_comp = completion_loss_criterion(reconstruction, target_sdf)
                loss_class = classification_loss_criterion(class_pred,class_target)

                # Weighted loss calculation
                scaled_loss_CE = weight_CE * loss_class
                scaled_loss_comp = weight_L1 * loss_comp
                loss = scaled_loss_CE + scaled_loss_comp

            # Backpropagation
            loss.backward()

            # Doing an optimizer step
            if config["train_mode"] == "completion":
                cmp_optim.step()
            elif config["train_mode"] == "classification":
                cls_optim.step()
            elif config["train_mode"] == "all":
                cmp_optim.step()
                cls_optim.step()

                # # Updating weights based on gradients after each batch iteration
                # weight_CE.data -= config["cls_net"]['learning_rate'] * weight_CE.grad.data
                # weight_L1.data -= config['learning_rate'] * weight_L1.grad.data

                # Updating weights based on gradients after each batch iteration
                weight_CE.data -= config['learning_rate_loss_weights'] * weight_CE.grad.data
                weight_L1.data -= config['learning_rate_loss_weights'] * weight_L1.grad.data

                #Setting gradients to 0 after each batch iteration
                weight_L1.grad.data.zero_()
                weight_CE.grad.data.zero_()

            # Obtaining the batch loss after each batch iteration
            batch_loss += loss.item()
            # Computing current iteration
            iteration = epoch * len(train_dataloader) + batch_idx

            # Calculating training accuracy for each batch iteration
            if config["train_mode"] != "completion":
                predicted_labels = torch.argmax(class_pred, dim=1) 
                target_labels = torch.argmax(class_target, dim=1) 
                correct = (predicted_labels == target_labels).sum().item()
                total = len(class_target)
                train_accuracy += correct / total

            # Computing validation after a certain amount of batch iterations
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
             
                # Setting the trained net to evaluation mode
                if config["train_mode"] == "completion":
                    model_comp.eval()
                elif config["train_mode"] == "classification":
                    model_clas.eval()
                elif config["train_mode"] == "all":
                    model_comp.eval()
                    model_clas.eval()

                # Initializing validation loss, accuracy (for classification), and flag 
                # for obtention of tensorboard mesh visualization
                val_loss = 0.
                val_accuracy = 0.
                sample_mesh_obtained = False

                print("[INFO] Validating")

                # Iteration through whole validation set
                for batch_val in val_dataloader:
                    ScanObjectNNDataset.send_data_to_device(batch_val, device)

                    # Computing forward pass through the trained net(s)
                    with torch.no_grad():
                        if config["dataset"] =="Shapenet":
                            reconstruction, skip = model_comp(batch_val["incomplete_view"])
                        else:
                            if config["train_mode"] == "classification":
                                reconstruction = batch_val["target_sdf"]
                                skip = {
                                    '32_r': torch.ones([config["batch_size"], 32, 32, 32, 32],device=device),
                                    '16_r': torch.ones([config["batch_size"], 64, 16, 16, 16],device=device),
                                    '8_r': torch.ones([config["batch_size"], 128, 8, 8, 8],device=device)
                                }
                                class_pred = model_clas(reconstruction,skip)
                            else:
                                reconstruction, skip = model_comp(batch_val["incomplete_view"])
                                cls_recon = torch.exp(reconstruction) - 1
                                skip_detached = {key: value.detach() for key, value in skip.items()}
                                class_pred = model_clas(cls_recon.detach(),skip_detached)

                        # Generate visualization meshes for the reconstruction, input and target
                        # SDFs to observe completion net changes through diffferent steps
                        if config["train_mode"] != "classification":
                            # Reconstruction mesh obtention
                            vis_recon = reconstruction[0]
                            vis_recon = torch.exp(vis_recon) - 1
                            vis_recon = vis_recon.detach().cpu().numpy()
                            vis_recon = vis_recon.reshape((64, 64, 64))
                            vertices, faces, normals, _ = skimage.measure.marching_cubes(vis_recon, level=0)
                            vert_recon = torch.as_tensor(np.array([vertices]), dtype=torch.float)
                            faces_recon = torch.as_tensor(np.array([faces]), dtype=torch.int)
                            # Input mesh obtention
                            vis_inc = batch_val["incomplete_view"][0]
                            vis_inc = vis_inc.detach().cpu().numpy()
                            vis_inc = vis_inc.reshape((64, 64, 64))
                            vertices, faces, normals, _ = skimage.measure.marching_cubes(vis_inc, level=0)
                            vert_inc = torch.as_tensor(np.array([vertices]), dtype=torch.float)
                            faces_inc = torch.as_tensor(np.array([faces]), dtype=torch.int)
                            # Target mesh obtention
                            vis_com = batch_val["target_sdf"][0]
                            vis_com = vis_com.detach().cpu().numpy()
                            vis_com = vis_com.reshape((64, 64, 64))
                            vertices, faces, normals, _ = skimage.measure.marching_cubes(vis_com, level=0)
                            vert_com = torch.as_tensor(np.array([vertices]), dtype=torch.float)
                            faces_com = torch.as_tensor(np.array([faces]), dtype=torch.int)

                        # Extract targets for loss computation
                        target_sdf = batch_val["target_sdf"]
                        if config["dataset"] != "Shapenet":
                            class_target = batch_val["class"]

                        # Applying a mask to only compare new data from the SDFs
                        # when computing the completion loss
                        reconstruction[batch_val["incomplete_view"] > 0] = 0
                        target_sdf[batch_val["incomplete_view"] > 0] = 0

                        # Compute loss based on training mode
                        if config["train_mode"] == "completion":
                            loss = completion_loss_criterion(reconstruction, target_sdf)
                        elif config["train_mode"] == "classification":
                            loss = classification_loss_criterion(class_pred, class_target)
                        elif config["train_mode"] == "all":
                            loss_comp = completion_loss_criterion(reconstruction, target_sdf)
                            loss_class = classification_loss_criterion(class_pred, class_target)
                            scaled_loss_CE = weight_CE * loss_class
                            scaled_loss_comp = weight_L1 * loss_comp
                            loss = scaled_loss_CE + scaled_loss_comp

                    # Obtaining validation loss after each val batch iteration
                    val_loss += loss.item()

                    # Calculating validation accuracy for each batch iteration
                    if config["train_mode"] != "completion":
                        predicted_labels = torch.argmax(class_pred, dim=1) 
                        target_labels = torch.argmax(class_target, dim=1) 
                        correct = (predicted_labels == target_labels).sum().item()
                        total = len(class_target)
                        val_accuracy += correct / total
                
                # Averaging the validation based on the number of val batches
                val_loss /= len(val_dataloader)
                print(f'[{epoch:03d}/{batch_idx:05d}] val_loss: {val_loss:.6f}')
                # Averaging validation accuracy based on the number of val batches
                val_accuracy = val_accuracy/len(val_dataloader)

                # Setting back the net(s) to train mode
                if config["train_mode"] == "completion":
                    model_comp.train()
                    model_clas.eval()
                elif config["train_mode"] == "classification":
                    model_comp.eval()
                    model_clas.train()
                elif config["train_mode"] == "all":
                    model_comp.train()
                    model_clas.train()

                # Writing the validation loss and accuracy in tensorboard
                tb.add_scalar("Val loss", val_loss, epoch)
                if config["train_mode"] != "completion":
                    tb.add_scalar("Val Accuracy", val_accuracy, epoch)

                # Writing the reconstruction, input, and target meshes to tensorboard
                if config["train_mode"] != "classification":
                    tb.add_mesh("Recon Mesh", vertices=vert_recon, faces=faces_recon,global_step=epoch)
                    tb.add_mesh("Input Mesh", vertices=vert_inc, faces=faces_inc,global_step=epoch)
                    tb.add_mesh("Target Mesh", vertices=vert_com, faces=faces_com,global_step=epoch)

        # Averaging batch loss and train accuracy based on the number of training batches
        batch_loss = batch_loss/len(train_dataloader)
        train_accuracy = train_accuracy/len(train_dataloader)
        print(f'[{epoch:03d}] Train_loss: {batch_loss:.6f}, weight for CE loss:{weight_CE:.4f}, weight for L1 loss:{weight_L1:.6f}')
        # Writing batch loss to tensorboard every epoch
        tb.add_scalar("Train_Loss", batch_loss, epoch)
        metrics_dict ={"train loss": batch_loss,
                        "train accuracy": train_accuracy,
                        "epoch": epoch}

        # Doing a learning rate scheduler step after each epoch
        if config["train_mode"] == "completion":
            cmp_scheduler.step()
        elif config["train_mode"] == "classification":
            cls_scheduler.step(batch_loss)
            # Writing accuracy, hyperparams, and some metrics to tensorboard
            tb.add_scalar("Cls Train Accuracy", train_accuracy, epoch)
            tb.add_hparams(config["cls_net"], metrics_dict)
        elif config["train_mode"] == "all":
            cmp_scheduler.step()
            cls_scheduler.step(batch_loss)
            # Writing classification train accuracy to tensorboard
            tb.add_scalar("Cls Train Accuracy", train_accuracy, epoch)
        
        # Write new checkpoint after a certain amount of epochs
        if epoch%config["save_freq"] == 0:
            # Assign checkpoint name based on how good the current loss is
            file_name = 'ckpt-best-' if batch_loss<best_loss else 'ckpt-epoch-%03d-' % epoch
            file_name = file_name + config["train_mode"] + ".pth"
            output_path = "./ckpts/"+config["dataset"]+"/"+file_name
            # Save checkpoint
            torch.save({
                'epoch_index': epoch,
                'model_comp': model_comp.state_dict(),
                'model_clas': model_clas.state_dict(),
                'cmp_optim': cmp_optim.state_dict(),
                'cls_optim': cls_optim.state_dict(),
                'cmp_scheduler': cmp_scheduler.state_dict(),
                'cls_scheduler': cls_scheduler.state_dict(),
                "weight_CE": weight_CE,
                "weight_L1:": weight_L1
            }, output_path)  # yapf: disable

            # Update best loss if necessary
            if batch_loss<best_loss:
                best_loss = batch_loss


def main(config):

    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')
    
    print(f'Resume Training: {config["resume"]}')


    # Create Dataloaders
    if config["dataset"] == "Shapenet":
        train_dataset = ShapeNet('train' if not config['is_overfit'] else 'overfit')
    else:
        train_dataset = ScanObjectNNDataset('train' if not config['is_overfit'] else 'overfit')
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=config['num_workers'],   # Data is usually loaded in parallel by num_workers
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
    )

    val_dataset = ScanObjectNNDataset('val')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size_val'],
        num_workers=config['num_workers'],
        pin_memory=True
    )

    model_comp = GRNet_comp()
    model_comp.to(device)
    model_clas = GRNet_clas()
    model_clas.to(device)

    train(model_comp=model_comp,
          model_clas=model_clas,
          train_dataloader=train_dataloader,
          device=device,
          config=config,
          val_dataloader=val_dataloader)
