from math import gamma
import torch
from torch.utils.data import DataLoader
from utils.data_loaders import ScanObjectNNDataset
from data_e3.shapenet import ShapeNet
from model.grnet import GRNet_clas, GRNet_comp
import skimage
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
#from model.extensions.chamfer_dist import ChamferDistance
import utils.data_loaders
import os

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

    completion_loss_criterion = log_space_L1_loss()
    completion_loss_criterion.to(device)
    classification_loss_criterion = torch.nn.BCEWithLogitsLoss()
    classification_loss_criterion.to(device)

    if config["resume"]:
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

    weight_CE = torch.tensor(0.5, requires_grad=True)
    weight_L1 = torch.tensor(0.5, requires_grad=True)

    cmp_optim = torch.optim.Adam(model_comp.parameters(),
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'],
                                 betas=config["betas"])
    cmp_scheduler = torch.optim.lr_scheduler.MultiStepLR(cmp_optim, 
                                                         milestones = config["milestones"],
                                                         gamma = config["gamma"])
    cls_optim = torch.optim.Adam(model_clas.parameters(),
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'],
                                 betas=config["betas"])
    cls_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(cls_optim, min_lr=1e-10)

    
    if config["train_mode"] == "completion":
        model_comp.train()
        model_clas.eval()
    elif config["train_mode"] == "classification":
        model_comp.eval()
        model_clas.train()
    elif config["train_mode"] == "all":
        model_comp.train()
        model_clas.train()
   
    if config["resume"]:
        cmp_optim.load_state_dict(ckpt["cmp_optim"])
        cmp_scheduler.load_state_dict(ckpt["cmp_scheduler"])
        cls_optim.load_state_dict(ckpt["cls_optim"])
        cls_scheduler.load_state_dict(ckpt["cls_scheduler"])

    train_loss_running = 0.

    tb = SummaryWriter()
    best_loss = 100
    for epoch in range(config['max_epochs']):
        batch_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            ScanObjectNNDataset.send_data_to_device(batch, device)

            cmp_optim.zero_grad()
            cls_optim.zero_grad()

            if config["dataset"] =="Shapenet":
                reconstruction, skip = model_comp(batch["incomplete_view"])
            else:
                if config["train_mode"] == "classification":
                    reconstruction = batch["incomplete_view"]
                    # TODO: IF YOU HAVE A BETTER ALTERNATIVE PLEASE CHANGE
                    skip = {
                        '32_r': torch.ones([config["batch_size"], 32, 32, 32, 32],device=device),
                        '16_r': torch.ones([config["batch_size"], 64, 16, 16, 16],device=device),
                        '8_r': torch.ones([config["batch_size"], 128, 8, 8, 8],device=device)
                    }
                    class_pred = model_clas(reconstruction,skip)
                else:
                    reconstruction, skip = model_comp(batch["incomplete_view"])
                    # HAD TO DETACH TO STOP THE CLASS PREDICTION NETWORK TO TRY TO COMPUTE 
                    # GRADIENTS ALL THE WAY BACK IN THE COMPLETION NETWORK
                    skip_detached = {key: value.detach() for key, value in skip.items()}
                    class_pred = model_clas(reconstruction.detach(),skip_detached)

            # TODO: understand the following mask:
            #  reconstruction[batch_val['input_sdf'][:, 1] == 1] = 0
            #  target[batch_val['input_sdf'][:, 1] == 1] = 0
            target_sdf = batch["target_sdf"]
            if config["dataset"] !="Shapenet":
                class_target = batch["class"]
                #class_target = class_target.long()
            reconstruction[batch["incomplete_view"] > 0] = 0
            target_sdf[batch["incomplete_view"] > 0] = 0
            
            if config["train_mode"] == "completion":

                loss = completion_loss_criterion(reconstruction, target_sdf)
            elif config["train_mode"] == "classification":
                loss = classification_loss_criterion(class_pred,class_target)
            elif config["train_mode"] == "all":
                loss_comp = completion_loss_criterion(reconstruction, target_sdf)
                loss_class = classification_loss_criterion(class_pred,class_target)

                scaled_loss_CE = weight_CE * loss_class
                scaled_loss_comp = weight_L1 * loss_comp
                loss = scaled_loss_CE + scaled_loss_comp

            loss.backward()

            if config["train_mode"] == "completion":
                cmp_optim.step()
            elif config["train_mode"] == "classification":
                cls_optim.step()
            elif config["train_mode"] == "all":
                cmp_optim.step()
                cmp_scheduler.step()

                weight_CE -= config['learning_rate'] * weight_CE.grad.data
                weight_L1 -= config['learning_rate'] * weight_L1.grad.data
                weight_L1.grad.data.zero()
                weight_CE.grad.data.zero()

            train_loss_running += loss.item()
            batch_loss += loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx

            if iteration % config["print_every_n"] == (config["print_every_n"] - 1):
                tb.add_scalar("Train_Loss",
                              train_loss_running / (config["print_every_n"] * batch["incomplete_view"].shape[0]), epoch)
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {(train_loss_running / (config["print_every_n"])):.6f}')
                train_loss_running = 0.

            # Validation
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                if config["train_mode"] == "completion":
                    model_comp.eval()
                elif config["train_mode"] == "classification":
                    model_clas.eval()
                elif config["train_mode"] == "all":
                    model_comp.eval()
                    model_clas.eval()

                val_loss = 0.
                print("[INFO] Validating")

                for batch_val in val_dataloader:
                    ScanObjectNNDataset.send_data_to_device(batch_val, device)

                    with torch.no_grad():
                        if config["dataset"] =="Shapenet":
                            reconstruction, skip = model_comp(batch_val["incomplete_view"])
                        else:
                            if config["train_mode"] == "classification":
                                reconstruction = batch_val["target_sdf"]
                                # TODO: IF YOU HAVE A BETTER ALTERNATIVE PLEASE CHANGE
                                skip = {
                                    '32_r': torch.ones([config["batch_size"], 32, 32, 32, 32],device=device),
                                    '16_r': torch.ones([config["batch_size"], 64, 16, 16, 16],device=device),
                                    '8_r': torch.ones([config["batch_size"], 128, 8, 8, 8],device=device)
                                }
                                class_pred = model_clas(reconstruction,skip)
                            else:
                                reconstruction, skip = model_comp(batch_val["incomplete_view"])
                                # HAD TO DETACH TO STOP THE CLASS PREDICTION NETWORK TO TRY TO COMPUTE 
                                # GRADIENTS ALL THE WAY BACK IN THE COMPLETION NETWORK
                                skip_detached = {key: value.detach() for key, value in skip.items()}
                                class_pred = model_clas(reconstruction.detach(),skip_detached)
                        if config["train_mode"] != "classification":
                            vis_recon = reconstruction[10]
                            vis_recon = torch.exp(vis_recon)-1
                            vis_recon = vis_recon.detach().cpu().numpy()
                            vis_recon = vis_recon.reshape((64, 64, 64))
                            vertices, faces, normals, _ = skimage.measure.marching_cubes(vis_recon, level=0)
                            vert_recon = torch.as_tensor([vertices], dtype=torch.float)
                            faces_recon = torch.as_tensor([faces], dtype=torch.int)
                            vis_inc = batch_val["incomplete_view"][0]
                            vis_inc = vis_inc.detach().cpu().numpy()
                            vis_inc = vis_inc.reshape((64, 64, 64))
                            vertices, faces, normals, _ = skimage.measure.marching_cubes(vis_inc, level=0)
                            vert_inc = torch.as_tensor([vertices], dtype=torch.float)
                            faces_inc = torch.as_tensor([faces], dtype=torch.int)
                            vis_com = batch_val["target_sdf"][0]
                            vis_com = vis_com.detach().cpu().numpy()
                            vis_com = vis_com.reshape((64, 64, 64))
                            vertices, faces, normals, _ = skimage.measure.marching_cubes(vis_com, level=0)
                            vert_com = torch.as_tensor([vertices], dtype=torch.float)
                            faces_com = torch.as_tensor([faces], dtype=torch.int)

                        target_sdf = batch_val["target_sdf"]
                        if config["dataset"] != "Shapenet":
                            class_target = batch_val["class"]
                        reconstruction[batch_val["incomplete_view"] > 0] = 0
                        target_sdf[batch_val["incomplete_view"] > 0] = 0

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

                    val_loss += loss.item()
                val_loss /= len(val_dataloader)
                print(f'[{epoch:03d}/{batch_idx:05d}] val_loss: {val_loss:.6f}')

                if config["train_mode"] == "completion":
                    model_comp.train()
                    model_clas.eval()
                elif config["train_mode"] == "classification":
                    model_comp.eval()
                    model_clas.train()
                elif config["train_mode"] == "all":
                    model_comp.train()
                    model_clas.train()

                tb.add_scalar("Val_Loss", val_loss, epoch)
                if config["train_mode"] != "classification":
                    tb.add_mesh("Recon Mesh", vertices=vert_recon, faces=faces_recon)
                    tb.add_mesh("Input Mesh", vertices=vert_inc, faces=faces_inc)
                    tb.add_mesh("Target Mesh", vertices=vert_com, faces=faces_com)



        if config["train_mode"] == "completion":
            cmp_scheduler.step()
        elif config["train_mode"] == "classification":
            cls_scheduler.step(val_loss)
        elif config["train_mode"] == "all":
            cmp_scheduler.step()
            cls_scheduler.step(val_loss)
    
        #Path(f'/ckpts').mkdir(exist_ok=True, parents=True)
        batch_loss = batch_loss/len(train_dataloader)
        tb.add_scalar("Train_Loss", batch_loss, epoch)
        #print(batch_loss)
        #if epoch%config["save_freq"] == 0 or batch_loss<best_loss:
        if epoch%config["save_freq"] == 0:
            file_name = 'ckpt-best-' if batch_loss<best_loss else 'ckpt-epoch-%03d-' % epoch
            file_name = file_name + config["train_mode"] + ".pth"
            output_path = "./ckpts/"+config["dataset"]+"/"+file_name
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

            # print(f'Saved checkpoint to {output_path}')
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
