from config import grnet_config
from model import train_grnet, grnet
import argparse


if __name__ == "__main__":
    #TRAINABLE PARAMETERS: completion, classification, all
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-tr", "--trainable", 
                           help="trainable parameters", type=str, 
                           default="completion")
    args = argParser.parse_args()
    print("Trainable Params: %s" % args.trainable)
    model = grnet.GRNet()
    trainable_params = []
    training_mode = args.trainable
    config = grnet_config.config
    config["train_mode"] = training_mode
    for name, module in model.named_children():
        #print(name)
        if name in grnet_config.completion_layers and training_mode == "completion":
            trainable_params += list(module.parameters())
        elif name in grnet_config.classification_layers and training_mode == "classification":
            trainable_params += list(module.parameters())
        elif training_mode=="all":
            trainable_params += list(module.parameters())
    
    train_grnet.main(config, trainable_params)