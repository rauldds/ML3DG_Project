from config import grnet_config
from model import train_grnet
import argparse

if __name__ == "__main__":
    #TRAINABLE PARAMETERS: completion, classification, all
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-tr", "--trainable",
                           help="trainable parameters", type=str,
                           default="completion")
    argParser.add_argument("-r", "--resume",
                           help="retrieve checkpoint", type=bool,
                           default=False)
    argParser.add_argument("-r", "--dataset",
                           help="retrieve checkpoint", type=str,
                           default="ScanObjectNN")

    args = argParser.parse_args()
    print("Trainable Params: %s" % args.trainable)
    training_mode = "completion"
    if args.dataset != "Shapenet":
        training_mode = args.trainable
    config = grnet_config.config
    config["train_mode"] = training_mode
    config["resume"] = args.resume
    config["dataset"] = args.dataset
    
    train_grnet.main(config)