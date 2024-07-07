"""
Train Settings.
"""


import argparse


mode = "train" # training or testing state: train or test
data_path = "./dataset/train"    # e.g., /home/xxx/dataset/
weights = "./models/pretain/AlexNet/alexnet-owt-7be5be79.pth" # train/test weights


class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Cardboard Detection')

        # -----------------------------
        # Params for data
        # -----------------------------
        parser.add_argument('--data_path', default=data_path, type=str,
                            help='root directory path of data')
        parser.add_argument('--batch_size', default=32, type=int,
                            help='batch size')
        
        # -----------------------------
        # Misc
        # -----------------------------
        parser.add_argument('--num_save_samples', type=int, default=10,
                            help='number of samples saved for visualization')
        parser.add_argument('--ckpt', default='./Carboard Detection',
                            help='folder to output checkpoints')

        self.parser = parser

    def add_train_arguments(self):
        parser = self.parser
        parser.add_argument('--mode', default=mode, type=str,
                            help="training or testing state: train or test")
        parser.add_argument('--optimizer', default='adam', type=str,
                            help='optimizer')
        parser.add_argument('--num_epoch', default=500, type=int,
                            help='epochs for training')
        parser.add_argument('--eval_epoch', default=5, type=int,
                            help='epochs for training')
        parser.add_argument('--lr', default=0.01, type=float,
                            help='learning rate')

    def parse_train_arguments(self):

        self.add_train_arguments()
        args = self.parser.parse_args()

        print("------------------------ Options ------------------------")
        for key, val in vars(args).items():
            print("{:16} {}".format(key, val))
        print("------------------------ Options ------------------------")

        return args