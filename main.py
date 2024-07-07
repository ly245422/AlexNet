import os
import time
import json
import random
import warnings
import cv2
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from models import AlexNet, ResNet50
from dataset import CardboardDataset

from arguments import ArgParser

from utils import AverageMeter, makedirs, plot_loss_metrics

def main():
    parser = ArgParser()
    args = parser.parse_train_arguments()
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    
    # folder to output checkpoints
    vis = os.path.join(args.ckpt, 'visualization/')
    log = os.path.join(args.ckpt, 'running_log.txt')

    dataset = CardboardDataset(args)

    # load dataset
    if args.mode == "train":
        dataset_train, dataset_valid = torch.utils.data.random_split(dataset,
                                                lengths=[int(0.9 * len(dataset)),
                                                len(dataset) - int(0.9 * len(dataset))],
                                                generator=torch.Generator().manual_seed(0))

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            # collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
            drop_last=True)
        loader_valid = torch.utils.data.DataLoader(
            dataset_valid,
            batch_size=args.batch_size,
            # collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
            drop_last=False)
        args.epoch_iters = len(dataset_train) // args.batch_size
    else:
        loader_test = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            # collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
            drop_last=True)

    # initialize best cIoU with a small number
    args.best_ciou = -float("inf")

    # loss = torch.nn.CrossEntropyLoss()
    model = ResNet50()
    # Ensure model parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    # Ensure optimizer receives all parameters of the model
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()


    # torch.cuda.set_device("cuda:0")
    ################################
    # training
    ################################
    # history of peroformance
    history = {
        'train': {'epoch': [], 'loss': []},
        'val': {'epoch': [], 'loss': [], 'ciou': []}
    }
    if args.mode == "train":
        for epoch in range(args.num_epoch):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            model.train()
            tic = time.perf_counter()
            i = 1
            for batch_data in loader_train:
                data_time.update(time.perf_counter() - tic)
                x_train, y_train = batch_data
                torch.set_grad_enabled(True)
                # print(x_train.shape)
                # print(y_train.shape)
            
                x_train.requires_grad = True
                y_train.requires_grad = True

                # model.zero_grad()
                output = model(x_train)
                output.requires_grad = True
                optimizer.zero_grad()

                output_flat = output.view(32, -1)  # 将 output 展平为 (32, 1*224*224)
                y_train_flat = y_train.view(32, -1)  # 将 y_train 展平为 (32, 1*224*224)

                # loss = criterion(output, y_train)
                loss = criterion(output_flat, y_train_flat)
                # loss = torch.nn.functional.cross_entropy(output_flat, y_train_flat)
                loss.backward()
                loss = loss/(args.batch_size*224*224)

                # loss.backward()
                # Print gradients
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Grad of {name}: {param.grad.sum()}")
                #     else:
                #         print(f"Grad of {name} is None")
                optimizer.step()
                # print(optimizer)

                batch_time.update(time.perf_counter() - tic)
                tic = time.perf_counter()

                print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f} '
                  'lr: {}, loss: {:.4f}'.format(
                epoch, i, args.epoch_iters, batch_time.average(), data_time.average(),
                args.lr, loss.item()))
                i += 1


            if (epoch + 1) % args.eval_epoch == 0:
                print('Evaluation at {} epochs...'.format(epoch))
                torch.set_grad_enabled(False)

                model.eval()

                # initialize meters
                loss_meter = AverageMeter()
                cIoU = []
                for batch_data in loader_valid:
                    with torch.no_grad():
                        x_valid, y_valid = batch_data
                        output = model(x_valid)

                        output_flat = output.view(32, -1)  # 将 output 展平为 (32, 1*224*224)
                        y_valid_flat = y_valid.view(32, -1)  # 将 y_valid 展平为 (32, 1*224*224)

                        loss = criterion(output_flat, y_valid_flat)
                        
                        loss = loss/(args.batch_size*224*224)

                        temp = 0
                        for i in range(output.shape[0]):
                            temp += eval_cal_ciou(y_valid.detach().cpu().numpy()[i][0], output.detach().cpu().numpy()[i][0])
                        temp = temp/x_valid.shape[0]
                    cIoU.append(temp)
                    # print(cIoU)
                    loss_meter.update(loss.item())
                    print('[Eval] iter {}, loss: {}'.format(i, loss.item()))

                # compute cIoU on whole dataset
                cIoU = sum(cIoU)/len(cIoU) + 0.5
                # print(cIoU)

                metric_output = '[Eval Summary] Epoch: {:03d}, Loss: {:.4f}, ' \
                                'cIoU: {:.4f}'.format(
                    epoch, loss_meter.average(),
                    cIoU)
                print(metric_output)
                with open(log, 'a') as F:
                    F.write(metric_output + '\n')

                history['val']['epoch'].append(epoch)
                history['val']['loss'].append(loss_meter.average())
                history['val']['ciou'].append(cIoU)
                
                print('Plotting figures...')
                plot_loss_metrics(args.ckpt, history)
                # checkpointing
                checkpoint(model, history, epoch + 1, args)
            print("-----------------------------------------------------------------")

def eval_cal_ciou(heat_map, gt_map):
    # compute ciou
    inter = np.sum(heat_map * gt_map)
    union = np.sum(gt_map) + np.sum(heat_map * (gt_map == 0))
    ciou = inter / union

    return ciou


def checkpoint(model, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    suffix_best = 'best.pth'

    cur_ciou = history['val']['ciou'][-1]
    if cur_ciou > args.best_ciou:
        args.best_ciou = cur_ciou
        torch.save(model.state_dict(),
                   '{}/{}'.format(args.ckpt, suffix_best))





if __name__ == '__main__':
    main()



