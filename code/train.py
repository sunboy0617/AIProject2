import os
import sys
import torch
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from pdb import set_trace as st
from tqdm.autonotebook import tqdm
import pprint

import utils
import dataset
import model
import optim


def train(opt_path):
    ## initialization
    log_saveroot = '../logs'
    opt = utils.read_option(opt_path, log_saveroot)
    if opt['use_gpu']:
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    else:
        device = 'cpu'
    if opt['fixed_random']:
        utils.set_random_seed()
    opt_train = opt['train']
    opt_data = opt['dataset']
    if opt_data['type'] == 1:
        out_nc = 20
    elif opt_data['type'] == 2:
        out_nc = 100
    pprint.pprint(opt)

    ## define dataset
    train_dataset = dataset.Q1Dataset_pic(opt_data=opt_data, phase='train', root=None)
    val_dataset = dataset.Q1Dataset_pic(opt_data=opt_data, phase='val', root=None)
    train_dataloader = DataLoader(train_dataset, batch_size=opt_train['batch_size'], num_workers=opt_train['num_workers'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt_train['batch_size'], num_workers=opt_train['num_workers'], shuffle=False)

    ## define network, optimizer, scheduler, criterion
    net_m = model.init_model(opt_train, out_nc).to(device)
    net_m.model = nn.DataParallel(net_m.model)
    optimizer = optim.init_optim(opt_train, net_m)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt_train['milestones'], gamma=opt_train['gamma'])
    if opt['loss'] == "CEloss":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    ## define tensorboard
    path_log = opt['save_root']
    all_scalars = {}
    writer = utils.create_writer(path_log)

    ## train
    with tqdm(total = opt_train['epoch']*len(train_dataloader)) as pbar:
        for epoch in range(opt_train['epoch']):
            train_loss = 0.0
            train_accu = 0.0
            total = 0
            total_right = 0
            net_m.train()
            scheduler.step()

            for img, label in train_dataloader:
                optimizer.zero_grad()
                img = img.to(device)
                label = label.to(device)
                total += label.shape[0]

                if opt_train['model'] == 'ResNext':
                    att_outputs, out, _ = net_m(img)
                    att_loss = criterion(att_outputs, label)
                    per_loss = criterion(out, label)
                    loss = att_loss + per_loss
                else:
                    out = net_m(img)
                    loss = criterion(out, label)
                loss.backward()
                if opt_train['optim'] == 'SAM':
                    optimizer.first_step(zero_grad=True)
                    criterion(net_m(img),label).backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.step()
                train_loss += loss.item()

                _,pred = out.max(1)
                total_right += (pred == label).sum().item()
                pbar.update(1)

            train_accu = total_right / total
            writer, all_scalars = utils.add_scalar(path_log, all_scalars, writer, 'train/loss', train_loss, epoch)
            writer, all_scalars = utils.add_scalar(path_log, all_scalars, writer, 'train/accu', train_accu, epoch)
            tqdm.write("Epoch: %d, loss: %.6f, accu: %.6f" % ((epoch+1), train_loss, train_accu))

            ## val
            if((epoch+1)%opt_train['val_interval'] == 0):
                net_m.eval()

                val_total = 0
                val_total_right = 0
                for img, label in val_dataloader:
                    img = img.to(device)
                    label = label.to(device)
                    val_total += label.shape[0]

                    with torch.no_grad():
                        if opt_train['model'] == 'ResNext':
                            _, out, _ = net_m(img)
                        else:
                            out = net_m(img)
                        _, pred = out.max(1)
                        val_total_right += (pred == label).sum().item()

                val_accu = val_total_right / val_total
                writer, all_scalars = utils.add_scalar(path_log, all_scalars, writer, 'val/accu', val_accu, epoch)
                tqdm.write("VAL Epoch: %d, VAL accu: %.6f" % ((epoch+1), val_accu))

            ## save model
            if((epoch+1)%opt_train['save_interval'] == 0):
                save_model_path = os.path.join(opt['save_root']+'/checkpoints',
                                           'model_%05d.pth' % (epoch + 1))
                net_m.save(save_model_path)
            
if __name__ == "__main__":
    assert(len(sys.argv) == 2)
    train(sys.argv[1])
    