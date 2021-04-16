import os
import sys
import torch
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import pprint
import csv
import json

import dataset
import model


def test(experiment_name,epoch):
    model_saveroot = '../logs/'+ experiment_name
    csv_root = '../results/' + experiment_name
    if not os.path.exists(csv_root):
        os.makedirs(csv_root)
    opt_path = os.path.join(model_saveroot,'opt.json')
    with open(opt_path, 'r') as f:
        opt = json.load(f)
    if opt['use_gpu']:
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    else:
        device = 'cpu'
    opt_train = opt['train']
    opt_data = opt['dataset']
    opt_test = opt['test']
    save_model_path = os.path.join(model_saveroot+'/checkpoints','model_%05d.pth' % (epoch))
    save_model_path2 = os.path.join(model_saveroot+'/checkpoints','model2_%05d.pth' % (epoch))

    if opt_data['type'] == 1:
        out_nc = 20
        test_phase = 'coarse'
    elif opt_data['type'] == 2:
        out_nc = 100
        test_phase = 'fine'
    pprint.pprint(opt)

    test_dataset = dataset.Q1Dataset_pic(opt_data=opt_data, phase='test', root=None)
    test_dataloader = DataLoader(test_dataset,opt_test['batch_size'],num_workers=opt_test['num_workers'])
    net_m = model.init_model(opt_train, out_nc).to(device)
    net_m.load(save_model_path)
    net_m2 = model.init_model(opt_train, out_nc).to(device)
    net_m2.load(save_model_path2)
    
    pred_final = []
    idx_list = []

    with tqdm(total=len(test_dataloader)) as pbar:
        for img,idx in test_dataloader:
            img = img.to(device)
            idx = idx.to(device)
            net_m.eval()
            net_m2.eval()

            with torch.no_grad():
                out1 = net_m(img)
                out2 = net_m2(img)
                _, pred = (out1+out2).max(1)
                pred_final += pred.cpu().numpy().tolist()
                idx_list += idx.cpu().numpy().tolist()
                pbar.update(1)
            

    with open(os.path.join(csv_root,test_phase + '_'+str(epoch)+'.csv'),'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['image_id',test_phase + '_label'])
        
        print("Saving...")
        for i in range(len(pred_final)):
            writer.writerow([idx_list[i],pred_final[i]])




if __name__ == "__main__":
    assert (len(sys.argv) == 3)
    experiment_name = sys.argv[1]
    epoch = int(sys.argv[2])
    test(experiment_name, epoch)