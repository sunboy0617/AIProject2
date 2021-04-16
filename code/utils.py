import json
import os
import time
import random
import numpy as np
import torch
from tensorboardX import SummaryWriter

def read_option(opt_path, savepath):
    experiment_id = str(len(os.listdir(savepath))+1)
    assert os.path.exists(opt_path)
    with open(opt_path, 'r') as f:
        opt = json.load(f)
    opt['exp_time'] = time.strftime("%m-%d_%H-%M-%S")
    opt['save_root'] = os.path.join(savepath, experiment_id+'_'+opt['experiment_name'])
    os.makedirs(opt['save_root'])
    os.makedirs(opt['save_root']+'/checkpoints')
    f = open(opt_path, 'w')
    jsondata = json.dumps(opt,indent=4,separators=(',', ':'))
    f.write(jsondata)
    f.close()
    f = open(os.path.join(opt['save_root'], 'opt.json'), 'w')
    jsondata = json.dumps(opt,indent=4,separators=(',', ':'))
    f.write(jsondata)
    f.close()
    return opt

def set_random_seed(seed=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_writer(log_dir):
    writer = SummaryWriter(log_dir=str(log_dir))
    return writer

def add_scalar(log_dir, all_scalars, writer, key, value, epoch, this_tboard=True):
    keys = key.split('/')
    this_dict = all_scalars
    for key_ in keys:
        if key_ not in this_dict:
            this_dict[key_] = {}
        this_dict = this_dict[key_]
    this_dict[epoch] = value
    writer.add_scalar(key, value, epoch)
    return writer, all_scalars

def export_scalars(export_dir, all_scalars, writer, suffix=''):
    file_name = os.path.join(export_dir, 'all_scalars' + suffix + '.json')
    with open(file_name, 'w') as f:
        json.dump(all_scalars, f, cls=NumpyEncoder,
                  indent=2)
    writer.close()