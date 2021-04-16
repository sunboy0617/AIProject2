import torch
from . import sam

def init_optim(opt_train, net_m):
    if opt_train['optim'] == 'Adam':
        return torch.optim.Adam(net_m.parameters(), lr=opt_train['lr'], betas=[0.9, 0.999], weight_decay=5e-4)
    elif opt_train['optim'] == 'SAM':
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        return sam.SAM(net_m.parameters(), base_optimizer, lr=opt_train['lr'], momentum=0.9)
    elif opt_train['optim'] == 'SGD':
        return torch.optim.SGD(net_m.parameters(), lr=opt_train['lr'], momentum=0.9, weight_decay=5e-4)
    else:
        raise NotImplementedError('Optimizer type not recognized.')