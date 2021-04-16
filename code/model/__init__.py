from . import network
from . import wide_res_net

def init_model(opt_train, out_nc):
    if opt_train['model'] == 'Resnet':
        return network.ResNetClassifier(out_nc)
    elif opt_train['model'] == 'SunNet':
        return network.SunNetClassifier(out_nc)
    elif opt_train['model'] == 'VGGNet':
        return network.VGGNetClassifier(out_nc, opt_train['modeltype'])
    elif opt_train['model'] == 'WideResNet':
        return network.WideResNetClassifier(out_nc,depth=opt_train['depth'], width_factor=opt_train['width'])
    elif opt_train['model'] == 'WideResNet_v2':
        return network.WideResNetClassifier_v2(out_nc,depth=opt_train['depth'], width_factor=opt_train['width'])
    elif opt_train['model'] == 'ResNext':
        return network.ResNextClassifier(opt_train['cardinality'], out_nc,depth=opt_train['depth'])
    elif opt_train['model'] == 'SEResNet':
        return network.SEResNetClassifier(out_nc)
    else:
        raise NotImplementedError('Model type not recognized.')