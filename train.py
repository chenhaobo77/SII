import os
import json
import argparse
import shutil
import random
import numpy as np
import torch
from scipy.stats import gaussian_kde, entropy
from sklearn.manifold import TSNE

import dataloaders
import models as mm
import math
from torchvision import models, transforms
from PIL import Image
import torch
import os
from utils import Logger
from trainer import Trainer
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
from utils.losses import abCE_loss, CE_loss, consistency_weight, FocalLoss, softmax_helper, get_alpha
from sklearn.decomposition import PCA
import torch.multiprocessing as mp


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume):
    mp.set_start_method('spawn')
    train_logger = Logger()

    # DATA LOADERS
    config['train_supervised']['percnt_lbl'] = config["sup_percent"]
    config['train_unsupervised']['percnt_lbl'] = config["unsup_percent"]
    config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']
    supervised_loader = dataloaders.CDDataset(config['train_supervised'])
    unsupervised_loader = dataloaders.CDDataset(config['train_unsupervised'])
    val_loader = dataloaders.CDDataset(config['val_loader'])
    iter_per_epoch = len(unsupervised_loader)


    torch.cuda.empty_cache()

    # SUPERVISED LOSS
    if config['model']['sup_loss'] == 'CE':
        sup_loss = CE_loss
    elif config['model']['sup_loss'] == 'FL':
        alpha = get_alpha(supervised_loader)  # calculare class occurences

        sup_loss = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha, gamma=2.0, smooth=1e-5)
    else:
        sup_loss = abCE_loss(iters_per_epoch=iter_per_epoch, epochs=config['trainer']['epochs'],
                             num_classes=val_loader.dataset.num_classes)

    # MODEL
    rampup_ends = int(config['ramp_up'] * config['trainer']['epochs'])

    cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], iters_per_epoch=len(unsupervised_loader),
                                      rampup_ends=rampup_ends)

    model = mm.Consistency_ResNet50_CD(num_classes=val_loader.dataset.num_classes, conf=config['model'],
                                       sup_loss=sup_loss, cons_w_unsup=cons_w_unsup,
                                       weakly_loss_w=config['weakly_loss_w'], use_weak_lables=config['use_weak_lables'])
    print(f'\n{model}\n')

    # TRAINING
    trainer = Trainer(
        model=model,
        resume=resume,
        config=config,
        supervised_loader=supervised_loader,
        unsupervised_loader=unsupervised_loader,
        val_loader=val_loader,
        iter_per_epoch=iter_per_epoch,
        train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config_LEVIR-sup_WHU-unsup.json', type=str,
                        help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local', action='store_true', default=False)
    args = parser.parse_args()

    config = json.load(open(args.config))
    torch.backends.cudnn.benchmark = True
    main(config, args.resume)
