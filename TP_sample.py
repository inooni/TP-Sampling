from dataclasses import dataclass, field
import io
import csv
import numpy as np
import importlib
import os
import functools
import itertools
import torch
import torchvision
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tensorflow_datasets as tfds
import tqdm
import io
import likelihood
import controllable_generation
from utils import restore_checkpoint


import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets


sde = 'VPSDE'
if sde.lower() == 'vpsde':
  from configs.vp import cifar10_ddpmpp_continuous as configs  
  ckpt_filename = "exp/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth"
  config = configs.get_config()
  sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  sampling_eps = 1e-3

sde.N = 10

tp_model = None
pretrained = True
model_name = 'pretrained_0.0001'

if pretrained:
    from torchvision.models import EfficientNet_V2_L_Weights
    tp_model = torchvision.models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    tp_model.classifier[1] = nn.Linear(1280,1,bias = True)
else:
    tp_model = torchvision.models.efficientnet_v2_l(num_classes = (2 if model_name=='softmax' else 1))

tp_model.load_state_dict(torch.load('results/one-output/'+model_name+'/25.pt'))
device = 'cuda:0'
tp_model.to(device)


batch_size =   512
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())


img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)
predictor = EulerMaruyamaPredictor
corrector = None
snr = 0.16 
n_steps =  1
probability_flow = False 
sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device, tp_model = tp_model)

import os

sample_folder = str(f'sample/'+model_name+'_'+str(sde.N)+'step_direct/')

if not os.path.exists(sample_folder):
  os.makedirs(sample_folder)

sample_num = 60000
for b in range(sample_num//batch_size + 1):
    print(b)
    x, n = sampling_fn(score_model)
    for i, image in enumerate(x):
        torchvision.utils.save_image(image, sample_folder + f'{b*batch_size+i}.png')
    
