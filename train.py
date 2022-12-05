import numpy as np
import json
from PIL import Image
import torch
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
import random

import tensorflow_datasets as tfds
import datasets

from sde_lib import VPSDE
import wandb

from tqdm import tqdm


from configs.vp import cifar10_ddpmpp_continuous as configs  

#dataset
batch_size = 512

config = configs.get_config()
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

training_dataset, validation_dataset, _ = datasets.get_dataset(config, evaluation=True) # evalutaion = True -> repeat = 1
scaler = datasets.get_data_scaler(config)

#train code

def train_model(model, criterion, optimizer, init_epoch=0, num_epochs=25, save_model_every = 10, model_name = 'one-output', lr = 0.01):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_idx = 0
    best_loss = 0.0
    
    vpsde = VPSDE()
    
    device = 'cuda:0'
        
    for epoch in range(init_epoch,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dl = training_dataset
            else:
#                model.eval()   # Set model to evaluate mode
                dl = validation_dataset
            running_loss, num_cnt = 0.0, 0
            
            with torch.set_grad_enabled(phase == 'train'):
                # Iterate over data.
                pbar = tqdm(enumerate(dl),total=len(dl))
                for i, batch in pbar:

                    # get data from dataloader
                    image_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
                    image_batch = image_batch.permute(0, 3, 1, 2)
                    
                    # (0,1) -> (-1, 1)
                    image_batch = scaler(image_batch)

                    optimizer.zero_grad()

                    t = torch.rand(image_batch.shape[0])
                    t = t.to(device)
                    mean, std = vpsde.marginal_prob(image_batch,t)
                    std = std.reshape(-1,1,1,1)
                    image_batch = mean + std*torch.rand_like(mean)
                    outputs = model(image_batch)
                    
                    if model_name == 'softmax':
                        outputs = torch.softmax(outputs,dim=1)
                        outputs = torch.transpose(outputs,0,1)
                        outputs = outputs[0]
                    if model_name == 'sigmoid':
                        outputs = torch.sigmoid(outputs)
                    t = t.reshape(-1,1)
                    loss = criterion(outputs, t)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * image_batch.shape[0]
                    num_cnt += image_batch.shape[0]
                    pbar.set_postfix({'loss' : '{:.10f}'.format(loss.item())})
            
            epoch_loss = float(running_loss / num_cnt)
            
            #wandb
            if phase == 'train':
                wandb.log({"train_loss": epoch_loss})
            else:
                wandb.log({"valid_loss": epoch_loss})

            print('epoch: {} phase: {} Loss: {:.10f}'.format(epoch ,phase, epoch_loss))
        
            # deep copy the model
            if phase == 'valid' and epoch_loss < best_loss:
                best_idx = epoch
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            
        if epoch !=0 and epoch % save_model_every == 0:
            torch.save(model.state_dict(), 'results/'+model_name+'/pretrained_'+str(lr)+'/'+str(epoch//save_model_every)+'.pt')
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.10f' %(best_idx, best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'results/'+model_name+'/pretrained_'+str(lr)+'/'+'best.pt')
    print('model saved')


#model

pretrained = True

model_list = ['one-output', 'softmax', 'sigmoid']

model_name = model_list[0]

if pretrained:
    from torchvision.models import EfficientNet_V2_L_Weights
    model = torchvision.models.efficientnet_v2_l(num_classes = 1000, weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(1280,(2 if model_name=='softmax' else 1),bias = True)
else:
    model = torchvision.models.efficientnet_v2_l(num_classes = (2 if model_name=='softmax' else 1))
device = 'cuda:0'
model.to(device)
criterion = nn.MSELoss(reduction = 'mean')

lr = 0.0001

opt = torch.optim.Adam(model.parameters(),lr=lr)

#wandb

wandb.init(project="score-sde", entity="inooni")
wandb.config = {
  "name": model_name,
  "pre-trained" : pretrained,
  "learning_rate": lr,
  "epochs": 5000,
  "batch_size": batch_size
}

train_model(model, criterion, opt, init_epoch = 0, num_epochs=5000, save_model_every = 100, model_name = model_name, lr = lr)

