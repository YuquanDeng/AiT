import os
import os
import argparse
import random
import math
import torch
import os
import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
from torchvision import transforms as T
from torch.utils.data import Dataset
import torch.distributed as dist

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from timm.scheduler.cosine_lr import CosineLRScheduler
from mmcv.utils import Config, DictAction
from datetime import datetime

from utils.train_api import save_model
from utils.logger import get_logger

from dataset.custom import CustomDataset
from utils.metrics import eval_depth, cropping_img
from models.VQVAE import VQVAE
import json

import pdb;pdb.set_trace()

image_size = 320

val_dataset = CustomDataset(
        data_path=[
            # "/data/input/jiafei/datasets/Full_OXE/pose_depth/bc_z",
            # "/data/input/jiahttps://files.slack.com/files-pri/T0608KCRH-F08KHR4RR2S/screenshot_2025-03-27_at_7.19.40_pm.pngfei/datasets/Full_OXE/pose_depth/bridge_dataset",
            # "/data/input/jiafei/datasets/Full_OXE/pose_depth/fractal20220817_data"
            # "/data/input/jiafei/GroundedVLA/data/libero/libero_10_no_noops"
            # "/data/input/jiafei/hqfang/AiT/vae/test.png"
            # "/data/input/jiafei/GroundedVLA/data/libero/libero_spatial_no_noops"
            
        ],
        crop_size=(image_size, image_size),
        single_img=False)

# val_dataset = CustomDataset(data_path='/gscratch/krishna/mahtab/depth/only_depth/' , crop_size=(image_size, image_size))


val_data_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=None,
        batch_size=1,
        num_workers=1,
    )

cfg_model = dict(
    image_size=320,
    num_resnet_blocks=2,
    downsample_ratio=32,
    num_tokens=128,
    codebook_dim=512,
    hidden_dim=16,
    use_norm=False,
    channels=1,
    train_objective='regression',
    max_value=10.,
    residul_type='v1',
    loss_type='mse',
)
vae = VQVAE(
    **cfg_model,
    ).cuda()

model_path = "/data/input/jiafei/hqfang/AiT/vae/outputs_oxe_320_20epoch/vae-final.pt"
ckpt = torch.load(model_path)['weights']
if 'module' in list(ckpt.keys())[0]:
    new_ckpt = {}
    for key in list(ckpt.keys()):
        ## remove module.
        new_key = key[7:]
        new_ckpt[new_key] = ckpt[key]
    vae.load_state_dict(
        new_ckpt,
    )
else:
    vae.load_state_dict(
        ckpt,
    )

from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
vae.eval()
save_path = "/data/input/jiafei/GroundedVLA/data/libero/depth/perception_tokens_libero_spatial_no_noops.json"
if os.path.exists(save_path):
    # final_codes = np.load(save_path, allow_pickle=True).item()

    with open(save_path, "r") as file:
        final_codes = json.load(file)  # Converts JSON into a Python dictionary
else:
    final_codes = {}
with torch.no_grad():
    counter = 0
    for iter, (image, _, path) in tqdm(enumerate(val_data_loader)):
        # if path[0] in final_codes:
        #     continue
        
        codes = vae(
                img=image.cuda(),return_indices= True
        )

        # print(codes.shape)

        codes = codes[0].flatten().detach().cpu().tolist()
        assert len(codes) == 100
        depth_string = "<DEPTH_START>"
        depth_string += "".join([f"<DEPTH_{num}>" for num in codes ])
        depth_string += "<DEPTH_END>"
        final_codes[path[0]] = depth_string
        # print(path[0])
        # print(codes)
        # print(depth_string)

        # break
     
    # np.save(save_path, final_codes)   
    
    with open(save_path, "w") as file:
        json.dump(final_codes, file, indent=4)
        
            
print(counter)