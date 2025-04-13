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
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing RGB images")
    cfg = parser.parse_args()
    
    # input_path = "/home/nil/manipulation/dataset/pick_apple_100_328/rgb/pick_apple_2/train/episode_0"
    input_path = cfg.input_folder
    
    # load dataset
    image_size = 320
    val_dataset = CustomDataset(
            data_path=[input_path],
            crop_size=(image_size, image_size),
            single_img=False)

    val_data_loader = torch.utils.data.DataLoader(
            val_dataset, sampler=None,
            batch_size=1,
            num_workers=1,
        )

    # load Model
    cfg_model = dict(
        image_size=image_size, # TODO
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

    model_path = "/home/nil/manipulation/groundvla/libs/AiT/checkpoints/vae-final.pt"
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

    # Generate depth code
    vae.eval()
    # save_path = "/home/nil/manipulation/groundvla/libs/AiT/vae/tmp.json"
    save_path = input_path.replace("rgb", "depth") + ".json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if os.path.exists(save_path):
        with open(save_path, "r") as file:
            final_codes = json.load(file)
    else:
        final_codes = {}
        
    with torch.no_grad():
        counter = 0
        for iter, (image, _, path) in tqdm(enumerate(val_data_loader)):
            
            codes = vae(
                    img=image.cuda(),return_indices= True
            )

            codes = codes[0].flatten().detach().cpu().tolist()
            print(codes)
            print("len(codes):", len(codes))
            assert len(codes) == 100
            depth_string = "<DEPTH_START>"
            depth_string += "".join([f"<DEPTH_{num}>" for num in codes ])
            depth_string += "<DEPTH_END>"
            
            # TODO: make sure there is no double slash for path[0
            sanitized_path = os.path.normpath(path[0])
            final_codes[sanitized_path] = depth_string  
        
        with open(save_path, "w") as file:
            json.dump(final_codes, file, indent=4)
    print("Saved under:", save_path)
    print(counter)


if __name__ == "__main__":
    main()