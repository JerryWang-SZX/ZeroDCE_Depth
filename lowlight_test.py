import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    scale_factor = 12
    data_lowlight = Image.open(image_path)

    data_lowlight = (np.asarray(data_lowlight) / 255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()

    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[0:h, 0:w, :]
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net = model.enhance_net_nopool(scale_factor).cuda()
    DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch99.pth'))
    start = time.time()
    enhanced_image, params_maps = DCE_net(data_lowlight)

    end_time = (time.time() - start)

    print(end_time)
    # Save into result directory: result_Zero_DCE++/real/<original_filename>
    base_name = os.path.basename(image_path)
    result_dir = os.path.join('result_Zero_DCE++', 'real')
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, base_name)
    torchvision.utils.save_image(enhanced_image, result_path)
    return end_time


if __name__ == '__main__':

    with torch.no_grad():

        filePath = 'bdd100k-night-v3.yolov11/test/images'
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
        image_list = []
        for patt in patterns:
            image_list.extend(glob.glob(os.path.join(filePath, patt)))
        image_list = sorted(image_list)

        sum_time = 0
        for image in image_list:
            print(image)
            sum_time = sum_time + lowlight(image)

        print(sum_time)

