import numpy as np
from torchvision import transforms
import os
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import cv2


# tensor to img
def tensor2img(img):
    #assert len(img) == 3
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().squeeze().astype(np.uint8)),
    ])

    imgs = reverse_transforms(img)

    return imgs


def tensorboard_writer(timestr):
    log_path = os.path.join('logs', timestr)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    return writer

def logger(timestr):
    log_dir = os.path.join('logs',timestr)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, "log.txt")
    fw = open(log_path, "a+")
    return fw


def save_model(model, epoch,timestr):
    dir_path = os.path.join("weight",timestr)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    ckpt_name = "epoch_" + str(epoch) + ".pt"
    ckpt_path = os.path.join(dir_path, ckpt_name)
    torch.save(model.state_dict(), ckpt_path)
