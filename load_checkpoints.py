from distutils.log import debug
from email.policy import default
import os
import logging
from datetime import datetime
import argparse
import shutil
from tkinter.messagebox import NO
import zipfile
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import albumentations as A

import sys
sys.path.append('.../NIR-ISL2021master/')
from datasets import eyeDataset
from models import EfficientUNet
from loss import Make_Criterion
from evaluation import evaluate_loc

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

loading_net = EfficientUNet(num_classes=2).to(device)
