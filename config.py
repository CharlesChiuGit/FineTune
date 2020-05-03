import os
import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

MOON_RADIUS = 1.74590086  # 1000 km
FOV = 120
LR = 1e-3
# LR = 0.05
COORD = "Spherical"  # "Cartesian" "Spherical"
LOSS = "ssim"  # "MSE" "cos_sim" "ssim"
EXP_NAME = "{}_{}".format(COORD, LOSS)
EPOCH = 500

# Set the cuda device
DEVICE = torch.device("cuda:{}".format(0))
torch.cuda.set_device(DEVICE)

# Set paths
DATA_DIR = "../../Moon_8K_model"
OBJ_filename = os.path.join(DATA_DIR, "Moon_8K.obj")


def check_directory(directory):
    directory_path = os.path.join(".", directory)
    if not os.path.exists(directory_path):
        # logging.info('Create directory {}'.format(directory))
        os.makedirs(directory_path)


check_directory(EXP_NAME)
