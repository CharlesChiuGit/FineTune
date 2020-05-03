import torch
import copy
import os
import cv2
import numpy as np

from net import Model
from config import FOV, LR, EPOCH, DEVICE, OBJ_filename
from helper import camera_direction
from renderer.io import load_moon_mesh
from renderer import build_renderer, render_single_image

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def fine_tuning(target_image_path, init_pos):
    target_image = cv2.imread(target_image_path)
    image_size = target_image.shape[0]
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    mesh = load_moon_mesh(OBJ_filename)
    renderer = build_renderer(image_size)

    model = Model(mesh=mesh, renderer=renderer, image_ref=target_image, init_pos=init_pos).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_loss = 1000000
    best_position = [0, 0, 0]

    for i in range(EPOCH):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

        if best_loss > loss.item():
            best_loss = copy.deepcopy(loss.item())
            best_position = copy.deepcopy([model.dist.cpu().item(),
                                           model.elev.cpu().item(),
                                           model.azim.cpu().item(),
                                           model.p.cpu()[0][0].item(),
                                           model.p.cpu()[0][1].item(),
                                           model.p.cpu()[0][2].item(),
                                           model.u.cpu()[0][0].item(),
                                           model.u.cpu()[0][1].item(),
                                           model.u.cpu()[0][2].item()])
            # print("Best Loss:{}, Best Pos:{}".format(best_loss, best_position))

        if loss.item() < 0.05:  # ssim Loss
            break

    return best_position
