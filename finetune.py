import torch
import copy
import os
# rendering components
from pytorch3d.renderer import look_at_view_transform

from net import Model
from config import FOV, LR, EPOCH, DEVICE, OBJ_filename
from helper import camera_direction
from renderer.io import load_moon_mesh
from renderer import build_renderer, render_single_image

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def fine_tuning(target_pos, init_pos):
    mesh = load_moon_mesh(OBJ_filename)
    renderer = build_renderer()

    u_x, u_y, u_z = camera_direction(target_pos[0][0], target_pos[0][1], target_pos[0][2], 0, 0, 0)
    R, T = look_at_view_transform(eye=target_pos,
                                  at=((0, 0, 0),),
                                  up=((u_x, u_y, u_z),), device=DEVICE)
    target_image = renderer(meshes_world=mesh.clone(), R=R, T=T)

    model = Model(mesh=mesh, renderer=renderer, image_ref=target_image.cpu().numpy(), init_pos=init_pos).to(DEVICE)

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
                                           model.azim.cpu().item()])
            # print("Best Loss:{}, Best Pos:{}".format(best_loss, best_position))

        if loss.item() < 0.05:  # ssim Loss
            break

    return best_position
