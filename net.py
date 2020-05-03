import torch
import torch.nn as nn
import numpy as np
from pytorch3d.renderer import look_at_view_transform
import pytorch_ssim
from config import SIZE
from helper import camera_direction


class Model(nn.Module):
    def __init__(self, mesh, renderer, image_ref, init_pos):
        super().__init__()
        self.mesh = mesh
        self.device = mesh.device
        self.renderer = renderer

        image_ref = torch.from_numpy(image_ref.astype(np.float32))  # (1, SIZE, SIZE, 4)
        self.register_buffer('image_ref', image_ref)
        self.gray_image_ref = self.image_ref[..., :3].to(self.device).reshape((1, 3, SIZE, SIZE))

        self.dist = nn.Parameter(
            torch.from_numpy(np.array(init_pos[0], dtype=np.float32)).to(mesh.device), requires_grad=True)
        self.elev = nn.Parameter(
            torch.from_numpy(np.array(init_pos[1], dtype=np.float32)).to(mesh.device), requires_grad=True)
        self.azim = nn.Parameter(
            torch.from_numpy(np.array(init_pos[2], dtype=np.float32)).to(mesh.device), requires_grad=True)

    def forward(self):
        # u_x, u_y, u_z = camera_direction(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2],
        #                                  at_point[0], at_point[1], at_point[2])
        R, T = look_at_view_transform(dist=self.dist, elev=self.elev, azim=self.azim, device=self.device)

        predict_image = self.renderer(meshes_world=self.mesh.clone(), R=R, T=T)

        ssim_loss = pytorch_ssim.SSIM(window_size=11)
        predict_image = predict_image[..., :3].reshape((1, 3, SIZE, SIZE))
        loss = 1 - ssim_loss(self.gray_image_ref, predict_image)

        return loss
