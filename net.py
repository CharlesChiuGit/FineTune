import torch
import torch.nn as nn
import numpy as np
from pytorch3d.renderer import look_at_view_transform
import pytorch_ssim


class Model(nn.Module):
    def __init__(self, mesh, renderer, image_ref, init_pos):
        super().__init__()
        self.mesh = mesh
        self.device = mesh.device
        self.renderer = renderer

        image_ref = torch.from_numpy(image_ref.astype(np.float32))  # (1, SIZE, SIZE, 4)
        self.register_buffer('image_ref', image_ref)
        self.gray_image_ref = self.image_ref[..., :3].unsqueeze(0).to(self.device)

        self.dist = nn.Parameter(
            torch.from_numpy(np.array(init_pos[0], dtype=np.float32)).to(mesh.device), requires_grad=True)
        self.elev = nn.Parameter(
            torch.from_numpy(np.array(init_pos[1], dtype=np.float32)).to(mesh.device), requires_grad=True)
        self.azim = nn.Parameter(
            torch.from_numpy(np.array(init_pos[2], dtype=np.float32)).to(mesh.device), requires_grad=True)

        self.p = nn.Parameter(torch.from_numpy(np.array([[init_pos[3], init_pos[4], init_pos[5]]],
                                                        dtype=np.float32)).to(mesh.device), requires_grad=False)

        self.u = nn.Parameter(torch.from_numpy(np.array([[init_pos[6], init_pos[7], init_pos[8]]],
                                                        dtype=np.float32)).to(mesh.device), requires_grad=False)

    def forward(self):
        R, T = look_at_view_transform(dist=self.dist, elev=self.elev, azim=self.azim, at=self.p, up=self.u,
                                      device=self.device)

        predict_image = self.renderer(meshes_world=self.mesh.clone(), R=R, T=T)

        ssim_loss = pytorch_ssim.SSIM(window_size=11)
        predict_image = predict_image[..., :3]
        loss = 1 - ssim_loss(self.gray_image_ref, predict_image)

        return loss
