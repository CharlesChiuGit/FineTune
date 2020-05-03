import torch


def normalize(coord):
    temp = [0, 0, 0]
    temp[0] = coord[0]
    temp[1] = coord[1]
    temp[2] = coord[2]
    l = (temp[0] ** 2 + temp[1] ** 2 + temp[2] ** 2) ** 0.5
    temp[0] /= l
    temp[1] /= l
    temp[2] /= l
    return temp


def crossf(a, b):
    temp = [0, 0, 0]
    temp[0] = a[1] * b[2] - a[2] * b[1]
    temp[1] = a[2] * b[0] - a[0] * b[2]
    temp[2] = a[0] * b[1] - a[1] * b[0]
    return temp


def camera_direction(c_x, c_y, c_z, p_x, p_y, p_z):
    """
    Set direction of the camera.
    """
    forward = [0, 0, 0]
    up = [0, 0, 0]
    camera_position = [c_x, c_y, c_z]
    optical_axis_position = [p_x, p_y, p_z]
    for i in range(3):
        forward[i] = optical_axis_position[i] - camera_position[i]
        # up[i] = random.uniform(0, 1)
    up = [0, 1, 0]

    norm_forward = normalize(forward)
    side = normalize(crossf(norm_forward, up))
    up = normalize(crossf(side, norm_forward))

    return up[0], up[1], up[2]


def rgb2gray(rgb, device):
    return torch.matmul(rgb[..., :3].to(device), torch.FloatTensor([0.2989, 0.5870, 0.1140]).to(device))


def get_cpu_image(gpu_image):
    cpu_image = gpu_image[0, ..., :3].detach().squeeze().cpu().numpy()

    return cpu_image